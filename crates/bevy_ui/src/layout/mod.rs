#[cfg(feature = "ghost_nodes")]
use crate::experimental::GhostNode;
use crate::{
    experimental::{UiChildren, UiRootNodes},
    ui_transform::{UiGlobalTransform, UiTransform},
    ComputedNode, ComputedUiRenderTargetInfo, ContentSize, Display, IgnoreScroll,
    IndependentLayout, LayoutConfig, Node, Outline, OverflowAxis, ScrollPosition,
};
use bevy_ecs::{
    change_detection::{DetectChanges, DetectChangesMut},
    entity::{Entity, EntityHashMap, EntityHashSet},
    hierarchy::{ChildOf, Children},
    lifecycle::RemovedComponents,
    query::{Added, Changed, Or, With},
    system::{Local, Query, ResMut, SystemParam},
    world::Ref,
};

use bevy_math::{Affine2, Vec2};
use bevy_sprite::BorderRect;
use thiserror::Error;
use ui_surface::UiSurface;

use bevy_text::ComputedTextBlock;

use bevy_text::FontCx;

mod convert;
pub mod debug;
pub mod ui_surface;

pub struct LayoutContext {
    pub scale_factor: f32,
    pub physical_size: Vec2,
}

impl LayoutContext {
    pub const DEFAULT: Self = Self {
        scale_factor: 1.0,
        physical_size: Vec2::ZERO,
    };
    /// Create a new [`LayoutContext`] from the window's physical size and scale factor
    #[inline]
    const fn new(scale_factor: f32, physical_size: Vec2) -> Self {
        Self {
            scale_factor,
            physical_size,
        }
    }
}

#[cfg(test)]
impl LayoutContext {
    pub const TEST_CONTEXT: Self = Self {
        scale_factor: 1.0,
        physical_size: Vec2::new(1000.0, 1000.0),
    };
}

impl Default for LayoutContext {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("Invalid hierarchy")]
    InvalidHierarchy,
    #[error("Taffy error: {0}")]
    TaffyError(taffy::tree::TaffyError),
}

/// What an [`IndependentLayout`] root was last placed against: its parent's
/// global transform, size and scroll position, and whether an ancestor was
/// hidden. A root none of whose own nodes changed is placed again only when
/// one of these did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndependentPlacement {
    parent_transform: Affine2,
    parent_size: Vec2,
    parent_scroll: Vec2,
    hidden: bool,
}

/// The queries [`ui_layout_system`] finds its **dirty layout roots** with.
///
/// A layout root is a root [`Node`] or an [`IndependentLayout`] node. A change
/// to a node dirties the layout root it belongs to — the nearest ancestor (or
/// itself) that is one — and only dirty roots are synced, laid out and
/// placed.
#[derive(SystemParam)]
pub struct LayoutRoots<'w, 's> {
    /// Every independent layout root.
    independent: Query<'w, 's, (), With<IndependentLayout>>,
    /// Nodes that became independent this frame.
    added_independent: Query<'w, 's, Entity, Added<IndependentLayout>>,
    /// Nodes that stopped being independent.
    removed_independent: RemovedComponents<'w, 's, IndependentLayout>,
    /// The hierarchy, walked up to find a node's layout root.
    parents: Query<'w, 's, &'static ChildOf>,
    /// UI nodes, for the walk and the hidden-ancestor check.
    nodes: Query<'w, 's, &'static Node>,
    /// Nodes whose children changed.
    changed_children: Query<'w, 's, Entity, (Changed<Children>, With<Node>)>,
    /// Nodes moved to another parent: their new layout root is dirty (their
    /// old one is, through the old parent's children changing).
    changed_parent: Query<'w, 's, Entity, (Changed<ChildOf>, With<Node>)>,
    /// Nodes un-parented: each is now a root of its own, to be laid out.
    removed_parent: RemovedComponents<'w, 's, ChildOf>,
    /// Nodes whose geometry inputs (not their layout) changed.
    changed_geometry: Query<
        'w,
        's,
        Entity,
        (
            With<Node>,
            Or<(
                Changed<ScrollPosition>,
                Changed<UiTransform>,
                Changed<Outline>,
                Changed<LayoutConfig>,
                Changed<IgnoreScroll>,
            )>,
        ),
    >,
}

impl LayoutRoots<'_, '_> {
    /// The layout root `entity` belongs to: the nearest of itself and its
    /// ancestors that is an [`IndependentLayout`] node or has no UI parent.
    fn root_of(&self, entity: Entity, memo: &mut EntityHashMap<Entity>) -> Entity {
        let mut walked = Vec::new();
        let mut current = entity;
        let root = loop {
            if let Some(root) = memo.get(&current) {
                break *root;
            }
            walked.push(current);
            if self.independent.contains(current) {
                break current;
            }
            match self.parents.get(current) {
                Ok(parent) if self.nodes.contains(parent.parent()) => current = parent.parent(),
                _ => break current,
            }
        };
        for node in walked {
            memo.insert(node, root);
        }
        root
    }

    /// The UI parent of `entity`, if it has one.
    fn parent_of(&self, entity: Entity) -> Option<Entity> {
        self.parents
            .get(entity)
            .ok()
            .map(ChildOf::parent)
            .filter(|parent| self.nodes.contains(*parent))
    }

    /// Whether any strict UI ancestor of `entity` has [`Display::None`].
    fn ancestor_hidden(&self, entity: Entity) -> bool {
        let mut current = entity;
        while let Some(parent) = self.parent_of(current) {
            if self
                .nodes
                .get(parent)
                .is_ok_and(|node| node.display == Display::None)
            {
                return true;
            }
            current = parent;
        }
        false
    }

    /// How many UI ancestors `entity` has.
    fn depth(&self, entity: Entity) -> usize {
        let mut depth = 0;
        let mut current = entity;
        while let Some(parent) = self.parent_of(current) {
            depth += 1;
            current = parent;
        }
        depth
    }
}

/// Updates the UI's layout tree, computes the new layout geometry and then updates the sizes and transforms of the UI nodes.
///
/// Only **dirty layout roots** are processed: a layout root is a root [`Node`]
/// or an [`IndependentLayout`] node, and a change to any node (its `Node`, its
/// measure, its render target, its children, or one of the inputs to its
/// placement) dirties the root it belongs to. An independent root whose own
/// nodes did not change is placed again only when its parent moved, resized,
/// scrolled, or an ancestor was hidden or shown.
pub fn ui_layout_system(
    mut ui_surface: ResMut<UiSurface>,
    ui_root_node_query: UiRootNodes,
    ui_children: UiChildren,
    mut node_query: Query<(
        Entity,
        Ref<Node>,
        &mut ContentSize,
        Ref<ComputedUiRenderTargetInfo>,
    )>,
    added_node_query: Query<(), Added<Node>>,
    mut node_update_query: Query<(
        &mut ComputedNode,
        &UiTransform,
        &mut UiGlobalTransform,
        &Node,
        Option<&LayoutConfig>,
        Option<&Outline>,
        Option<&ScrollPosition>,
        Option<&IgnoreScroll>,
    )>,
    mut buffer_query: Query<&mut ComputedTextBlock>,
    mut font_system: ResMut<FontCx>,
    mut removed_children: RemovedComponents<Children>,
    mut removed_nodes: RemovedComponents<Node>,
    #[cfg(feature = "ghost_nodes")] mut removed_ghost_nodes: RemovedComponents<GhostNode>,
    #[cfg(feature = "ghost_nodes")] added_ghost_node_query: Query<Entity, Added<GhostNode>>,
    #[cfg(feature = "ghost_nodes")] ghost_node_query: Query<(), With<GhostNode>>,
    mut roots: LayoutRoots,
    mut placed: Local<EntityHashMap<IndependentPlacement>>,
) {
    // Every node whose layout or placement changed, mapped to its layout root
    // below.
    let mut changed_nodes: Vec<Entity> = Vec::new();

    // Sync Node and ContentSize to Taffy for all nodes
    node_query
        .iter_mut()
        .for_each(|(entity, node, mut content_size, computed_target)| {
            if computed_target.is_changed() || node.is_changed() || content_size.is_changed() {
                changed_nodes.push(entity);
                let layout_context = LayoutContext::new(
                    computed_target.scale_factor,
                    computed_target.physical_size.as_vec2(),
                );
                if content_size.is_changed() && content_size.measure.is_none() {
                    ui_surface.try_remove_node_context(entity);
                }
                let measure = content_size.bypass_change_detection().measure.take();
                ui_surface.upsert_node(&layout_context, entity, &node, measure);
            }
        });

    // update and remove children
    #[cfg(not(feature = "ghost_nodes"))]
    {
        for entity in removed_children.read() {
            ui_surface.try_remove_children(entity);
            changed_nodes.push(entity);
        }
    }

    #[cfg(feature = "ghost_nodes")]
    {
        // Collect the closest non-ghost ancestors of entities that had `GhostNode` added or removed since last layout update.
        ui_surface.dirty_ghost_children_scratch.clear();
        for entity in added_ghost_node_query
            .iter()
            .chain(removed_ghost_nodes.read())
        {
            if let Some(parent) = ui_children.get_parent(entity) {
                ui_surface.dirty_ghost_children_scratch.insert(parent);
            }
        }

        for entity in removed_children.read() {
            ui_surface.try_remove_children(entity);
            changed_nodes.push(entity);
            if ghost_node_query.contains(entity)
                && let Some(parent) = ui_children.get_parent(entity)
            {
                ui_surface.dirty_ghost_children_scratch.insert(parent);
            }
        }
    }

    // clean up removed nodes after syncing children to avoid potential panic (invalid SlotMap key used)
    ui_surface.remove_entities(
        removed_nodes
            .read()
            .filter(|entity| !node_query.contains(*entity)),
    );

    placed.retain(|entity, _| node_query.contains(*entity));

    // A node gaining or losing `IndependentLayout` leaves or rejoins its
    // parent's layout children: the parent's children are synced again, and
    // both the parent's root and the node's own are dirty.
    let mut resync_children_of = EntityHashSet::default();
    let independence_changed: Vec<Entity> = roots
        .added_independent
        .iter()
        .chain(roots.removed_independent.read())
        .filter(|entity| node_query.contains(*entity))
        .collect();
    for entity in independence_changed {
        changed_nodes.push(entity);
        if let Some(parent) = roots.parent_of(entity) {
            resync_children_of.insert(parent);
            changed_nodes.push(parent);
        }
    }
    changed_nodes.extend(roots.changed_children.iter());
    changed_nodes.extend(roots.changed_parent.iter());
    let unparented: Vec<Entity> = roots.removed_parent.read().collect();
    changed_nodes.extend(unparented);
    changed_nodes.extend(roots.changed_geometry.iter());
    #[cfg(feature = "ghost_nodes")]
    changed_nodes.extend(ui_surface.dirty_ghost_children_scratch.iter().copied());

    let mut memo = EntityHashMap::default();
    let dirty_roots: EntityHashSet = changed_nodes
        .into_iter()
        .filter(|entity| node_query.contains(*entity))
        .map(|entity| roots.root_of(entity, &mut memo))
        .collect();

    // The ordinary roots first: an independent root is placed against its
    // parent's final geometry, so its parent's tree goes before it.
    let mut pending_independent: Vec<Entity> = dirty_roots
        .iter()
        .copied()
        .filter(|root| roots.independent.contains(*root) && roots.parent_of(*root).is_some())
        .collect();
    for ui_root_entity in ui_root_node_query.iter() {
        if !dirty_roots.contains(&ui_root_entity) {
            continue;
        }
        update_children_recursively(
            &mut ui_surface,
            &ui_children,
            &added_node_query,
            &roots,
            &resync_children_of,
            ui_root_entity,
        );

        let (_, _, _, computed_target) = node_query.get(ui_root_entity).unwrap();

        ui_surface.compute_layout(
            ui_root_entity,
            computed_target.physical_size,
            &mut buffer_query,
            &mut font_system,
        );

        update_uinode_geometry_recursive(
            ui_root_entity,
            &mut ui_surface,
            true,
            computed_target.physical_size().as_vec2(),
            Affine2::IDENTITY,
            &mut node_update_query,
            &ui_children,
            computed_target.scale_factor.recip(),
            Vec2::ZERO,
            Vec2::ZERO,
            &roots,
            &mut pending_independent,
        );
    }

    // Then the independent roots, shallowest first, each dirty one laid out
    // and each one whose parent moved (or was hidden or shown) placed again.
    // Placing one can surface independent roots nested inside it, which are
    // deeper and so come after it.
    let mut done = EntityHashSet::default();
    while !pending_independent.is_empty() {
        pending_independent.sort_by_key(|entity| core::cmp::Reverse(roots.depth(*entity)));
        pending_independent.dedup();
        let Some(root) = pending_independent.pop() else {
            break;
        };
        if !done.insert(root) {
            continue;
        }
        let Some(parent) = roots.parent_of(root) else {
            continue;
        };
        let Ok((parent_node, _, parent_transform, ..)) = node_update_query.get(parent) else {
            continue;
        };
        let placement = IndependentPlacement {
            parent_transform: **parent_transform,
            parent_size: parent_node.size,
            parent_scroll: parent_node.scroll_position,
            hidden: roots.ancestor_hidden(root),
        };
        let dirty = dirty_roots.contains(&root);
        if !dirty && placed.get(&root) == Some(&placement) {
            continue;
        }
        placed.insert(root, placement);

        if placement.hidden {
            collapse_subtree(root, &mut node_update_query, &ui_children);
            continue;
        }

        let Ok((_, _, _, computed_target)) = node_query.get(root) else {
            continue;
        };
        let physical_size = computed_target.physical_size;
        let scale_factor = computed_target.scale_factor;
        if dirty {
            update_children_recursively(
                &mut ui_surface,
                &ui_children,
                &added_node_query,
                &roots,
                &resync_children_of,
                root,
            );
            ui_surface.compute_layout(root, physical_size, &mut buffer_query, &mut font_system);
        }
        update_uinode_geometry_recursive(
            root,
            &mut ui_surface,
            true,
            physical_size.as_vec2(),
            placement.parent_transform,
            &mut node_update_query,
            &ui_children,
            scale_factor.recip(),
            placement.parent_size,
            placement.parent_scroll,
            &roots,
            &mut pending_independent,
        );
    }

    /// Sync the taffy children of every node in `entity`'s layout tree that
    /// needs it, leaving [`IndependentLayout`] children (and their subtrees)
    /// out: they are layout roots of their own.
    fn update_children_recursively(
        ui_surface: &mut UiSurface,
        ui_children: &UiChildren,
        added_node_query: &Query<(), Added<Node>>,
        roots: &LayoutRoots,
        resync_children_of: &EntityHashSet,
        entity: Entity,
    ) {
        let children_changed = ui_children.is_changed(entity)
            || resync_children_of.contains(&entity)
            || ui_children
                .iter_ui_children(entity)
                .any(|child| added_node_query.contains(child));
        #[cfg(feature = "ghost_nodes")]
        let children_changed =
            children_changed || ui_surface.dirty_ghost_children_scratch.contains(&entity);

        if ui_surface.entity_to_taffy.contains_key(&entity)
            && (added_node_query.contains(entity) || children_changed)
        {
            ui_surface.update_children(
                entity,
                ui_children
                    .iter_ui_children(entity)
                    .filter(|child| !roots.independent.contains(*child)),
            );
        }

        for child in ui_children.iter_ui_children(entity) {
            if roots.independent.contains(child) {
                continue;
            }
            update_children_recursively(
                ui_surface,
                ui_children,
                added_node_query,
                roots,
                resync_children_of,
                child,
            );
        }
    }

    /// Give every node in `entity`'s subtree a zero size: the subtree of an
    /// [`IndependentLayout`] root under a hidden ancestor, which as a layout
    /// child would have been laid out at nothing.
    fn collapse_subtree(
        entity: Entity,
        node_update_query: &mut Query<(
            &mut ComputedNode,
            &UiTransform,
            &mut UiGlobalTransform,
            &Node,
            Option<&LayoutConfig>,
            Option<&Outline>,
            Option<&ScrollPosition>,
            Option<&IgnoreScroll>,
        )>,
        ui_children: &UiChildren,
    ) {
        if let Ok((mut node, ..)) = node_update_query.get_mut(entity)
            && (node.size != Vec2::ZERO || node.unrounded_size != Vec2::ZERO)
        {
            node.size = Vec2::ZERO;
            node.unrounded_size = Vec2::ZERO;
        }
        for child in ui_children.iter_ui_children(entity) {
            collapse_subtree(child, node_update_query, ui_children);
        }
    }

    // Returns the combined bounding box of the node and any of its overflowing children.
    fn update_uinode_geometry_recursive(
        entity: Entity,
        ui_surface: &mut UiSurface,
        inherited_use_rounding: bool,
        target_size: Vec2,
        mut inherited_transform: Affine2,
        node_update_query: &mut Query<(
            &mut ComputedNode,
            &UiTransform,
            &mut UiGlobalTransform,
            &Node,
            Option<&LayoutConfig>,
            Option<&Outline>,
            Option<&ScrollPosition>,
            Option<&IgnoreScroll>,
        )>,
        ui_children: &UiChildren,
        inverse_target_scale_factor: f32,
        parent_size: Vec2,
        parent_scroll_position: Vec2,
        roots: &LayoutRoots,
        independent_children: &mut Vec<Entity>,
    ) {
        if let Ok((
            mut node,
            transform,
            mut global_transform,
            style,
            maybe_layout_config,
            maybe_outline,
            maybe_scroll_position,
            maybe_scroll_sticky,
        )) = node_update_query.get_mut(entity)
        {
            let use_rounding = maybe_layout_config
                .map(|layout_config| layout_config.use_rounding)
                .unwrap_or(inherited_use_rounding);

            let Ok((layout, unrounded_size)) = ui_surface.get_layout(entity, use_rounding) else {
                return;
            };

            let layout_size = Vec2::new(layout.size.width, layout.size.height);

            // Taffy layout position of the top-left corner of the node, relative to its parent.
            let layout_location = Vec2::new(layout.location.x, layout.location.y);

            // If IgnoreScroll is set, parent scroll position is ignored along the specified axes.
            let effective_parent_scroll = maybe_scroll_sticky
                .map(|scroll_sticky| parent_scroll_position * Vec2::from(!scroll_sticky.0))
                .unwrap_or(parent_scroll_position);

            // The position of the center of the node relative to its top-left corner.
            let local_center =
                layout_location - effective_parent_scroll + 0.5 * (layout_size - parent_size);

            // only trigger change detection when the new values are different
            if node.size != layout_size
                || node.unrounded_size != unrounded_size
                || node.inverse_scale_factor != inverse_target_scale_factor
            {
                node.size = layout_size;
                node.unrounded_size = unrounded_size;
                node.inverse_scale_factor = inverse_target_scale_factor;
            }

            let content_size = Vec2::new(layout.content_size.width, layout.content_size.height);
            node.bypass_change_detection().content_size = content_size;

            let taffy_rect_to_border_rect = |rect: taffy::Rect<f32>| BorderRect {
                min_inset: Vec2::new(rect.left, rect.top),
                max_inset: Vec2::new(rect.right, rect.bottom),
            };

            node.bypass_change_detection().border = taffy_rect_to_border_rect(layout.border);
            node.bypass_change_detection().padding = taffy_rect_to_border_rect(layout.padding);

            // Compute the node's new global transform
            let mut local_transform = transform.compute_affine(
                inverse_target_scale_factor.recip(),
                layout_size,
                target_size,
            );
            local_transform.translation += local_center;
            inherited_transform *= local_transform;

            if inherited_transform != **global_transform {
                *global_transform = inherited_transform.into();
            }

            // We don't trigger change detection for changes to border radius
            node.bypass_change_detection().border_radius = style.border_radius.resolve(
                inverse_target_scale_factor.recip(),
                node.size,
                target_size,
            );

            if let Some(outline) = maybe_outline {
                // don't trigger change detection when only outlines are changed
                let node = node.bypass_change_detection();
                node.outline_width = if style.display != Display::None {
                    outline
                        .width
                        .resolve(
                            inverse_target_scale_factor.recip(),
                            node.size().x,
                            target_size,
                        )
                        .unwrap_or(0.)
                        .max(0.)
                } else {
                    0.
                };

                node.outline_offset = outline
                    .offset
                    .resolve(
                        inverse_target_scale_factor.recip(),
                        node.size().x,
                        target_size,
                    )
                    .unwrap_or(0.)
                    // Clamp outline offsets to at least the length of the node's shorter side
                    // Negative offset outlines can be useful to create thing like in-set focus indicators
                    .max(-0.5 * node.size.min_element());
            }

            node.bypass_change_detection().scrollbar_size =
                Vec2::new(layout.scrollbar_size.width, layout.scrollbar_size.height);

            let scroll_position: Vec2 = maybe_scroll_position
                .map(|scroll_pos| {
                    Vec2::new(
                        if style.overflow.x == OverflowAxis::Scroll {
                            scroll_pos.x * inverse_target_scale_factor.recip()
                        } else {
                            0.0
                        },
                        if style.overflow.y == OverflowAxis::Scroll {
                            scroll_pos.y * inverse_target_scale_factor.recip()
                        } else {
                            0.0
                        },
                    )
                })
                .unwrap_or_default();

            let max_possible_offset =
                (content_size - layout_size + node.scrollbar_size).max(Vec2::ZERO);
            let clamped_scroll_position = scroll_position.clamp(Vec2::ZERO, max_possible_offset);

            let physical_scroll_position = clamped_scroll_position.floor();

            node.bypass_change_detection().scroll_position = physical_scroll_position;

            for child_uinode in ui_children.iter_ui_children(entity) {
                // An independent child is its own layout root, placed after
                // this tree against this node's final geometry.
                if roots.independent.contains(child_uinode) {
                    independent_children.push(child_uinode);
                    continue;
                }
                update_uinode_geometry_recursive(
                    child_uinode,
                    ui_surface,
                    use_rounding,
                    target_size,
                    inherited_transform,
                    node_update_query,
                    ui_children,
                    inverse_target_scale_factor,
                    layout_size,
                    physical_scroll_position,
                    roots,
                    independent_children,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        layout::ui_surface::UiSurface, prelude::*, ui_layout_system,
        update::propagate_ui_target_cameras, ContentSize, LayoutContext,
    };
    use bevy_app::{App, HierarchyPropagatePlugin, PostUpdate, PropagateSet, TaskPoolPlugin};
    use bevy_camera::{Camera, Camera2d, ComputedCameraValues, RenderTargetInfo, Viewport};
    use bevy_ecs::{prelude::*, system::RunSystemOnce};
    use bevy_math::{Rect, UVec2, Vec2};
    use bevy_platform::collections::HashMap;
    use bevy_transform::systems::mark_dirty_trees;
    use bevy_transform::systems::{propagate_parent_transforms, sync_simple_transforms};
    use bevy_utils::prelude::default;

    use taffy::TraversePartialTree;

    // these window dimensions are easy to convert to and from percentage values
    const TARGET_WIDTH: u32 = 1000;
    const TARGET_HEIGHT: u32 = 100;

    fn setup_ui_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(TaskPoolPlugin::default());

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ));
        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiRenderTargetInfo>::new(
            PostUpdate,
        ));
        app.init_resource::<UiScale>();
        app.init_resource::<UiSurface>();
        app.init_resource::<bevy_text::TextPipeline>();
        app.init_resource::<bevy_text::FontCx>();
        app.init_resource::<bevy_text::ScaleCx>();
        app.init_resource::<bevy_transform::StaticTransformOptimizations>();

        app.add_systems(
            PostUpdate,
            (
                ApplyDeferred,
                propagate_ui_target_cameras,
                ui_layout_system,
                mark_dirty_trees,
                sync_simple_transforms,
                propagate_parent_transforms,
            )
                .chain(),
        );

        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiTargetCamera>::default()
                .after(propagate_ui_target_cameras)
                .before(ui_layout_system),
        );

        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiRenderTargetInfo>::default()
                .after(propagate_ui_target_cameras)
                .before(ui_layout_system),
        );

        let world = app.world_mut();
        // spawn a camera with a dummy render target
        world.spawn((
            Camera2d,
            Camera {
                computed: ComputedCameraValues {
                    target_info: Some(RenderTargetInfo {
                        physical_size: UVec2::new(TARGET_WIDTH, TARGET_HEIGHT),
                        scale_factor: 1.,
                    }),
                    ..Default::default()
                },
                viewport: Some(Viewport {
                    physical_size: UVec2::new(TARGET_WIDTH, TARGET_HEIGHT),
                    ..default()
                }),
                ..Default::default()
            },
        ));

        app
    }

    #[test]
    fn ui_nodes_with_percent_100_dimensions_should_fill_their_parent() {
        let mut app = setup_ui_test_app();

        let world = app.world_mut();

        // spawn a root entity with width and height set to fill 100% of its parent
        let ui_root = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();

        let ui_child = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();

        world.entity_mut(ui_root).add_child(ui_child);

        app.update();

        let mut ui_surface = app.world_mut().resource_mut::<UiSurface>();

        for ui_entity in [ui_root, ui_child] {
            let layout = ui_surface.get_layout(ui_entity, true).unwrap().0;
            assert_eq!(layout.size.width, TARGET_WIDTH as f32);
            assert_eq!(layout.size.height, TARGET_HEIGHT as f32);
        }
    }

    #[test]
    fn ui_surface_tracks_ui_entities() {
        let mut app = setup_ui_test_app();

        let world = app.world_mut();
        // no UI entities in world, none in UiSurface
        let ui_surface = world.resource::<UiSurface>();
        assert!(ui_surface.entity_to_taffy.is_empty());

        let ui_entity = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();
        assert!(ui_surface.entity_to_taffy.contains_key(&ui_entity));
        assert_eq!(ui_surface.entity_to_taffy.len(), 1);

        world.despawn(ui_entity);

        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();
        assert!(!ui_surface.entity_to_taffy.contains_key(&ui_entity));
        assert!(ui_surface.entity_to_taffy.is_empty());
    }

    #[test]
    #[should_panic]
    fn despawning_a_ui_entity_should_remove_its_corresponding_ui_node() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let ui_entity = world.spawn(Node::default()).id();

        // `ui_layout_system` will insert a ui node into the internal layout tree corresponding to `ui_entity`
        app.update();
        let world = app.world_mut();

        // retrieve the ui node corresponding to `ui_entity` from ui surface
        let ui_surface = world.resource::<UiSurface>();
        let ui_node = ui_surface.entity_to_taffy[&ui_entity];

        world.despawn(ui_entity);

        // `ui_layout_system` will receive a `RemovedComponents<Node>` event for `ui_entity`
        // and remove `ui_entity` from `ui_node` from the internal layout tree
        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();

        // `ui_node` is removed, attempting to retrieve a style for `ui_node` panics
        let _ = ui_surface.taffy.style(ui_node.id);
    }

    #[test]
    fn changes_to_children_of_a_ui_entity_change_its_corresponding_ui_nodes_children() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let ui_parent_entity = world.spawn(Node::default()).id();

        // `ui_layout_system` will insert a ui node into the internal layout tree corresponding to `ui_entity`
        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();
        let ui_parent_node = ui_surface.entity_to_taffy[&ui_parent_entity];

        // `ui_parent_node` shouldn't have any children yet
        assert_eq!(ui_surface.taffy.child_count(ui_parent_node.id), 0);

        let mut ui_child_entities = (0..10)
            .map(|_| {
                let child = world.spawn(Node::default()).id();
                world.entity_mut(ui_parent_entity).add_child(child);
                child
            })
            .collect::<Vec<_>>();

        app.update();
        let world = app.world_mut();

        // `ui_parent_node` should have children now
        let ui_surface = world.resource::<UiSurface>();
        assert_eq!(
            ui_surface.entity_to_taffy.len(),
            1 + ui_child_entities.len()
        );
        assert_eq!(
            ui_surface.taffy.child_count(ui_parent_node.id),
            ui_child_entities.len()
        );

        let child_node_map = <HashMap<_, _>>::from_iter(
            ui_child_entities
                .iter()
                .map(|child_entity| (*child_entity, ui_surface.entity_to_taffy[child_entity])),
        );

        // the children should have a corresponding ui node and that ui node's parent should be `ui_parent_node`
        for node in child_node_map.values() {
            assert_eq!(ui_surface.taffy.parent(node.id), Some(ui_parent_node.id));
        }

        // delete every second child
        let mut deleted_children = vec![];
        for i in (0..ui_child_entities.len()).rev().step_by(2) {
            let child = ui_child_entities.remove(i);
            world.despawn(child);
            deleted_children.push(child);
        }

        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();
        assert_eq!(
            ui_surface.entity_to_taffy.len(),
            1 + ui_child_entities.len()
        );
        assert_eq!(
            ui_surface.taffy.child_count(ui_parent_node.id),
            ui_child_entities.len()
        );

        // the remaining children should still have nodes in the layout tree
        for child_entity in &ui_child_entities {
            let child_node = child_node_map[child_entity];
            assert_eq!(ui_surface.entity_to_taffy[child_entity], child_node);
            assert_eq!(
                ui_surface.taffy.parent(child_node.id),
                Some(ui_parent_node.id)
            );
            assert!(ui_surface
                .taffy
                .children(ui_parent_node.id)
                .unwrap()
                .contains(&child_node.id));
        }

        // the nodes of the deleted children should have been removed from the layout tree
        for deleted_child_entity in &deleted_children {
            assert!(!ui_surface
                .entity_to_taffy
                .contains_key(deleted_child_entity));
            let deleted_child_node = child_node_map[deleted_child_entity];
            assert!(!ui_surface
                .taffy
                .children(ui_parent_node.id)
                .unwrap()
                .contains(&deleted_child_node.id));
        }

        // despawn the parent entity and its descendants
        world.entity_mut(ui_parent_entity).despawn();

        app.update();
        let world = app.world_mut();

        // all nodes should have been deleted
        let ui_surface = world.resource::<UiSurface>();
        assert!(ui_surface.entity_to_taffy.is_empty());
    }

    /// bugfix test, see [#16288](https://github.com/bevyengine/bevy/pull/16288)
    #[test]
    fn node_removal_and_reinsert_should_work() {
        let mut app = setup_ui_test_app();

        app.update();
        let world = app.world_mut();

        // no UI entities in world, none in UiSurface
        let ui_surface = world.resource::<UiSurface>();
        assert!(ui_surface.entity_to_taffy.is_empty());

        let ui_entity = world.spawn(Node::default()).id();

        // `ui_layout_system` should map `ui_entity` to a ui node in `UiSurface::entity_to_taffy`
        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();
        assert!(ui_surface.entity_to_taffy.contains_key(&ui_entity));
        assert_eq!(ui_surface.entity_to_taffy.len(), 1);

        // remove and re-insert Node to trigger removal code in `ui_layout_system`
        world.entity_mut(ui_entity).remove::<Node>();
        world.entity_mut(ui_entity).insert(Node::default());

        // `ui_layout_system` should still have `ui_entity`
        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();
        assert!(ui_surface.entity_to_taffy.contains_key(&ui_entity));
        assert_eq!(ui_surface.entity_to_taffy.len(), 1);
    }

    #[test]
    fn node_addition_should_sync_children() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        // spawn an invalid UI root node
        let root_node = world.spawn(()).with_child(Node::default()).id();

        app.update();
        let world = app.world_mut();

        // fix the invalid root node by inserting a Node
        world.entity_mut(root_node).insert(Node::default());

        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource_mut::<UiSurface>();
        let taffy_root = ui_surface.entity_to_taffy[&root_node];

        // There should be one child of the root node after fixing it
        assert_eq!(ui_surface.taffy.child_count(taffy_root.id), 1);
    }

    #[test]
    fn node_addition_should_sync_parent_and_children() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let d = world.spawn(Node::default()).id();
        let c = world.spawn(()).add_child(d).id();
        let b = world.spawn(Node::default()).id();
        let a = world.spawn(Node::default()).add_children(&[b, c]).id();

        app.update();
        let world = app.world_mut();

        // fix the invalid middle node by inserting a Node
        world.entity_mut(c).insert(Node::default());

        app.update();
        let world = app.world_mut();

        let ui_surface = world.resource::<UiSurface>();
        for (entity, n) in [(a, 2), (b, 0), (c, 1), (d, 0)] {
            let taffy_id = ui_surface.entity_to_taffy[&entity].id;
            assert_eq!(ui_surface.taffy.child_count(taffy_id), n);
        }
    }

    /// regression test for >=0.13.1 root node layouts
    /// ensure root nodes act like they are absolutely positioned
    /// without explicitly declaring it.
    #[test]
    fn ui_root_node_should_act_like_position_absolute() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let mut size = 150.;

        world.spawn(Node {
            // test should pass without explicitly requiring position_type to be set to Absolute
            // position_type: PositionType::Absolute,
            width: Val::Px(size),
            height: Val::Px(size),
            ..default()
        });

        size -= 50.;

        world.spawn(Node {
            // position_type: PositionType::Absolute,
            width: Val::Px(size),
            height: Val::Px(size),
            ..default()
        });

        size -= 50.;

        world.spawn(Node {
            // position_type: PositionType::Absolute,
            width: Val::Px(size),
            height: Val::Px(size),
            ..default()
        });

        app.update();
        let world = app.world_mut();

        let overlap_check = world
            .query_filtered::<(Entity, &ComputedNode, &UiGlobalTransform), Without<ChildOf>>()
            .iter(world)
            .fold(
                Option::<(Rect, bool)>::None,
                |option_rect, (entity, node, transform)| {
                    let current_rect = Rect::from_center_size(transform.translation, node.size());
                    assert!(
                        current_rect.height().abs() + current_rect.width().abs() > 0.,
                        "root ui node {entity} doesn't have a logical size"
                    );
                    assert_ne!(
                        *transform,
                        UiGlobalTransform::default(),
                        "root ui node {entity} transform is not populated"
                    );
                    let Some((rect, is_overlapping)) = option_rect else {
                        return Some((current_rect, false));
                    };
                    if rect.contains(current_rect.center()) {
                        Some((current_rect, true))
                    } else {
                        Some((current_rect, is_overlapping))
                    }
                },
            );

        let Some((_rect, is_overlapping)) = overlap_check else {
            unreachable!("test not setup properly");
        };
        assert!(is_overlapping, "root ui nodes are expected to behave like they have absolute position and be independent from each other");
    }

    #[test]
    fn ui_node_should_properly_update_when_changing_target_camera() {
        #[derive(Component)]
        struct MovingUiNode;

        fn update_camera_viewports(mut cameras: Query<&mut Camera>) {
            let camera_count = cameras.iter().len();
            for (camera_index, mut camera) in cameras.iter_mut().enumerate() {
                let target_size = camera.physical_target_size().unwrap();
                let viewport_width = target_size.x / camera_count as u32;
                let physical_position = UVec2::new(viewport_width * camera_index as u32, 0);
                let physical_size = UVec2::new(target_size.x / camera_count as u32, target_size.y);
                camera.viewport = Some(Viewport {
                    physical_position,
                    physical_size,
                    ..default()
                });
            }
        }

        fn move_ui_node(
            In(pos): In<Vec2>,
            mut commands: Commands,
            cameras: Query<(Entity, &Camera)>,
            moving_ui_query: Query<Entity, With<MovingUiNode>>,
        ) {
            let (target_camera_entity, _) = cameras
                .iter()
                .find(|(_, camera)| {
                    let Some(logical_viewport_rect) = camera.logical_viewport_rect() else {
                        panic!("missing logical viewport")
                    };
                    // make sure cursor is in viewport and that viewport has at least 1px of size
                    logical_viewport_rect.contains(pos)
                        && logical_viewport_rect.max.cmpge(Vec2::splat(0.)).any()
                })
                .expect("cursor position outside of camera viewport");
            for moving_ui_entity in moving_ui_query.iter() {
                commands
                    .entity(moving_ui_entity)
                    .insert(UiTargetCamera(target_camera_entity))
                    .insert(Node {
                        position_type: PositionType::Absolute,
                        top: Val::Px(pos.y),
                        left: Val::Px(pos.x),
                        ..default()
                    });
            }
        }

        fn do_move_and_test(app: &mut App, new_pos: Vec2, expected_camera_entity: &Entity) {
            let world = app.world_mut();
            world.run_system_once_with(move_ui_node, new_pos).unwrap();
            app.update();
            let world = app.world_mut();
            let (ui_node_entity, UiTargetCamera(target_camera_entity)) = world
                .query_filtered::<(Entity, &UiTargetCamera), With<MovingUiNode>>()
                .single(world)
                .expect("missing MovingUiNode");
            assert_eq!(expected_camera_entity, target_camera_entity);
            let mut ui_surface = world.resource_mut::<UiSurface>();

            let layout = ui_surface
                .get_layout(ui_node_entity, true)
                .expect("failed to get layout")
                .0;

            // negative test for #12255
            assert_eq!(Vec2::new(layout.location.x, layout.location.y), new_pos);
        }

        fn get_taffy_node_count(world: &World) -> usize {
            world.resource::<UiSurface>().taffy.total_node_count()
        }

        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        world.spawn((
            Camera2d,
            Camera {
                order: 1,
                computed: ComputedCameraValues {
                    target_info: Some(RenderTargetInfo {
                        physical_size: UVec2::new(TARGET_WIDTH, TARGET_HEIGHT),
                        scale_factor: 1.,
                    }),
                    ..default()
                },
                viewport: Some(Viewport {
                    physical_size: UVec2::new(TARGET_WIDTH, TARGET_HEIGHT),
                    ..default()
                }),
                ..default()
            },
        ));

        world.spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.),
                left: Val::Px(0.),
                ..default()
            },
            MovingUiNode,
        ));

        app.update();
        let world = app.world_mut();

        let pos_inc = Vec2::splat(1.);
        let total_cameras = world.query::<&Camera>().iter(world).len();
        // add total cameras - 1 (the assumed default) to get an idea for how many nodes we should expect
        let expected_max_taffy_node_count = get_taffy_node_count(world) + total_cameras - 1;

        world.run_system_once(update_camera_viewports).unwrap();

        app.update();
        let world = app.world_mut();

        let viewport_rects = world
            .query::<(Entity, &Camera)>()
            .iter(world)
            .map(|(e, c)| (e, c.logical_viewport_rect().expect("missing viewport")))
            .collect::<Vec<_>>();

        for (camera_entity, viewport) in viewport_rects.iter() {
            let target_pos = viewport.min + pos_inc;
            do_move_and_test(&mut app, target_pos, camera_entity);
        }

        // reverse direction
        let mut viewport_rects = viewport_rects.clone();
        viewport_rects.reverse();
        for (camera_entity, viewport) in viewport_rects.iter() {
            let target_pos = viewport.max - pos_inc;
            do_move_and_test(&mut app, target_pos, camera_entity);
        }

        let world = app.world();
        let current_taffy_node_count = get_taffy_node_count(world);
        if current_taffy_node_count > expected_max_taffy_node_count {
            panic!("extra taffy nodes detected: current: {current_taffy_node_count} max expected: {expected_max_taffy_node_count}");
        }
    }

    #[test]
    fn ui_node_should_be_set_to_its_content_size() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let content_size = Vec2::new(50., 25.);

        let ui_entity = world
            .spawn((
                Node {
                    align_self: AlignSelf::Start,
                    ..default()
                },
                ContentSize::fixed_size(content_size),
            ))
            .id();

        app.update();
        let world = app.world_mut();

        let mut ui_surface = world.resource_mut::<UiSurface>();
        let layout = ui_surface.get_layout(ui_entity, true).unwrap().0;

        // the node should takes its size from the fixed size measure func
        assert_eq!(layout.size.width, content_size.x);
        assert_eq!(layout.size.height, content_size.y);
    }

    #[test]
    fn measured_node_includes_border_and_padding() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let ui_node = world
            .spawn((
                Node {
                    align_self: AlignSelf::Start,
                    border: UiRect {
                        left: px(2.0),
                        right: px(6.0),
                        top: px(4.0),
                        bottom: px(8.0),
                    },
                    padding: UiRect {
                        left: px(3.0),
                        right: px(5.0),
                        top: px(7.0),
                        bottom: px(11.0),
                    },
                    ..default()
                },
                ContentSize::fixed_size(Vec2::new(50.0, 25.0)),
            ))
            .id();

        app.update();
        let world = app.world_mut();
        let mut ui_surface = world.resource_mut::<UiSurface>();
        let layout = ui_surface.get_layout(ui_node, true).unwrap().0;

        assert_eq!(layout.border.left, 2.0);
        assert_eq!(layout.border.right, 6.0);
        assert_eq!(layout.border.top, 4.0);
        assert_eq!(layout.border.bottom, 8.0);
        assert_eq!(layout.padding.left, 3.0);
        assert_eq!(layout.padding.right, 5.0);
        assert_eq!(layout.padding.top, 7.0);
        assert_eq!(layout.padding.bottom, 11.0);
        assert_eq!(layout.size.width, 66.0);
        assert_eq!(layout.size.height, 55.0);
        assert_eq!(layout.content_size.width, 58.0);
        assert_eq!(layout.content_size.height, 43.0);
        assert_eq!(layout.content_box_width(), 50.0);
        assert_eq!(layout.content_box_height(), 25.0);
    }

    #[test]
    fn measure_funcs_should_be_removed_on_content_size_clear() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let content_size = Vec2::new(50., 25.);
        let ui_entity = world
            .spawn((
                Node {
                    align_self: AlignSelf::Start,
                    ..Default::default()
                },
                ContentSize::fixed_size(content_size),
            ))
            .id();

        app.update();
        let world = app.world_mut();

        let mut ui_surface = world.resource_mut::<UiSurface>();
        let ui_node = ui_surface.entity_to_taffy[&ui_entity];

        // a node with a content size should have taffy context
        assert!(ui_surface.taffy.get_node_context(ui_node.id).is_some());
        let layout = ui_surface.get_layout(ui_entity, true).unwrap().0;
        assert_eq!(layout.size.width, content_size.x);
        assert_eq!(layout.size.height, content_size.y);

        world
            .entity_mut(ui_entity)
            .get_mut::<ContentSize>()
            .unwrap()
            .clear();

        app.update();
        let world = app.world_mut();

        let mut ui_surface = world.resource_mut::<UiSurface>();
        // a node with a cleared content size should not have taffy context
        assert!(ui_surface.taffy.get_node_context(ui_node.id).is_none());

        // Without a content size, the node has no width or height constraints so the length of both dimensions is 0.
        let layout = ui_surface.get_layout(ui_entity, true).unwrap().0;
        assert_eq!(layout.size.width, 0.);
        assert_eq!(layout.size.height, 0.);
    }

    #[test]
    fn measure_funcs_should_persist_until_cleared() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let content_size = Vec2::new(50., 25.);
        let ui_entity = world
            .spawn((Node::default(), ContentSize::fixed_size(content_size)))
            .id();

        app.update();
        let world = app.world_mut();
        let mut ui_surface = world.resource_mut::<UiSurface>();
        let ui_node = ui_surface.entity_to_taffy[&ui_entity];
        assert!(ui_surface.taffy.get_node_context(ui_node.id).is_some());
        let layout = ui_surface.get_layout(ui_entity, true).unwrap().0;
        assert_eq!(layout.size.width, content_size.x);
        assert_eq!(layout.size.height, content_size.y);

        world.entity_mut(ui_entity).insert(Node::default());

        app.update();
        let world = app.world_mut();
        let mut ui_surface = world.resource_mut::<UiSurface>();
        assert!(ui_surface.taffy.get_node_context(ui_node.id).is_some());
        let layout = ui_surface.get_layout(ui_entity, true).unwrap().0;
        assert_eq!(layout.size.width, content_size.x);
        assert_eq!(layout.size.height, content_size.y);

        world
            .entity_mut(ui_entity)
            .get_mut::<ContentSize>()
            .unwrap()
            .clear();

        app.update();
        let world = app.world_mut();
        let mut ui_surface = world.resource_mut::<UiSurface>();
        assert!(ui_surface.taffy.get_node_context(ui_node.id).is_none());
        let layout = ui_surface.get_layout(ui_entity, true).unwrap().0;
        assert_eq!(layout.size.width, 0.);
        assert_eq!(layout.size.height, 0.);
    }

    #[test]
    fn ui_rounding_test() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let parent = world
            .spawn(Node {
                display: Display::Grid,
                grid_template_columns: RepeatedGridTrack::min_content(2),
                margin: UiRect::all(Val::Px(4.0)),
                ..default()
            })
            .with_children(|commands| {
                for _ in 0..2 {
                    commands.spawn(Node {
                        display: Display::Grid,
                        width: Val::Px(160.),
                        height: Val::Px(160.),
                        ..default()
                    });
                }
            })
            .id();

        let children = world
            .entity(parent)
            .get::<Children>()
            .unwrap()
            .iter()
            .collect::<Vec<Entity>>();

        for r in [2, 3, 5, 7, 11, 13, 17, 19, 21, 23, 29, 31].map(|n| (n as f32).recip()) {
            // This fails with very small / unrealistic scale values
            let mut s = 1. - r;
            while s <= 5. {
                app.world_mut().resource_mut::<UiScale>().0 = s;
                app.update();
                let world = app.world_mut();
                let width_sum: f32 = children
                    .iter()
                    .map(|child| world.get::<ComputedNode>(*child).unwrap().size.x)
                    .sum();
                let parent_width = world.get::<ComputedNode>(parent).unwrap().size.x;
                assert!((width_sum - parent_width).abs() < 0.001);
                assert!((width_sum - 320. * s).abs() <= 1.);
                s += r;
            }
        }
    }

    #[test]
    fn no_camera_ui() {
        let mut app = App::new();

        app.add_systems(
            PostUpdate,
            (propagate_ui_target_cameras, ApplyDeferred, ui_layout_system).chain(),
        );

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ));

        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiTargetCamera>::default()
                .after(propagate_ui_target_cameras)
                .before(ui_layout_system),
        );

        let world = app.world_mut();
        world.init_resource::<UiScale>();
        world.init_resource::<UiSurface>();

        world.init_resource::<bevy_text::TextPipeline>();

        world.init_resource::<bevy_text::FontCx>();

        world.init_resource::<bevy_text::ScaleCx>();

        let ui_root = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();

        let ui_child = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();

        world.entity_mut(ui_root).add_child(ui_child);

        app.update();
    }

    #[test]
    fn test_ui_surface_compute_camera_layout() {
        use bevy_ecs::prelude::ResMut;

        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let root_node_entity = Entity::from_raw_u32(1).unwrap();

        struct TestSystemParam {
            root_node_entity: Entity,
        }

        fn test_system(
            params: In<TestSystemParam>,
            mut ui_surface: ResMut<UiSurface>,
            mut computed_text_block_query: Query<&mut bevy_text::ComputedTextBlock>,
            mut font_system: ResMut<bevy_text::FontCx>,
        ) {
            ui_surface.upsert_node(
                &LayoutContext::TEST_CONTEXT,
                params.root_node_entity,
                &Node::default(),
                None,
            );

            ui_surface.compute_layout(
                params.root_node_entity,
                UVec2::new(800, 600),
                &mut computed_text_block_query,
                &mut font_system,
            );
        }

        let _ = world.run_system_once_with(test_system, TestSystemParam { root_node_entity });

        let ui_surface = world.resource::<UiSurface>();

        let taffy_node = ui_surface.entity_to_taffy.get(&root_node_entity).unwrap();
        assert!(ui_surface.taffy.layout(taffy_node.id).is_ok());
    }

    #[test]
    fn no_viewport_node_leak_on_root_despawned() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let ui_root_entity = world.spawn(Node::default()).id();

        // The UI schedule synchronizes Bevy UI's internal `TaffyTree` with the
        // main world's tree of `Node` entities.
        app.update();
        let world = app.world_mut();

        // Two taffy nodes are added to the internal `TaffyTree` for each root UI entity.
        // An implicit taffy node representing the viewport and a taffy node corresponding to the
        // root UI entity which is parented to the viewport taffy node.
        assert_eq!(
            world.resource_mut::<UiSurface>().taffy.total_node_count(),
            2
        );

        world.despawn(ui_root_entity);

        // The UI schedule removes both the taffy node corresponding to `ui_root_entity` and its
        // parent viewport node.
        app.update();
        let world = app.world_mut();

        // Both taffy nodes should now be removed from the internal `TaffyTree`
        assert_eq!(
            world.resource_mut::<UiSurface>().taffy.total_node_count(),
            0
        );
    }

    #[test]
    fn no_viewport_node_leak_on_parented_root() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();

        let ui_root_entity_1 = world.spawn(Node::default()).id();
        let ui_root_entity_2 = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        // There are two UI root entities. Each root taffy node is given it's own viewport node parent,
        // so a total of four taffy nodes are added to the `TaffyTree` by the UI schedule.
        assert_eq!(
            world.resource_mut::<UiSurface>().taffy.total_node_count(),
            4
        );

        // Should be two viewport nodes tracked in the root to viewport node map.
        assert_eq!(
            world
                .resource_mut::<UiSurface>()
                .root_entity_to_viewport_node
                .len(),
            2
        );

        // Parent `ui_root_entity_2` onto `ui_root_entity_1` so now only `ui_root_entity_1` is a
        // UI root entity.
        world
            .entity_mut(ui_root_entity_1)
            .add_child(ui_root_entity_2);

        // Now there is only one root node so the second viewport node is removed by
        // the UI schedule.
        app.update();
        let world = app.world_mut();

        // There is only one viewport node now, so the `TaffyTree` contains 3 nodes in total.
        assert_eq!(
            world.resource_mut::<UiSurface>().taffy.total_node_count(),
            3
        );

        // The entry for `ui_root_entity_2` should have been removed from `root_entity_to_viewport_node`
        assert_eq!(
            world
                .resource_mut::<UiSurface>()
                .root_entity_to_viewport_node
                .len(),
            1
        );
        assert!(world
            .resource_mut::<UiSurface>()
            .root_entity_to_viewport_node
            .contains_key(&ui_root_entity_1));
    }

    /// The laid-out size and the top-left corner (physical px) of a UI node.
    fn size_and_corner(world: &mut World, entity: Entity) -> (Vec2, Vec2) {
        let node = world.get::<ComputedNode>(entity).unwrap();
        let transform = world.get::<UiGlobalTransform>(entity).unwrap();
        (node.size, transform.translation - 0.5 * node.size)
    }

    /// A full-viewport root holding, in a row, an independent node absolutely
    /// placed at (30, 20) and a 50 px wide sibling. Returns (root, independent,
    /// sibling).
    fn spawn_independent_fixture(world: &mut World) -> (Entity, Entity, Entity) {
        let root = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();
        let independent = world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(30.),
                    top: Val::Px(20.),
                    width: Val::Px(100.),
                    height: Val::Px(40.),
                    ..default()
                },
                IndependentLayout,
                ChildOf(root),
            ))
            .id();
        let sibling = world
            .spawn((
                Node {
                    width: Val::Px(50.),
                    height: Val::Px(10.),
                    ..default()
                },
                ChildOf(root),
            ))
            .id();
        (root, independent, sibling)
    }

    #[test]
    fn an_independent_node_is_placed_relative_to_its_parent_and_takes_no_space() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();
        let (_root, independent, sibling) = spawn_independent_fixture(world);
        // A non-absolute independent node would have taken the row's first
        // 200 px; it must not.
        let flowing = world
            .spawn((
                Node {
                    width: Val::Px(200.),
                    height: Val::Px(10.),
                    ..default()
                },
                IndependentLayout,
                ChildOf(_root),
            ))
            .id();
        app.update();
        let world = app.world_mut();

        let (size, corner) = size_and_corner(world, independent);
        assert_eq!(size, Vec2::new(100., 40.));
        assert_eq!(corner, Vec2::new(30., 20.));
        let (_, sibling_corner) = size_and_corner(world, sibling);
        assert_eq!(
            sibling_corner,
            Vec2::ZERO,
            "the sibling is first in the row"
        );
        let (flowing_size, flowing_corner) = size_and_corner(world, flowing);
        assert_eq!(flowing_size, Vec2::new(200., 10.));
        assert_eq!(flowing_corner, Vec2::ZERO);
    }

    #[test]
    fn a_change_inside_an_independent_node_leaves_its_parents_tree_alone() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();
        let (_root, independent, sibling) = spawn_independent_fixture(world);
        let inner = world
            .spawn((
                Node {
                    width: Val::Px(10.),
                    height: Val::Px(10.),
                    ..default()
                },
                ChildOf(independent),
            ))
            .id();
        app.update();
        let (_, before) = size_and_corner(app.world_mut(), sibling);

        // An offset on the sibling that change detection is not told about: a
        // pass over the root's tree would apply it, so the sibling staying put
        // is the root's tree not having been walked.
        let world = app.world_mut();
        world
            .get_mut::<UiTransform>(sibling)
            .unwrap()
            .bypass_change_detection()
            .translation = Val2::px(7., 0.);
        world.get_mut::<Node>(inner).unwrap().width = Val::Px(20.);
        app.update();
        let world = app.world_mut();

        let (inner_size, _) = size_and_corner(world, inner);
        assert_eq!(inner_size.x, 20., "the independent node was laid out");
        let (_, after) = size_and_corner(world, sibling);
        assert_eq!(
            after, before,
            "a change inside an independent node must not walk its parent's tree"
        );

        // With teeth: a change in the root's own tree does walk it, and the
        // planted offset then shows.
        app.world_mut().get_mut::<Node>(sibling).unwrap().height = Val::Px(11.);
        app.update();
        let (_, walked) = size_and_corner(app.world_mut(), sibling);
        assert_eq!(walked, before + Vec2::new(7., 0.));
    }

    #[test]
    fn an_independent_node_under_a_hidden_ancestor_is_collapsed_and_restored() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();
        let root = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();
        let container = world
            .spawn((
                Node {
                    width: Val::Px(300.),
                    height: Val::Px(60.),
                    ..default()
                },
                ChildOf(root),
            ))
            .id();
        let independent = world
            .spawn((
                Node {
                    width: Val::Px(100.),
                    height: Val::Px(40.),
                    ..default()
                },
                IndependentLayout,
                ChildOf(container),
            ))
            .id();
        app.update();
        assert_eq!(
            size_and_corner(app.world_mut(), independent).0,
            Vec2::new(100., 40.)
        );

        app.world_mut().get_mut::<Node>(container).unwrap().display = Display::None;
        app.update();
        assert_eq!(size_and_corner(app.world_mut(), independent).0, Vec2::ZERO);

        app.world_mut().get_mut::<Node>(container).unwrap().display = Display::Flex;
        app.update();
        assert_eq!(
            size_and_corner(app.world_mut(), independent).0,
            Vec2::new(100., 40.)
        );
    }

    #[test]
    fn removing_independence_rejoins_the_parents_layout() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();
        let root = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();
        let first = world
            .spawn((
                Node {
                    width: Val::Px(200.),
                    height: Val::Px(10.),
                    ..default()
                },
                IndependentLayout,
                ChildOf(root),
            ))
            .id();
        let second = world
            .spawn((
                Node {
                    width: Val::Px(50.),
                    height: Val::Px(10.),
                    ..default()
                },
                ChildOf(root),
            ))
            .id();
        app.update();
        assert_eq!(size_and_corner(app.world_mut(), second).1, Vec2::ZERO);

        app.world_mut()
            .entity_mut(first)
            .remove::<IndependentLayout>();
        app.update();
        assert_eq!(
            size_and_corner(app.world_mut(), second).1,
            Vec2::new(200., 0.),
            "the formerly independent node takes its place in the row again"
        );

        app.world_mut().entity_mut(first).insert(IndependentLayout);
        app.update();
        assert_eq!(size_and_corner(app.world_mut(), second).1, Vec2::ZERO);
    }

    #[test]
    fn an_independent_node_follows_its_parent() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();
        let root = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();
        let parent = world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(100.),
                    top: Val::Px(10.),
                    width: Val::Px(300.),
                    height: Val::Px(60.),
                    ..default()
                },
                ChildOf(root),
            ))
            .id();
        let independent = world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(5.),
                    top: Val::Px(5.),
                    width: Val::Px(20.),
                    height: Val::Px(20.),
                    ..default()
                },
                IndependentLayout,
                ChildOf(parent),
            ))
            .id();
        app.update();
        assert_eq!(
            size_and_corner(app.world_mut(), independent).1,
            Vec2::new(105., 15.)
        );

        app.world_mut().get_mut::<Node>(parent).unwrap().left = Val::Px(200.);
        app.update();
        assert_eq!(
            size_and_corner(app.world_mut(), independent).1,
            Vec2::new(205., 15.),
            "an unchanged independent node is placed again when its parent moves"
        );
    }

    #[test]
    fn an_unparented_node_is_laid_out_as_a_root() {
        let mut app = setup_ui_test_app();
        let world = app.world_mut();
        let root = world
            .spawn(Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                ..default()
            })
            .id();
        let child = world
            .spawn((
                Node {
                    width: Val::Percent(50.),
                    height: Val::Px(10.),
                    ..default()
                },
                ChildOf(root),
            ))
            .id();
        app.update();

        // Nothing about the child itself changes: only its parent goes.
        app.world_mut().entity_mut(child).remove::<ChildOf>();
        app.update();
        let world = app.world_mut();
        assert!(
            world
                .resource::<UiSurface>()
                .root_entity_to_viewport_node
                .contains_key(&child),
            "an un-parented node becomes a layout root of its own"
        );
        assert_eq!(
            size_and_corner(world, child).0.x,
            TARGET_WIDTH as f32 / 2.,
            "and is laid out against the viewport"
        );
    }

    #[cfg(feature = "ghost_nodes")]
    mod ghost_node_tests {
        use super::*;
        use crate::experimental::GhostNode;

        fn compare_taffy_children(
            ui_surface: &UiSurface,
            parent: Entity,
            children: &[Entity],
        ) -> bool {
            let parent_to_taffy_children = ui_surface
                .taffy
                .children(ui_surface.entity_to_taffy[&parent].id)
                .unwrap();
            let children_to_taffy_children = children
                .iter()
                .map(|entity| ui_surface.entity_to_taffy[entity].id)
                .collect::<Vec<_>>();

            parent_to_taffy_children == children_to_taffy_children
        }

        fn compare_taffy_parent(
            ui_surface: &UiSurface,
            child: Entity,
            parent: Option<Entity>,
        ) -> bool {
            let child_to_taffy_parent = ui_surface
                .taffy
                .parent(ui_surface.entity_to_taffy[&child].id);
            let parent_to_taffy_parent =
                parent.map(|entity| ui_surface.entity_to_taffy[&entity].id);

            child_to_taffy_parent == parent_to_taffy_parent
        }

        #[test]
        fn unparenting_ghost_child_should_unparent_taffy_child() {
            let mut app = setup_ui_test_app();
            let world = app.world_mut();

            let child = world.spawn(Node::default()).id();
            let ghost = world.spawn(GhostNode).add_child(child).id();
            let root = world.spawn(Node::default()).add_child(ghost).id();

            app.update();
            let world = app.world_mut();

            let ui_surface = world.resource::<UiSurface>();
            assert!(compare_taffy_children(ui_surface, root, &[child]));
            assert!(compare_taffy_parent(ui_surface, child, Some(root)));
            assert!(!ui_surface.root_entity_to_viewport_node.contains_key(&child));

            world.entity_mut(ghost).detach_all_children();

            app.update();
            let world = app.world_mut();

            let ui_surface = world.resource::<UiSurface>();

            // Unparenting child from ghost should unparent the corresponding child taffy node from the
            // root taffy node.
            assert!(compare_taffy_children(ui_surface, root, &[]));

            let viewport_node = ui_surface
                .root_entity_to_viewport_node
                .get(&child)
                .copied()
                .expect(
                    "detached child should become a UI root and have an associated viewport node",
                );
            let taffy_child = ui_surface.entity_to_taffy[&child].id;
            assert_eq!(ui_surface.taffy.parent(taffy_child), Some(viewport_node));
        }

        #[test]
        fn adding_intermediate_ghost_node_attaches_taffy_nodes() {
            let mut app = setup_ui_test_app();
            let world = app.world_mut();

            let child = world.spawn(Node::default()).id();
            let mid = world.spawn_empty().add_child(child).id();
            let root = world.spawn(Node::default()).add_child(mid).id();

            app.update();
            let world = app.world_mut();

            let ui_surface = world.resource::<UiSurface>();
            assert!(compare_taffy_children(ui_surface, root, &[]));
            assert!(compare_taffy_parent(ui_surface, child, None));

            world.entity_mut(mid).insert(GhostNode);

            app.update();
            let world = app.world_mut();

            let ui_surface = world.resource::<UiSurface>();
            assert!(compare_taffy_children(ui_surface, root, &[child]));
            assert!(compare_taffy_parent(ui_surface, child, Some(root)));
        }

        #[test]
        fn removing_intermeditate_ghost_node_detaches_taffy_nodes() {
            let mut app = setup_ui_test_app();
            let world = app.world_mut();

            let child = world.spawn(Node::default()).id();
            let mid = world.spawn(GhostNode).add_child(child).id();
            let root = world.spawn(Node::default()).add_child(mid).id();

            app.update();
            let world = app.world_mut();

            let ui_surface = world.resource::<UiSurface>();
            assert!(compare_taffy_children(ui_surface, root, &[child]));
            assert!(compare_taffy_parent(ui_surface, child, Some(root)));

            world.entity_mut(mid).remove::<GhostNode>();

            app.update();
            let world = app.world_mut();

            let ui_surface = world.resource::<UiSurface>();
            assert!(compare_taffy_children(ui_surface, root, &[]));
            assert!(compare_taffy_parent(ui_surface, child, None));
            assert!(!ui_surface.root_entity_to_viewport_node.contains_key(&child));
        }
    }
}
