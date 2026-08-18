//! Provides shadow cascade configuration and construction helpers.

use bevy_camera::{Camera, Projection};
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_math::{ops, Mat4, Vec3A, Vec4};
use bevy_reflect::prelude::*;
use bevy_transform::components::GlobalTransform;

use crate::{DirectionalLight, DirectionalLightShadowMap};

/// Controls how cascaded shadow mapping works.
/// Prefer using [`CascadeShadowConfigBuilder`] to construct an instance.
///
/// ```
/// # use bevy_light::CascadeShadowConfig;
/// # use bevy_light::CascadeShadowConfigBuilder;
/// # use bevy_utils::default;
/// #
/// let config: CascadeShadowConfig = CascadeShadowConfigBuilder {
///   maximum_distance: 100.0,
///   ..default()
/// }.into();
/// ```
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct CascadeShadowConfig {
    /// The (positive) distance to the far boundary of each cascade.
    pub bounds: Vec<f32>,
    /// The proportion of overlap each cascade has with the previous cascade.
    pub overlap_proportion: f32,
    /// The (positive) distance to the near boundary of the first cascade.
    pub minimum_distance: f32,
}

impl Default for CascadeShadowConfig {
    fn default() -> Self {
        CascadeShadowConfigBuilder::default().into()
    }
}

fn calculate_cascade_bounds(
    num_cascades: usize,
    nearest_bound: f32,
    shadow_maximum_distance: f32,
) -> Vec<f32> {
    if num_cascades == 1 {
        return vec![shadow_maximum_distance];
    }
    let base = ops::powf(
        shadow_maximum_distance / nearest_bound,
        1.0 / (num_cascades - 1) as f32,
    );
    (0..num_cascades)
        .map(|i| nearest_bound * ops::powf(base, i as f32))
        .collect()
}

/// Builder for [`CascadeShadowConfig`].
pub struct CascadeShadowConfigBuilder {
    /// The number of shadow cascades.
    /// More cascades increases shadow quality by mitigating perspective aliasing - a phenomenon where areas
    /// nearer the camera are covered by fewer shadow map texels than areas further from the camera, causing
    /// blocky looking shadows.
    ///
    /// This does come at the cost increased rendering overhead, however this overhead is still less
    /// than if you were to use fewer cascades and much larger shadow map textures to achieve the
    /// same quality level.
    ///
    /// In case rendered geometry covers a relatively narrow and static depth relative to camera, it may
    /// make more sense to use fewer cascades and a higher resolution shadow map texture as perspective aliasing
    /// is not as much an issue. Be sure to adjust `minimum_distance` and `maximum_distance` appropriately.
    pub num_cascades: usize,
    /// The minimum shadow distance, which can help improve the texel resolution of the first cascade.
    /// Areas nearer to the camera than this will likely receive no shadows.
    ///
    /// NOTE: Due to implementation details, this usually does not impact shadow quality as much as
    /// `first_cascade_far_bound` and `maximum_distance`. At many view frustum field-of-views, the
    /// texel resolution of the first cascade is dominated by the width / height of the view frustum plane
    /// at `first_cascade_far_bound` rather than the depth of the frustum from `minimum_distance` to
    /// `first_cascade_far_bound`.
    pub minimum_distance: f32,
    /// The maximum shadow distance.
    /// Areas further from the camera than this will likely receive no shadows.
    pub maximum_distance: f32,
    /// Sets the far bound of the first cascade, relative to the view origin.
    /// In-between cascades will be exponentially spaced relative to the maximum shadow distance.
    /// NOTE: This is ignored if there is only one cascade, the maximum distance takes precedence.
    pub first_cascade_far_bound: f32,
    /// Sets the overlap proportion between cascades.
    /// The overlap is used to make the transition from one cascade's shadow map to the next
    /// less abrupt by blending between both shadow maps.
    pub overlap_proportion: f32,
}

impl CascadeShadowConfigBuilder {
    /// Returns the cascade config as specified by this builder.
    pub fn build(&self) -> CascadeShadowConfig {
        assert!(
            self.num_cascades > 0,
            "num_cascades must be positive, but was {}",
            self.num_cascades
        );
        assert!(
            self.minimum_distance >= 0.0,
            "maximum_distance must be non-negative, but was {}",
            self.minimum_distance
        );
        assert!(
            self.num_cascades == 1 || self.minimum_distance < self.first_cascade_far_bound,
            "minimum_distance must be less than first_cascade_far_bound, but was {}",
            self.minimum_distance
        );
        assert!(
            self.maximum_distance > self.minimum_distance,
            "maximum_distance must be greater than minimum_distance, but was {}",
            self.maximum_distance
        );
        assert!(
            (0.0..1.0).contains(&self.overlap_proportion),
            "overlap_proportion must be in [0.0, 1.0) but was {}",
            self.overlap_proportion
        );
        CascadeShadowConfig {
            bounds: calculate_cascade_bounds(
                self.num_cascades,
                self.first_cascade_far_bound,
                self.maximum_distance,
            ),
            overlap_proportion: self.overlap_proportion,
            minimum_distance: self.minimum_distance,
        }
    }
}

impl Default for CascadeShadowConfigBuilder {
    fn default() -> Self {
        // The defaults are chosen to be similar to be Unity, Unreal, and Godot.
        // Unity: first cascade far bound = 10.05, maximum distance = 150.0
        // Unreal Engine 5: maximum distance = 200.0
        // Godot: first cascade far bound = 10.0, maximum distance = 100.0
        Self {
            // Currently only support one cascade in WebGL 2.
            num_cascades: if cfg!(all(
                feature = "webgl",
                target_arch = "wasm32",
                not(feature = "webgpu")
            )) {
                1
            } else {
                4
            },
            minimum_distance: 0.1,
            maximum_distance: 150.0,
            first_cascade_far_bound: 10.0,
            overlap_proportion: 0.2,
        }
    }
}

impl From<CascadeShadowConfigBuilder> for CascadeShadowConfig {
    fn from(builder: CascadeShadowConfigBuilder) -> Self {
        builder.build()
    }
}

/// A [`DirectionalLight`]'s per-view list of [`Cascade`]s.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct Cascades {
    /// Map from a view to the configuration of each of its [`Cascade`]s.
    pub cascades: EntityHashMap<Vec<Cascade>>,
}

/// A single cascade of a view's shadow map cascade. Several of these are
/// used to cover most of the view to ensure most geometry gets shadows, with
/// some overlap for blending at cascade transitions. Farther away cascades
/// are larger and have a lower effective shadowmap texel per world unit
/// resolution. All cascades have the same pixel dimensions however.
#[derive(Clone, Debug, Default, Reflect)]
#[reflect(Clone, Default)]
pub struct Cascade {
    /// The transform of the light, i.e. the view to world matrix.
    pub world_from_cascade: Mat4,
    /// The orthographic projection for this cascade.
    pub clip_from_cascade: Mat4,
    /// The view-projection matrix for this cascade, converting world space into light clip space.
    /// Importantly, this is derived and stored separately from `view_transform` and `projection` to
    /// ensure shadow stability.
    pub clip_from_world: Mat4,
    /// Size of each shadow map texel in world units.
    pub texel_size: f32,
}

/// sl-client fork (cached-static-shadow-map, scope 2): coverage margin for the
/// retained *static* shadow cascade. The static map covers this multiple of the
/// dynamic cascade's diameter (and depth range) so the camera can pan within the
/// margin before the static map has to be re-baked. Larger values re-bake less
/// often at the cost of coarser static texels.
const STATIC_CASCADE_MARGIN: f32 = 1.5;

/// sl-client fork (cached-static-shadow-map, scope 2): the *retained* projection
/// for one cascade's static shadow layer.
///
/// Unlike [`Cascade`], which is rebuilt every frame to texel-snap onto the moving
/// camera, a `StaticCascade` is **persistent**: it is baked once and reused across
/// frames while the camera stays inside its (margin-expanded) coverage. The
/// shader samples the static depth layer with [`Self::clip_from_world`] — *not* the
/// live per-frame cascade projection — so the retained depths stay aligned with
/// the sample as the camera moves. It is only rebuilt (and the layer re-baked)
/// when the dynamic cascade's coverage would leave the retained coverage, or the
/// light direction changes.
#[derive(Clone, Debug, Reflect)]
#[reflect(Clone)]
pub struct StaticCascade {
    /// The static cascade's view-to-world matrix.
    pub world_from_cascade: Mat4,
    /// The static cascade's orthographic projection.
    pub clip_from_cascade: Mat4,
    /// World space into static-cascade light clip space; the shader samples the
    /// retained static layer with this.
    pub clip_from_world: Mat4,
    /// Size of each static-cascade shadow map texel in world units (coarser than
    /// the dynamic cascade by roughly [`STATIC_CASCADE_MARGIN`]).
    pub texel_size: f32,
    /// The light-space axis-aligned coverage this static cascade was built for,
    /// kept so the next frame can test whether the dynamic cascade still fits.
    pub light_space_min: Vec3A,
    /// The far corner of the light-space coverage (see [`Self::light_space_min`]).
    pub light_space_max: Vec3A,
    /// The light basis (`world_from_light`) this was built with; a change in sun
    /// direction rotates light space and forces a rebuild.
    pub world_from_light: Mat4,
    /// Whether the static depth layer must be re-baked this frame (the projection
    /// was just rebuilt). When false the retained layer is reused untouched.
    pub dirty: bool,
}

/// sl-client fork (cached-static-shadow-map, scope 2): a [`DirectionalLight`]'s
/// per-view list of retained [`StaticCascade`]s, persisted across frames (unlike
/// [`Cascades`], which is cleared and rebuilt every frame).
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct StaticCascades {
    /// Map from a view to the retained static cascade of each of its cascades.
    pub cascades: EntityHashMap<Vec<StaticCascade>>,
}

/// sl-client fork (cached-static-shadow-map, scope 2): build (or reuse) the
/// retained static cascade for one cascade slice.
///
/// Mirrors [`calculate_cascade`], but expands the coverage by
/// [`STATIC_CASCADE_MARGIN`] and reuses `previous` (leaving it `dirty == false`)
/// while the current dynamic coverage still fits inside the retained coverage and
/// the light direction is unchanged. Only when the fit is lost is a fresh,
/// texel-snapped static projection built and marked `dirty`.
fn calculate_static_cascade(
    frustum_corners: [Vec3A; 8],
    cascade_texture_size: f32,
    world_from_light: Mat4,
    light_from_camera: Mat4,
    previous: Option<&StaticCascade>,
) -> StaticCascade {
    let mut min = Vec3A::splat(f32::MAX);
    let mut max = Vec3A::splat(f32::MIN);
    for corner_camera_view in frustum_corners {
        let corner_light_view = light_from_camera.transform_point3a(corner_camera_view);
        min = min.min(corner_light_view);
        max = max.max(corner_light_view);
    }

    // The dynamic cascade's light-space coverage this frame. If it still fits the
    // retained static coverage and the light has not rotated, reuse the retained
    // projection so the static layer stays valid without a re-bake.
    if let Some(prev) = previous {
        let same_light = prev.world_from_light.abs_diff_eq(world_from_light, 1e-5);
        let fits = min.cmpge(prev.light_space_min).all() && max.cmple(prev.light_space_max).all();
        if same_light && fits {
            let mut reused = prev.clone();
            reused.dirty = false;
            return reused;
        }
    }

    // Rebuild: a fresh static projection centered on the current slice, expanded
    // by the margin so the camera can move before the next rebuild.
    let body_diagonal = (frustum_corners[0] - frustum_corners[6]).length_squared();
    let far_plane_diagonal = (frustum_corners[4] - frustum_corners[6]).length_squared();
    let dynamic_diameter = body_diagonal.max(far_plane_diagonal).sqrt().ceil();
    let cascade_diameter = (dynamic_diameter * STATIC_CASCADE_MARGIN).ceil();
    let cascade_texel_size = cascade_diameter / cascade_texture_size;

    // Expand the depth range symmetrically by the same margin so casters just in
    // front of / behind the slice stay covered as the camera moves.
    let z_margin = 0.5 * (max.z - min.z) * (STATIC_CASCADE_MARGIN - 1.0);
    let near_z = max.z + z_margin;
    let far_z = min.z - z_margin;

    // Texel-snap the center (as [`calculate_cascade`] does) so the retained map is
    // itself shimmer-free when it is baked.
    let near_plane_center = Vec3A::new(
        (0.5 * (min.x + max.x) / cascade_texel_size).floor() * cascade_texel_size,
        (0.5 * (min.y + max.y) / cascade_texel_size).floor() * cascade_texel_size,
        near_z,
    );

    let world_from_light_transpose = world_from_light.transpose();
    let cascade_from_world = Mat4::from_cols(
        world_from_light_transpose.x_axis,
        world_from_light_transpose.y_axis,
        world_from_light_transpose.z_axis,
        (-near_plane_center).extend(1.0),
    );
    let world_from_cascade = Mat4::from_cols(
        world_from_light.x_axis,
        world_from_light.y_axis,
        world_from_light.z_axis,
        world_from_light * near_plane_center.extend(1.0),
    );

    let r = (near_z - far_z).recip();
    let clip_from_cascade = Mat4::from_cols(
        Vec4::new(2.0 / cascade_diameter, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / cascade_diameter, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    );
    let clip_from_world = clip_from_cascade * cascade_from_world;

    // The light-space coverage the retained map now spans (used next frame for
    // the fit test): the snapped square in x/y and the margin-expanded depth.
    let half = 0.5 * cascade_diameter;
    let light_space_min = Vec3A::new(
        near_plane_center.x - half,
        near_plane_center.y - half,
        far_z,
    );
    let light_space_max = Vec3A::new(
        near_plane_center.x + half,
        near_plane_center.y + half,
        near_z,
    );

    StaticCascade {
        world_from_cascade,
        clip_from_cascade,
        clip_from_world,
        texel_size: cascade_texel_size,
        light_space_min,
        light_space_max,
        world_from_light,
        dirty: true,
    }
}

/// Sets up [`Cascades`] for all shadow mapped [`DirectionalLight`]s.
pub fn build_directional_light_cascades(
    directional_light_shadow_map: Res<DirectionalLightShadowMap>,
    views: Query<(Entity, &GlobalTransform, &Projection, &Camera)>,
    mut lights: Query<(
        &GlobalTransform,
        &DirectionalLight,
        &CascadeShadowConfig,
        &mut Cascades,
        // sl-client fork (cached-static-shadow-map, scope 2): the retained static
        // projections, persisted across frames alongside the per-frame cascades.
        &mut StaticCascades,
    )>,
) {
    let views = views
        .iter()
        .filter_map(|(entity, transform, projection, camera)| {
            if camera.is_active {
                Some((entity, projection, transform.to_matrix()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    for (transform, directional_light, cascades_config, mut cascades, mut static_cascades) in
        &mut lights
    {
        if !directional_light.shadow_maps_enabled {
            continue;
        }
        cascades.cascades.clear();
        // sl-client fork (cached-static-shadow-map, scope 2): the retained static
        // projections persist across frames, so rather than clearing we take the
        // previous map out and read each view's prior state back when rebuilding.
        // Views that vanished simply drop out (they are not re-inserted).
        let previous_static = core::mem::take(&mut static_cascades.cascades);

        // It is very important to the numerical and thus visual stability of shadows that
        // `world_from_light` has orthogonal upper-left 3x3 and zero translation.
        // Even though only the direction (i.e. rotation) of the light matters, we don't constrain
        // users to not change any other aspects of the transform - there's no guarantee
        // `transform.to_matrix()` will give us a matrix with our desired properties.
        // Instead, we directly create a good matrix from just the rotation.
        let world_from_light = Mat4::from_quat(transform.rotation());
        // The transpose is the inverse for orthogonal matrices.
        let light_from_world = world_from_light.transpose();

        for (view_entity, projection, world_from_view) in views.iter().copied() {
            let light_view_from_camera = light_from_world * world_from_view;
            let overlap_factor = 1.0 - cascades_config.overlap_proportion;
            let far_bounds = cascades_config.bounds.iter();
            let near_bounds = [cascades_config.minimum_distance]
                .into_iter()
                .chain(far_bounds.clone().map(|bound| overlap_factor * bound));
            // sl-client fork (cached-static-shadow-map, scope 2): the retained
            // static projection for each cascade, reusing this view's prior state
            // for the coverage-fit test.
            let previous_view_static = previous_static.get(&view_entity);
            let mut view_static_cascades = Vec::with_capacity(cascades_config.bounds.len());
            let view_cascades = near_bounds
                .zip(far_bounds)
                .enumerate()
                .map(|(cascade_index, (near_bound, far_bound))| {
                    // Negate bounds as -z is camera forward direction.
                    let corners = projection.get_frustum_corners(-near_bound, -far_bound);
                    view_static_cascades.push(calculate_static_cascade(
                        corners,
                        directional_light_shadow_map.size as f32,
                        world_from_light,
                        light_view_from_camera,
                        previous_view_static.and_then(|prev| prev.get(cascade_index)),
                    ));
                    calculate_cascade(
                        corners,
                        directional_light_shadow_map.size as f32,
                        world_from_light,
                        light_view_from_camera,
                    )
                })
                .collect();
            cascades.cascades.insert(view_entity, view_cascades);
            static_cascades
                .cascades
                .insert(view_entity, view_static_cascades);
        }
    }
}

/// Returns a [`Cascade`] for the frustum defined by `frustum_corners`.
///
/// The corner vertices should be specified in the following order:
/// first the bottom right, top right, top left, bottom left for the near plane, then similar for the far plane.
///
/// See this [reference](https://developer.download.nvidia.com/SDK/10.5/opengl/src/cascaded_shadow_maps/doc/cascaded_shadow_maps.pdf) for more details.
fn calculate_cascade(
    frustum_corners: [Vec3A; 8],
    cascade_texture_size: f32,
    world_from_light: Mat4,
    light_from_camera: Mat4,
) -> Cascade {
    let mut min = Vec3A::splat(f32::MAX);
    let mut max = Vec3A::splat(f32::MIN);
    for corner_camera_view in frustum_corners {
        let corner_light_view = light_from_camera.transform_point3a(corner_camera_view);
        min = min.min(corner_light_view);
        max = max.max(corner_light_view);
    }

    // NOTE: Use the larger of the frustum slice far plane diagonal and body diagonal lengths as this
    //       will be the maximum possible projection size. Use the ceiling to get an integer which is
    //       very important for floating point stability later. It is also important that these are
    //       calculated using the original camera space corner positions for floating point precision
    //       as even though the lengths using corner_light_view above should be the same, precision can
    //       introduce small but significant differences.
    // NOTE: The size remains the same unless the view frustum or cascade configuration is modified.
    let body_diagonal = (frustum_corners[0] - frustum_corners[6]).length_squared();
    let far_plane_diagonal = (frustum_corners[4] - frustum_corners[6]).length_squared();
    let cascade_diameter = body_diagonal.max(far_plane_diagonal).sqrt().ceil();

    // NOTE: If we ensure that cascade_texture_size is a power of 2, then as we made cascade_diameter an
    //       integer, cascade_texel_size is then an integer multiple of a power of 2 and can be
    //       exactly represented in a floating point value.
    let cascade_texel_size = cascade_diameter / cascade_texture_size;
    // NOTE: For shadow stability it is very important that the near_plane_center is at integer
    //       multiples of the texel size to be exactly representable in a floating point value.
    let near_plane_center = Vec3A::new(
        (0.5 * (min.x + max.x) / cascade_texel_size).floor() * cascade_texel_size,
        (0.5 * (min.y + max.y) / cascade_texel_size).floor() * cascade_texel_size,
        // NOTE: max.z is the near plane for right-handed y-up
        max.z,
    );

    // It is critical for `cascade_from_world` to be stable. So rather than forming `world_from_cascade`
    // and inverting it, which risks instability due to numerical precision, we directly form
    // `cascade_from_world` as the reference material suggests.
    let world_from_light_transpose = world_from_light.transpose();
    let cascade_from_world = Mat4::from_cols(
        world_from_light_transpose.x_axis,
        world_from_light_transpose.y_axis,
        world_from_light_transpose.z_axis,
        (-near_plane_center).extend(1.0),
    );
    let world_from_cascade = Mat4::from_cols(
        world_from_light.x_axis,
        world_from_light.y_axis,
        world_from_light.z_axis,
        world_from_light * near_plane_center.extend(1.0),
    );

    // Right-handed orthographic projection, centered at `near_plane_center`.
    // NOTE: This is different from the reference material, as we use reverse Z.
    let r = (max.z - min.z).recip();
    let clip_from_cascade = Mat4::from_cols(
        Vec4::new(2.0 / cascade_diameter, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / cascade_diameter, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    );

    let clip_from_world = clip_from_cascade * cascade_from_world;
    Cascade {
        world_from_cascade,
        clip_from_cascade,
        clip_from_world,
        texel_size: cascade_texel_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eight corners of an axis-aligned cube centered at `center`, in the corner
    /// order `calculate_*_cascade` expects (near bottom-right, top-right, top-left,
    /// bottom-left; then the same four for the far plane).
    fn cube_corners(center: Vec3A, half: f32) -> [Vec3A; 8] {
        [
            center + Vec3A::new(half, -half, half),
            center + Vec3A::new(half, half, half),
            center + Vec3A::new(-half, half, half),
            center + Vec3A::new(-half, -half, half),
            center + Vec3A::new(half, -half, -half),
            center + Vec3A::new(half, half, -half),
            center + Vec3A::new(-half, half, -half),
            center + Vec3A::new(-half, -half, -half),
        ]
    }

    /// The retained static cascade is reused (not `dirty`, same projection) while
    /// the camera stays inside the margin, and rebuilt once it leaves.
    #[test]
    fn static_cascade_reused_within_margin_rebuilt_beyond() {
        let size = 4096.0;
        let light = Mat4::IDENTITY;
        let corners = cube_corners(Vec3A::ZERO, 5.0);

        // First build: no previous → a fresh bake.
        let first = calculate_static_cascade(corners, size, light, light, None);
        assert!(first.dirty, "a fresh static cascade must bake");

        // Unchanged coverage → reused, not dirty, identical projection.
        let again = calculate_static_cascade(corners, size, light, light, Some(&first));
        assert!(
            !again.dirty,
            "unchanged coverage must reuse the retained bake"
        );
        assert_eq!(again.clip_from_world, first.clip_from_world);

        // A small move (well inside the margin) still fits → reused.
        let nudged = cube_corners(Vec3A::new(1.0, 0.0, 0.0), 5.0);
        let small = calculate_static_cascade(nudged, size, light, light, Some(&first));
        assert!(!small.dirty, "a move within the margin must reuse the bake");
        assert_eq!(small.clip_from_world, first.clip_from_world);

        // A large move leaves the coverage → rebuilt with a new projection.
        let far = cube_corners(Vec3A::new(100.0, 0.0, 0.0), 5.0);
        let rebuilt = calculate_static_cascade(far, size, light, light, Some(&first));
        assert!(
            rebuilt.dirty,
            "a move beyond the margin must rebuild the bake"
        );
        assert_ne!(rebuilt.clip_from_world, first.clip_from_world);
    }

    /// A change in sun direction rotates light space and forces a rebuild even
    /// when the camera-space coverage is unchanged.
    #[test]
    fn static_cascade_rebuilt_when_light_rotates() {
        let size = 4096.0;
        let light = Mat4::IDENTITY;
        let corners = cube_corners(Vec3A::ZERO, 5.0);
        let first = calculate_static_cascade(corners, size, light, light, None);

        let rotated = Mat4::from_rotation_y(0.5);
        let after = calculate_static_cascade(corners, size, rotated, light, Some(&first));
        assert!(
            after.dirty,
            "a sun rotation must rebuild the static cascade"
        );
    }

    /// The static cascade covers a strictly larger light-space area than the
    /// dynamic cascade fit to the same frustum slice (the margin), so a static
    /// caster just outside the dynamic cascade still lands in the retained bake.
    #[test]
    fn static_cascade_is_larger_than_dynamic() {
        let size = 4096.0;
        let light = Mat4::IDENTITY;
        let corners = cube_corners(Vec3A::ZERO, 5.0);

        let dynamic = calculate_cascade(corners, size, light, light);
        let stat = calculate_static_cascade(corners, size, light, light, None);
        assert!(
            stat.texel_size > dynamic.texel_size,
            "the margin-expanded static cascade must have coarser texels ({} vs {})",
            stat.texel_size,
            dynamic.texel_size,
        );
    }
}
