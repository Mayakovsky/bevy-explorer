//! Emanate: the Synth plays by what players see. On empty parcels the desktop app (Unity) draws its own ground and
//! scenery; this gives the headless Synth the same colliders, so it neither floats over hills nor passes through rocks.
//!
//! Data (`--landscape-data <dir>`, default off): `GenesisTrees.bin` (Unity's own prop placement, Apache-2.0, never committed)
//! and `occupied.txt` (one `x y` per line: the realm's occupied parcels, snapshotted at launch).
//! Ground: `common::unity_terrain` (realm-wide chamfer distance + mountain noise, exactly Unity). Rocks: the stones from
//! `GenesisTrees.bin` that survive Unity's occupied-parcel overlap rule, as capsule colliders. Tree trunks and Bush02 are not yet ported.

use std::{collections::HashSet, path::PathBuf};

use bevy::prelude::*;
use common::{
    genesis_props::{place, rock_capsule, GenesisProps, PARCEL},
    sets::PostUpdateSets,
    structs::PrimaryUser,
    unity_terrain::UnityTerrain,
};
use dcl_component::SceneEntityId;
use dcl_component::proto_components::sdk::components::ColliderLayer;
use scene_runner::update_world::mesh_collider::{ColliderId, GlobalGroundCollider, SceneColliderData, GROUND_COLLISION_MASK};

/// Side of the ground patch in metres (1 m spacing) and the grid it recentres on, as the full client does.
const PATCH: i32 = 64;
const RECENTRE: f32 = 16.0;
/// Rocks are built over the patch plus this margin so one is never cut in half at the edge.
const MARGIN: f32 = 8.0;

pub struct LandscapePlugin {
    pub dir: PathBuf,
}

#[derive(Resource)]
struct Landscape {
    terrain: UnityTerrain,
    props: GenesisProps,
}

#[derive(Component, Default)]
struct Patch {
    centre: Option<IVec2>,
    built: HashSet<ColliderId>,
}

impl Plugin for LandscapePlugin {
    fn build(&self, app: &mut App) {
        match load(&self.dir) {
            Ok(landscape) => {
                info!(
                    "[landscape] on: {} occupied parcels, terrain bounds {}..{}, {} max steps, {} props",
                    landscape.terrain.max_parcel.x - landscape.terrain.min_parcel.x + 1,
                    landscape.terrain.min_parcel,
                    landscape.terrain.max_parcel,
                    landscape.terrain.max_steps,
                    landscape.props.len()
                );
                app.insert_resource(landscape)
                    .add_systems(Startup, setup)
                    .add_systems(PostUpdate, update_patch.in_set(PostUpdateSets::ColliderUpdate))
                    .add_systems(PostUpdate, floor_synth.after(PostUpdateSets::PlayerUpdate));
            }
            Err(e) => error!("[landscape] disabled: {e}"),
        }
    }
}

fn load(dir: &std::path::Path) -> Result<Landscape, String> {
    let bytes = std::fs::read(dir.join("GenesisTrees.bin")).map_err(|e| format!("GenesisTrees.bin: {e}"))?;
    let props = GenesisProps::parse(&bytes)?;
    let text = std::fs::read_to_string(dir.join("occupied.txt")).map_err(|e| format!("occupied.txt: {e}"))?;
    let mut owned = HashSet::new();
    for line in text.lines() {
        let mut it = line.split_whitespace().map(|v| v.parse::<i32>());
        if let (Some(Ok(x)), Some(Ok(y))) = (it.next(), it.next()) {
            owned.insert(IVec2::new(x, y));
        }
    }
    let terrain = UnityTerrain::build(&owned).ok_or("occupied.txt has no parcels")?;
    Ok(Landscape { terrain, props })
}

fn setup(mut commands: Commands) {
    commands.spawn((Name::new("Landscape collider"), Patch::default(), SceneColliderData::default(), GlobalGroundCollider));
}

fn ground_id() -> ColliderId {
    ColliderId::new(SceneEntityId::ROOT, Some("landscape ground".into()), 0)
}

fn rock_id(parcel: IVec2, index: usize) -> ColliderId {
    ColliderId::new(SceneEntityId::ROOT, Some(format!("landscape rock {} {} {index}", parcel.x, parcel.y)), 0)
}

/// Rebuild the ground heightfield and the rocks around the Synth when it leaves the current patch.
fn update_patch(
    landscape: Res<Landscape>,
    player: Query<&GlobalTransform, With<PrimaryUser>>,
    mut patch: Single<(&mut Patch, &mut SceneColliderData)>,
) {
    let Ok(player) = player.single() else { return };
    let position = player.translation();
    if !position.is_finite() {
        return;
    }
    let (state, data) = &mut *patch;
    let centre = match state.centre {
        Some(c) if (position.xz() - c.as_vec2()).abs().max_element() < RECENTRE => c,
        _ => ((position.xz() / RECENTRE).round() * RECENTRE).as_ivec2(),
    };
    if state.centre == Some(centre) {
        return;
    }
    state.centre = Some(centre);
    let half = PATCH / 2;
    // ground: rows along +z (Bevy), Unity z is the negated Bevy z
    let side = (PATCH + 1) as usize;
    let mut heights = Vec::with_capacity(side * side);
    for row in 0..=PATCH {
        for col in 0..=PATCH {
            heights.push(landscape.terrain.height((centre.x - half + col) as f32, -((centre.y - half + row) as f32)));
        }
    }
    data.set_ground_heightfield(&ground_id(), side, side, &heights, Vec2::splat(PATCH as f32), Vec3::new(centre.x as f32, 0.0, centre.y as f32));
    // rocks: parcels (Unity) covering the patch plus a margin
    let (x0, x1) = (centre.x as f32 - half as f32 - MARGIN, centre.x as f32 + half as f32 + MARGIN);
    let (z0, z1) = (-(centre.y as f32) - half as f32 - MARGIN, -(centre.y as f32) + half as f32 + MARGIN); // Unity z
    let (p0, p1) = (
        IVec2::new((x0 / PARCEL).floor() as i32, (z0 / PARCEL).floor() as i32),
        IVec2::new((x1 / PARCEL).floor() as i32, (z1 / PARCEL).floor() as i32),
    );
    let occupied = |p: IVec2| landscape.terrain.is_occupied(p);
    let mut current = HashSet::new();
    let mut sample: Option<(Vec3, Vec3, f32)> = None;
    for py in p0.y..=p1.y {
        for px in p0.x..=p1.x {
            let parcel = IVec2::new(px, py);
            for (index, instance) in landscape.props.parcel(parcel).iter().enumerate() {
                let Some(placed) = place(parcel, instance, &occupied) else { continue };
                let ground = landscape.terrain.height(placed.position_xz.x, placed.position_xz.y);
                let Some((a, b, radius)) = rock_capsule(&placed, ground) else { continue };
                let id = rock_id(parcel, index);
                // Unity (x, y, z) to Bevy (x, y, -z)
                info!("[landscape] rock {:?} at unity ({:.1}, {:.1}) y {:.2}..{:.2} r {:.2}", id, a.x, a.z, a.y, b.y, radius);
                data.set_physics_capsule(&id, Vec3::new(a.x, a.y, -a.z), Vec3::new(b.x, b.y, -b.z), radius);
                sample = Some((Vec3::new(a.x, a.y, -a.z), Vec3::new(b.x, b.y, -b.z), radius));
                current.insert(id);
            }
        }
    }
    info!("[landscape] patch centre ({}, {}) unity parcels {}..{}: {} rocks built, ground {:.2}..{:.2}", centre.x, centre.y, p0, p1, current.len(), heights.iter().cloned().fold(f32::MAX, f32::min), heights.iter().cloned().fold(f32::MIN, f32::max));
    // diagnostic: sweep an avatar east into the nearest rock and say what the colliders do with it
    if let Some(rock) = current.iter().next() {
        let _ = rock;
    }
    for id in state.built.difference(&current) {
        data.remove_collider(id);
    }
    state.built = current;
    // diagnostic: an avatar sweeps east at its feet height into the last rock built, and the result is logged
    if let Some((a, _b, r)) = sample {
        let origin = bevy::math::DVec3::new(a.x as f64 - 8.0, (a.y.max(0.0) + 0.3) as f64, a.z as f64);
        let hit = data.cast_avatar_nearest(origin, bevy::math::DVec3::X, 16.0, ColliderLayer::ClPhysics as u32 | GROUND_COLLISION_MASK, false, false, Default::default(), 0.0);
        info!("[landscape] sweep from {:?} east 16 m toward a rock at x {:.1} (r {:.2}): {:?}", origin, a.x, r, hit.map(|h| (h.toi, h.normal)));
    }
}

/// Keep the Synth on or above the ground of an empty parcel (movement already floors at zero), like the full client's `floor_player`:
/// it rides rising ground up instead of fighting it through collision.
fn floor_synth(landscape: Res<Landscape>, mut player: Query<&mut Transform, With<PrimaryUser>>, time: Res<Time>, mut next_log: Local<f32>) {
    let Ok(mut transform) = player.single_mut() else { return };
    // one line a second so a live test can compare the Synth's feet with the closed-form ground
    if time.elapsed_secs() >= *next_log {
        *next_log = time.elapsed_secs() + 1.0;
        let p = transform.translation;
        info!("[landscape] feet ({:.2}, {:.2}, {:.2}) unity-z {:.2} ground {:.3}", p.x, p.y, p.z, -p.z, landscape.terrain.height(p.x, -p.z));
    }
    let p = transform.translation;
    if !p.is_finite() {
        return;
    }
    let parcel = IVec2::new((p.x / PARCEL).floor() as i32, (-p.z / PARCEL).floor() as i32);
    if landscape.terrain.is_occupied(parcel) {
        return;
    }
    let ground = landscape.terrain.height(p.x, -p.z);
    if p.y < ground {
        transform.translation.y = ground;
    }
}
