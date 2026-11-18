//! Emanate: the Synth walks real routes. `nav_to` plans a path over the colliders the engine has loaded (scenes plus the landscape; see
//! `common::nav`) and follows it waypoint by waypoint through the normal movement physics, jumping at stairs and replanning when blocked.
//! `nav_spots` lists the spots the Synth can reach (for the server's hiding choices: high or low, near or far).
//! Events: `nav` ({"status": "planned"|"failed"|"arrived"|"stuck"|"replanning"}) and `nav_spots`.

use std::time::Instant;

use bevy::prelude::*;
use common::{
    nav::{glide_options, glide_to_goal, GlideChoice, NavCache, NavWorld, PlanConfig, ReachSearch, PlanError, Planner, Waypoint, Move, AVATAR_RADIUS},
    structs::PrimaryUser,
};
use dcl_component::proto_components::sdk::components::ColliderLayer;
use scene_runner::update_world::mesh_collider::{SceneColliderData, GROUND_COLLISION_MASK};
use serde_json::json;

use super::{dcl_position, emit, to_bevy, Goal, SynthState, GLIDE_FALL, GLIDE_SPEED};
use crate::headless::landscape::Landscape;

const MASK: u32 = ColliderLayer::ClPhysics as u32 | GROUND_COLLISION_MASK;
/// A waypoint counts as reached within this horizontal distance (the last one: the arrive radius of the walk goal).
const REACH: f32 = 0.9;
/// a jump that rises more than this is a tall (double) jump: lined up on the takeoff cell, then straight up
const TALL_JUMP_RISE: f32 = 2.5;
const LINEUP: f32 = 0.5;
const MAX_REPLANS: u32 = 4;
const GIVE_UP_SECS: f32 = 150.0;

#[derive(Clone, Debug)]
pub enum NavRequest {
    To { target: Vec3, level: bool, run: bool },
    Spots { max_cost: f32 },
    /// Glide off a high place away from a threat at `away` (bevy x/z).
    Glide { away: Vec2 },
    /// Glide from here to a lower goal (bevy position); fails cleanly when it cannot be reached by gliding.
    GlideTo { target: Vec3 },
    /// Diagnostics: the surfaces and the blocked-by-height picture of a patch of the world around `center` (bevy x/z), as the planner sees it.
    Probe { center: Vec2, half: f32, y_lo: f32, y_hi: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GlidePhase {
    Run,
    Takeoff,
    Air,
}

#[derive(Clone, Copy, Debug)]
pub struct GlideRun {
    pub choice: GlideChoice,
    pub phase: GlidePhase,
    pub since: f32,
    pub airborne_since: Option<f32>,
}

#[derive(Resource, Default)]
pub struct Nav {
    pub pending: Option<NavRequest>,
    pub path: Vec<Waypoint>,
    pub idx: usize,
    pub run: bool,
    pub goal: Option<(Vec3, bool)>, // target and whether the y counts
    pub active: bool,
    pub replans: u32,
    pub blocks_here: u32,
    pub last_jump: f32,
    pub started: f32,
    pub id: u32,
    /// the walk goal was set by us last frame (so a missing goal now means the movement system dropped it: blocked)
    pub just_set: bool,
    pub glide: Option<GlideRun>,
    /// the Synth's power level (set by the `level` command): the planner's jumping moves depend on it
    pub level: u8,
    /// a reachable-spot search in progress (a slice of it runs every frame)
    pub spots_job: Option<SpotsJob>,
}

pub struct SpotsJob {
    search: ReachSearch,
    cache: NavCache,
    from: Vec3,
    started: Instant,
    work_ms: f32,
    frames: u32,
}

pub struct EngineWorld<'a, 'b> {
    scenes: &'a mut Vec<Mut<'b, SceneColliderData>>,
    landscape: Option<&'a Landscape>,
}

impl<'a, 'b> EngineWorld<'a, 'b> {
    pub fn new(scenes: &'a mut Vec<Mut<'b, SceneColliderData>>, landscape: Option<&'a Landscape>) -> Self {
        EngineWorld { scenes, landscape }
    }
}

impl NavWorld for EngineWorld<'_, '_> {
    fn surfaces(&mut self, x: f32, z: f32, from_y: f32) -> Vec<f32> {
        let top = from_y.min(900.0);
        let mut ys: Vec<f32> = Vec::new();
        for s in self.scenes.iter_mut() {
            ys.extend(s.surfaces_below(x, z, top, MASK, 0.5));
        }
        if let Some(g) = self.landscape.and_then(|l| l.ground(x, z)) {
            if g <= top {
                ys.push(g);
            }
        }
        ys.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        ys.dedup_by(|a, b| (*a - *b).abs() < 0.12);
        ys
    }
    fn blocked(&mut self, x: f32, z: f32, y0: f32, y1: f32) -> bool {
        self.scenes.iter_mut().any(|s| s.cylinder_blocked(x, z, y0, y1, AVATAR_RADIUS, MASK))
    }
}

/// planner limits at a power level: the highest ledge a jump gets up onto is about 1.4 m normally and about 8 m with the Jump level's double jump (measured: a double jump 0.4 s after the jump peaks at 10.7 m)
fn cfg(level: u8) -> PlanConfig {
    PlanConfig { radius_cells: 105, max_nodes: 60_000, goal_tolerance: 1.2, jump_rise: if level >= 2 { 8.0 } else { 1.4 }, ..Default::default() }
}

fn dcl_list(path: &[Waypoint]) -> Vec<serde_json::Value> {
    path.iter()
        .take(80)
        .map(|w| {
            let p = dcl_position(w.pos);
            json!({"x": (p[0] * 100.0).round() / 100.0, "y": (p[1] * 100.0).round() / 100.0, "z": (p[2] * 100.0).round() / 100.0, "how": format!("{:?}", w.how).to_lowercase()})
        })
        .collect()
}

fn start_glide(nav: &mut Nav, choice: Option<GlideChoice>, options: usize, from: Vec3, ms: u64, now: f32, kind: &str) {
    match choice {
        Some(choice) => {
            emit(json!({"event": "glide", "status": "planned", "kind": kind, "options": options, "from": dcl_position(from), "ledge": dcl_position(choice.ledge), "landing": dcl_position(choice.landing), "height": from.y - choice.landing.y, "ms": ms}));
            nav.active = false;
            nav.glide = Some(GlideRun { choice, phase: GlidePhase::Run, since: now, airborne_since: None });
        }
        None => emit(json!({"event": "glide", "status": "failed", "kind": kind, "reason": "no_glide", "options": options, "from": dcl_position(from), "ms": ms})),
    }
}

pub fn nav_plan(
    mut nav: ResMut<Nav>,
    mut scenes: Query<&mut SceneColliderData>,
    landscape: Option<Res<Landscape>>,
    player: Query<&Transform, With<PrimaryUser>>,
    time: Res<Time>,
) {
    let Some(request) = nav.pending.take() else { return };
    let Ok(transform) = player.single() else {
        emit(json!({"event": "nav", "status": "failed", "reason": "player not ready"}));
        return;
    };
    let from = transform.translation;
    let mut guards: Vec<Mut<SceneColliderData>> = scenes.iter_mut().collect();
    let mut world = EngineWorld { scenes: &mut guards, landscape: landscape.as_deref() };
    let t0 = Instant::now();
    match request {
        NavRequest::To { target, level, run } => {
            let result = Planner::new(&mut world).plan(from, target, level, &cfg(nav.level));
            let ms = t0.elapsed().as_millis() as u64;
            match result {
                Ok(plan) => {
                    let mut wps = plan.waypoints;
                    if wps.len() > 1 && Vec2::new(wps[0].pos.x - from.x, wps[0].pos.z - from.z).length() < REACH {
                        wps.remove(0);
                    }
                    nav.id += 1;
                    emit(json!({"event": "nav", "status": if nav.replans > 0 { "replanned" } else { "planned" }, "id": nav.id, "waypoints": dcl_list(&wps), "count": wps.len(), "cost": plan.cost, "expanded": plan.expanded, "ms": ms}));
                    nav.path = wps;
                    nav.idx = 0;
                    nav.run = run;
                    nav.goal = Some((target, level));
                    nav.active = true;
                    nav.blocks_here = 0;
                    if nav.replans == 0 {
                        nav.started = time.elapsed_secs();
                    }
                }
                Err(e) => {
                    let reason = match e {
                        PlanError::NoStart => "no_start".to_string(),
                        PlanError::NoGoal => "no_goal".to_string(),
                        PlanError::NoPath { expanded } => format!("no_path (searched {expanded})"),
                    };
                    emit(json!({"event": "nav", "status": "failed", "reason": reason, "ms": ms, "from": dcl_position(from), "goal": dcl_position(target)}));
                    nav.active = false;
                }
            }
        }
        NavRequest::Glide { away } => {
            let choices = glide_options(&mut world, from, 16, &[12.0, 20.0, 30.0, 45.0, 60.0], GLIDE_SPEED, GLIDE_FALL);
            let ms = t0.elapsed().as_millis() as u64;
            // the landing farthest from the threat, then the shortest run to the ledge
            let score = |a: &GlideChoice| Vec2::new(a.landing.x - away.x, a.landing.z - away.y).length() - 0.4 * Vec2::new(a.ledge.x - from.x, a.ledge.z - from.z).length();
            let best = choices.iter().copied().max_by(|a, b| score(a).partial_cmp(&score(b)).unwrap_or(std::cmp::Ordering::Equal));
            start_glide(&mut nav, best, choices.len(), from, ms, time.elapsed_secs(), "escape");
        }
        NavRequest::GlideTo { target } => {
            let choice = glide_to_goal(&mut world, from, target, GLIDE_SPEED, GLIDE_FALL);
            let ms = t0.elapsed().as_millis() as u64;
            start_glide(&mut nav, choice, usize::from(choice.is_some()), from, ms, time.elapsed_secs(), "to");
        }
        NavRequest::Probe { center, half, y_lo, y_hi } => {
            let mut rows = Vec::new();
            let steps = (half * 2.0 / 0.5) as i32;
            for iz in 0..=steps {
                for ix in 0..=steps {
                    let (x, z) = (center.x - half + ix as f32 * 0.5, center.y - half + iz as f32 * 0.5);
                    let surf: Vec<f32> = world.surfaces(x, z, y_hi).into_iter().filter(|y| *y >= y_lo - 0.5).take(4).map(|y| (y * 100.0).round() / 100.0).collect();
                    // blocked by the planner's cylinder (radius 0.3) for a body from y to y+2, every 0.5 m of height
                    let mut b = String::new();
                    let mut y = y_lo;
                    while y <= y_hi {
                        b.push(if world.blocked(x, z, y, y + 2.0) { '#' } else { '.' });
                        y += 0.5;
                    }
                    let d = dcl_position(Vec3::new(x, 0.0, z));
                    rows.push(json!([d[0], d[2], surf, b]));
                }
            }
            emit(json!({"event": "nav_probe", "y_lo": y_lo, "y_hi": y_hi, "step_y": 0.5, "rows": rows}));
        }
        NavRequest::Spots { max_cost } => {
            // the spot list reaches farther than a route plan does (a long run, up to about 100 m, needs spots near the top of its budget). It is computed a slice of
            // nodes per frame (see `nav_spots_step`) so the Synth keeps moving or hiding while the list is built, however long a chase is going on.
            let spot_cfg = PlanConfig { radius_cells: 105, max_nodes: 30_000, ..cfg(nav.level) };
            let mut planner = Planner::new(&mut world);
            match planner.reach_start(from, max_cost, &spot_cfg) {
                Some(search) => {
                    let cache = planner.into_cache();
                    nav.spots_job = Some(SpotsJob { search, cache, from, started: Instant::now(), work_ms: t0.elapsed().as_secs_f32() * 1000.0, frames: 0 });
                }
                None => emit(json!({"event": "nav_spots", "count": 0, "from": dcl_position(from), "spots": [], "ms": 0})),
            }
        }
    }
}

pub fn nav_follow(
    mut nav: ResMut<Nav>,
    mut state: ResMut<SynthState>,
    player: Query<&Transform, With<PrimaryUser>>,
    time: Res<Time>,
) {
    if !nav.active {
        return;
    }
    let Ok(transform) = player.single() else { return };
    let pos = transform.translation;
    let now = time.elapsed_secs();
    if now - nav.started > GIVE_UP_SECS {
        nav.active = false;
        state.goal = None;
        emit(json!({"event": "nav", "status": "stuck", "reason": "timeout", "position": dcl_position(pos)}));
        return;
    }
    while nav.idx < nav.path.len() {
        let wp = nav.path[nav.idx];
        let last = nav.idx + 1 == nav.path.len();
        let dh = Vec2::new(wp.pos.x - pos.x, wp.pos.z - pos.z).length();
        // round 24: a waypoint is reached HORIZONTALLY and at its HEIGHT. The old rule was horizontal only: a double jump onto a roof reaches the spot's column at the apex of the arc
        // (the live tests logged arrivals at 12.5 to 14.2 m for a 9.6 m spot) and reported `arrived` in mid-air, which the brain rightly refused as a wrong-height arrival. The last waypoint
        // (and any jump target) must also be within 1.5 m of its height, and the last one on the ground.
        let vert_ok = (pos.y - wp.pos.y).abs() < 1.5 && (!last || state.grounded);
        let needs_vert = last || wp.how == Move::Jump;
        if dh < if last { 0.6 } else { REACH } && (vert_ok || !needs_vert) {
            nav.idx += 1;
            nav.blocks_here = 0;
        } else {
            break;
        }
    }
    if nav.idx >= nav.path.len() {
        nav.active = false;
        nav.replans = 0;
        state.goal = None;
        emit(json!({"event": "nav", "status": "arrived", "position": dcl_position(pos)}));
        return;
    }
    let wp = nav.path[nav.idx];
    let dh = Vec2::new(wp.pos.x - pos.x, wp.pos.z - pos.z).length();
    // round 24 (level 2 probe, 18 Nov): a double jump overshoots the ledge (the apex was 14 to 17 m for a 9.6 m spot) and the walk goal clears itself when the avatar is over the goal's column:
    // that is an ARRIVAL IN THE AIR, not a block. Wait for the landing (the waypoint check above finishes the route once it is grounded at the height); replanning here made four
    // replans and a false `stuck/blocked` while the avatar was still falling onto the right level.
    let last = nav.idx + 1 == nav.path.len();
    if !state.grounded && pos.y > wp.pos.y + 1.0 && dh < if last { 0.9 } else { REACH + 0.5 } {
        state.goal = None;
        nav.just_set = false;
        return;
    }
    // blocked by the movement system (it cleared the goal): jump first, then replan
    if state.goal.is_none() && nav.just_set {
        nav.blocks_here += 1;
        if nav.blocks_here <= 2 && state.grounded {
            state.jump_requested = true;
            nav.last_jump = now;
        } else if nav.replans < MAX_REPLANS {
            nav.replans += 1;
            emit(json!({"event": "nav", "status": "replanning", "attempt": nav.replans, "position": dcl_position(pos)}));
            if let Some((t, level)) = nav.goal {
                nav.pending = Some(NavRequest::To { target: t, level, run: nav.run });
            }
            nav.just_set = false;
            return;
        } else {
            nav.active = false;
            nav.replans = 0;
            emit(json!({"event": "nav", "status": "stuck", "reason": "blocked", "position": dcl_position(pos)}));
            return;
        }
    }
    // a jump move: up onto a ledge or platform: jump near the foot of it, and with the Jump level's double jump near the top of the first jump
    let mut aim = wp.pos;
    if wp.how == Move::Jump {
        let rise = wp.pos.y - pos.y;
        // a TALL jump (the double jump) goes up first and over second: it lines up on the takeoff cell, stands still, jumps straight up and only walks onto the ledge once it is
        // near the ledge's height. (Round 20: it used to start the jump 1.4 to 2 m short and keep running, so the arc went up under the ledge's overhang and the head hit it.)
        let tall = rise > TALL_JUMP_RISE;
        let takeoff = if nav.idx > 0 { Some(nav.path[nav.idx - 1].pos) } else { None };
        let mut hold = false;
        if tall && state.grounded {
            let d_takeoff = takeoff.map(|t| Vec2::new(t.x - pos.x, t.z - pos.z).length()).unwrap_or(0.0);
            if d_takeoff > LINEUP {
                aim = takeoff.unwrap();
                if d_takeoff < LINEUP + 0.3 {
                    nav.just_set = false; // the walk goal clears itself on arrival: not a block
                }
            } else {
                hold = true;
                if rise > 0.15 && now - nav.last_jump > 0.7 {
                    state.jump_requested = true;
                    nav.last_jump = now;
                }
            }
        } else if tall && !state.grounded && pos.y < wp.pos.y - 1.2 {
            hold = true; // rising: no sideways drift until the ledge is near
        } else if rise > 0.15 && state.grounded && dh < 2.0 && now - nav.last_jump > 0.7 {
            state.jump_requested = true;
            nav.last_jump = now;
        }
        if !state.grounded && state.air_jumps == 0 && rise > 1.4 && state.vy < 3.0 {
            state.double_jump_requested = true;
        }
        if hold {
            state.goal = None;
            state.facing = None;
            nav.just_set = false;
            return;
        }
    }
    // stairs: a rise the capsule cannot slide over is jumped
    if wp.how == Move::Climb && wp.pos.y - pos.y > 0.15 && state.grounded && dh < 1.5 && now - nav.last_jump > 0.7 {
        state.jump_requested = true;
        nav.last_jump = now;
    }
    state.goal = Some(Goal { target: aim, run: nav.run });
    state.facing = None;
    nav.just_set = true;
}

/// Run to the ledge, jump, open the glider and steer to the landing.
pub fn glide_follow(mut nav: ResMut<Nav>, mut state: ResMut<SynthState>, player: Query<&Transform, With<PrimaryUser>>, time: Res<Time>) {
    let Some(mut g) = nav.glide else { return };
    let Ok(transform) = player.single() else { return };
    let pos = transform.translation;
    let now = time.elapsed_secs();
    let fail = |nav: &mut Nav, state: &mut SynthState, why: &str, pos: Vec3| {
        nav.glide = None;
        state.goal = None;
        state.glide = false;
        emit(json!({"event": "glide", "status": "failed", "reason": why, "position": dcl_position(pos)}));
    };
    if now - g.since > 40.0 {
        return fail(&mut nav, &mut state, "timeout", pos);
    }
    match g.phase {
        GlidePhase::Run => {
            let dh = Vec2::new(g.choice.ledge.x - pos.x, g.choice.ledge.z - pos.z).length();
            if dh < 0.8 {
                g.phase = GlidePhase::Takeoff;
                g.since = now;
                state.goal = None;
            } else if state.goal.is_none() && now - g.since > 1.0 {
                // blocked on the way: re-issue the run once
                state.goal = Some(Goal { target: Vec3::new(g.choice.ledge.x, pos.y, g.choice.ledge.z), run: true });
            } else {
                state.goal = Some(Goal { target: Vec3::new(g.choice.ledge.x, pos.y, g.choice.ledge.z), run: true });
            }
            state.facing = None;
        }
        GlidePhase::Takeoff => {
            if state.grounded {
                state.jump_requested = true;
                if now - g.since > 2.0 {
                    nav.glide = Some(g);
                    return fail(&mut nav, &mut state, "no_takeoff", pos);
                }
            } else {
                g.phase = GlidePhase::Air;
                g.airborne_since = Some(now);
            }
        }
        GlidePhase::Air => {
            if !state.glide {
                state.glide_requested = true;
            }
            let to_landing = Vec2::new(g.choice.landing.x - pos.x, g.choice.landing.z - pos.z).length();
            if to_landing < 1.5 {
                // above the landing: close the glider and drop the rest
                state.glide = false;
                state.glide_requested = false;
                state.goal = None;
            } else {
                state.goal = Some(Goal { target: Vec3::new(g.choice.landing.x, pos.y, g.choice.landing.z), run: true });
            }
            let up = g.airborne_since.map(|t| now - t).unwrap_or(0.0);
            if state.grounded && up > 0.6 {
                nav.glide = None;
                state.goal = None;
                emit(json!({"event": "glide", "status": "landed", "position": dcl_position(pos), "planned": dcl_position(g.choice.landing), "seconds": up}));
                return;
            }
        }
    }
    nav.glide = Some(g);
}

/// Work on the reachable-spot search for a few milliseconds a frame; when it finishes, emit the list (one spot per square: 3 m squares near the Synth, 6 m from 30 m of
/// walking, 10 m from 70 m, up to 900).
pub fn nav_spots_step(mut nav: ResMut<Nav>, mut scenes: Query<&mut SceneColliderData>, landscape: Option<Res<Landscape>>) {
    let Some(mut job) = nav.spots_job.take() else { return };
    let t0 = Instant::now();
    {
        let mut guards: Vec<Mut<SceneColliderData>> = scenes.iter_mut().collect();
        let mut world = EngineWorld { scenes: &mut guards, landscape: landscape.as_deref() };
        let cache = std::mem::take(&mut job.cache);
        let mut planner = Planner::with_cache(&mut world, cache);
        // about 6 ms of work a frame
        while !job.search.done && t0.elapsed().as_secs_f32() < 0.006 {
            planner.reach_step(&mut job.search, 150);
        }
        job.cache = planner.into_cache();
    }
    job.work_ms += t0.elapsed().as_secs_f32() * 1000.0;
    job.frames += 1;
    if !job.search.done {
        nav.spots_job = Some(job);
        return;
    }
    let mut best: std::collections::HashMap<(i32, i32, i32, i32), (Vec3, f32)> = std::collections::HashMap::new();
    for (p, c) in job.search.spots() {
        let (cell, band) = if c < 30.0 { (3.0, 0) } else if c < 70.0 { (6.0, 1) } else { (10.0, 2) };
        let key = ((p.x / cell).floor() as i32, (p.z / cell).floor() as i32, (p.y / 1.5).floor() as i32, band);
        let e = best.entry(key).or_insert((p, c));
        if c < e.1 {
            *e = (p, c);
        }
    }
    let mut spots: Vec<_> = best.into_values().collect();
    spots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    spots.truncate(1800); // was 900, which cut the list at about 92 m of walking: the far spots a long run needs were dropped
    let list: Vec<_> = spots
        .iter()
        .map(|(p, c)| {
            let d = dcl_position(*p);
            json!({"x": (d[0] * 10.0).round() / 10.0, "y": (d[1] * 10.0).round() / 10.0, "z": (d[2] * 10.0).round() / 10.0, "cost": (c * 10.0).round() / 10.0})
        })
        .collect();
    emit(json!({"event": "nav_spots", "count": list.len(), "from": dcl_position(job.from), "spots": list, "ms": job.started.elapsed().as_millis() as u64, "work_ms": job.work_ms.round(), "frames": job.frames, "expanded": job.search.expanded()}));
}
