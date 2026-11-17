//! Emanate round 15: the Synth flies at power level 3 (behind the `level 3` command; nothing changes at lower levels).
//! Commands (see `SynthCommand`): `fly_dir` (a fixed look for N seconds: the measurement flight), `fly_to` (straight at a point), `fly_plan` (free-space plan
//! in slices, then follow it). Events: `fly` (status planned / failed / arrived / ended / landed) and `fly_log` every 0.5 s (position, model velocity).
//! The thrust model and the planner are `common::fly` (unit tested); this file is the engine glue: takeoff (jump, double jump, open the glider), the
//! per-frame model step written as the Synth's velocity, arrival and landing.

use std::time::Instant;

use bevy::prelude::*;
use common::{
    nav::NavWorld,
    fly::{steer, FlightSearch, FlightState, FlyConfig, FlyStatus, FLY_SINK},
    structs::PrimaryUser,
};
use scene_runner::update_world::mesh_collider::SceneColliderData;
use serde_json::json;

use super::nav::{EngineWorld, Nav};
use super::{dcl_position, emit, SynthState};
use crate::headless::landscape::Landscape;

const ARRIVE: f32 = 3.0;
const LOG_EVERY: f32 = 0.5;

#[derive(Clone, Debug)]
pub enum FlyRequest {
    /// fixed look: yaw (degrees, the `face` convention), pitch (degrees), seconds
    Dir { yaw: f32, pitch: f32, secs: f32 },
    To { target: Vec3 },
    /// `hover`: the target is a hover over a standable spot: its column of air up to the target must be clear
    Plan { target: Vec3, hover: bool },
}

enum Mode {
    Idle,
    Dir { look: Vec3, until: f32 },
    Path,
}

struct Job {
    search: FlightSearch,
    started: Instant,
    work_ms: f32,
    target: Vec3,
}

#[derive(Resource)]
pub struct Fly {
    pub request: Option<FlyRequest>,
    /// the velocity the model wants this frame; `drive` writes it and clears it
    pub velocity: Option<Vec3>,
    mode: Mode,
    job: Option<Job>,
    path: Vec<Vec3>,
    idx: usize,
    st: FlightState,
    flown: f32,
    log_at: f32,
    was_airborne: bool,
    takeoff: bool,
}

impl Default for Fly {
    fn default() -> Self {
        Fly { request: None, velocity: None, mode: Mode::Idle, job: None, path: vec![], idx: 0, st: FlightState::default(), flown: 0.0, log_at: 0.0, was_airborne: false, takeoff: false }
    }
}

fn fail(reason: &str, extra: serde_json::Value) {
    emit(json!({"event": "fly", "status": "failed", "reason": reason, "info": extra}));
}

impl Fly {
    /// `stop`: end any flight and plan (the glider stays open and sinks slowly until the Synth lands)
    pub fn stop(&mut self) {
        self.mode = Mode::Idle;
        self.job = None;
        self.request = None;
        self.velocity = None;
    }
}

pub fn fly_drive(
    time: Res<Time>,
    mut fly: ResMut<Fly>,
    mut state: ResMut<SynthState>,
    nav: Res<Nav>,
    mut scenes: Query<&mut SceneColliderData>,
    landscape: Option<Res<Landscape>>,
    player: Query<&Transform, With<PrimaryUser>>,
) {
    let Ok(t) = player.single() else { return };
    let pos = t.translation;
    let dt = time.delta_secs().clamp(1e-4, 0.1);
    let now = time.elapsed_secs();
    fly.velocity = None;

    if let Some(req) = fly.request.take() {
        if nav.level < 3 {
            fail("level", json!({"level": nav.level}));
        } else {
            fly.job = None;
            fly.takeoff = false;
            match req {
                FlyRequest::Dir { yaw, pitch, secs } => {
                    let (y, p) = (yaw.to_radians(), pitch.to_radians());
                    let look = Vec3::new(-y.sin() * p.cos(), p.sin(), -y.cos() * p.cos());
                    fly.mode = Mode::Dir { look, until: now + secs };
                    fly.flown = 0.0;
                    emit(json!({"event": "fly", "status": "started", "kind": "dir", "pitch": pitch, "yaw": yaw, "secs": secs}));
                }
                FlyRequest::To { target } => {
                    fly.path = vec![target];
                    fly.idx = 0;
                    fly.mode = Mode::Path;
                    emit(json!({"event": "fly", "status": "planned", "kind": "direct", "waypoints": 1, "goal": dcl_position(target)}));
                }
                FlyRequest::Plan { target, hover } => {
                    fly.mode = Mode::Idle;
                    let mut guards: Vec<Mut<SceneColliderData>> = scenes.iter_mut().collect();
                    let mut world = EngineWorld::new(&mut guards, landscape.as_deref());
                    if hover {
                        let ground = world.surfaces(target.x, target.z, target.y).first().copied().unwrap_or(0.0);
                        if !common::fly::column_clear(&mut world, target.x, target.z, ground, target.y) {
                            fail("column_blocked", json!({"goal": dcl_position(target)}));
                            return;
                        }
                    }
                    match FlightSearch::new(&mut world, pos, target, FlyConfig::default()) {
                        Ok(search) => fly.job = Some(Job { search, started: Instant::now(), work_ms: 0.0, target }),
                        Err(e) => fail(&format!("{e:?}"), json!({"goal": dcl_position(target)})),
                    }
                }
            }
        }
    }

    // the planner, a few milliseconds a frame
    if let Some(mut job) = fly.job.take() {
        let t0 = Instant::now();
        let mut result = FlyStatus::Running;
        {
            let mut guards: Vec<Mut<SceneColliderData>> = scenes.iter_mut().collect();
            let mut world = EngineWorld::new(&mut guards, landscape.as_deref());
            while matches!(result, FlyStatus::Running) && t0.elapsed().as_secs_f32() < 0.006 {
                result = job.search.step(&mut world, 200);
            }
        }
        job.work_ms += t0.elapsed().as_secs_f32() * 1000.0;
        match result {
            FlyStatus::Running => fly.job = Some(job),
            FlyStatus::Done(path) => {
                emit(json!({"event": "fly", "status": "planned", "kind": "plan", "waypoints": path.len(), "expanded": job.search.expanded(), "ms": job.started.elapsed().as_millis() as u64, "work_ms": job.work_ms.round(), "goal": dcl_position(job.target), "path": path.iter().take(40).map(|p| dcl_position(*p)).collect::<Vec<_>>()}));
                fly.path = path;
                fly.idx = 1.min(fly.path.len().saturating_sub(1));
                fly.mode = Mode::Path;
            }
            FlyStatus::Failed(e) => fail(&format!("{e:?}"), json!({"goal": dcl_position(job.target), "ms": job.started.elapsed().as_millis() as u64})),
        }
    }

    if matches!(fly.mode, Mode::Idle) {
        fly.was_airborne = false;
        return;
    }

    // takeoff: jump, double jump, open the glider (the same sequence a player's flight starts with); the model takes over once the glider is open
    if state.grounded {
        if fly.was_airborne {
            emit(json!({"event": "fly", "status": "landed", "position": dcl_position(pos), "flown_secs": fly.flown}));
            fly.mode = Mode::Idle;
            fly.was_airborne = false;
            return;
        }
        state.jump_requested = true;
        fly.takeoff = true;
        return;
    }
    fly.was_airborne = true;
    if !state.glide {
        if state.air_jumps == 0 {
            state.double_jump_requested = true;
        } else {
            state.glide_requested = true;
        }
        return;
    }
    if fly.takeoff {
        fly.takeoff = false;
        fly.st = FlightState::start(Vec3::new(0.0, state.vy.max(-FLY_SINK), 0.0));
        emit(json!({"event": "fly", "status": "airborne", "position": dcl_position(pos)}));
    }

    let (look, thrust) = match fly.mode {
        Mode::Dir { look, until } => {
            if now >= until {
                emit(json!({"event": "fly", "status": "ended", "position": dcl_position(pos), "flown_secs": fly.flown}));
                fly.mode = Mode::Idle;
                return;
            }
            (look, true)
        }
        Mode::Path => {
            let target = fly.path[fly.idx];
            let near = if fly.idx + 1 == fly.path.len() { ARRIVE } else { 6.0 };
            if (target - pos).length() < near {
                if fly.idx + 1 >= fly.path.len() {
                    emit(json!({"event": "fly", "status": "arrived", "position": dcl_position(pos), "flown_secs": fly.flown}));
                    fly.mode = Mode::Idle;
                    return;
                }
                fly.idx += 1;
            }
            let target = fly.path[fly.idx];
            if fly.idx + 1 == fly.path.len() {
                steer(pos, fly.st.vel, target)
            } else {
                (steer(pos, fly.st.vel, target).0, true) // no braking at a waypoint that is not the last
            }
        }
        Mode::Idle => return,
    };
    let v = common::fly::step(&mut fly.st, look, thrust, pos.y, dt);
    fly.flown += dt;
    fly.velocity = Some(v);
    if now - fly.log_at >= LOG_EVERY {
        fly.log_at = now;
        emit(json!({"event": "fly_log", "t": (fly.flown * 100.0).round() / 100.0, "position": dcl_position(pos), "h": (fly.st.h_speed() * 100.0).round() / 100.0, "vy": (v.y * 100.0).round() / 100.0, "total": (v.length() * 100.0).round() / 100.0, "thrust": thrust}));
    }
}
