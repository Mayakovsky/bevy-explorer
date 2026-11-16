//! Emanate: the Synth control channel for the headless synthetic client.
//!
//! A line-based JSON interface on stdin/stdout that an orchestrator drives:
//!
//! * stdin, one JSON command per line: `{"cmd":"walk_to","x":..,"y":..,"z":..}` (see
//!   [`SynthCommand`]).
//! * stdout, one event per line, prefixed `@synth ` so it can be told apart from the engine's
//!   own log lines: `@synth {"event":"position",...}`.
//!
//! Coordinates in commands and events are Decentraland world space (x right, y up, z forward;
//! parcel = floor(x/16), floor(z/16)), the same as `agent_commands.rs`.
//!
//! Movement goes through the engine's real avatar physics (`user_input::AvatarMovementPlugin`):
//! this module only decides a *velocity* each frame (steering toward a target, gravity, jump,
//! glide) and writes it into the player's `AvatarMovement`; collision, ground following and the
//! Pulse position broadcast are the engine's own. The channel is stdin (not a socket) because the
//! orchestrator spawns this process and already owns its pipes.

mod nav;
use nav::{nav_follow, nav_plan, Nav, NavRequest};
use std::sync::{mpsc, Mutex};

use bevy::prelude::*;
use common::{
    rpc::{RpcCall, RpcCallEvent, RpcResultSender},
    sets::SceneSets,
    structs::{
        AppConfig, AvatarDynamicState, EmoteCommand, EmoteMask, EngineMovementControl,
        PrimaryPlayerRes, PrimaryUser, SynthBroadcastOverride,
    },
};
use comms::{
    broadcast, broadcast_to,
    global_crdt::{chat_wire_timestamp, ChatEvent, ForeignPlayer},
    profile::{CurrentUserProfile, UserProfile},
    BroadcastTarget, Transport,
};
use ipfs::IpfsAssetServer;
use dcl_component::{
    proto_components::kernel::comms::rfc4, transform_and_parent::DclTranslation,
};
use serde::Deserialize;
use serde_json::json;
use user_input::avatar_movement::{ActivePlayerComponent, AvatarMovement};
use wallet::Wallet;

/// Jump takeoff speed. `apply_movement` flags a jump for animation when `velocity.y > 10`, and the
/// default jump height is `PrimaryUser::jump_height` (1.9 m): g = 30 gives v0 = sqrt(2*30*1.9) =
/// 10.7 m/s. Tunable once seen from a second client.
const GRAVITY: f32 = 30.0;
/// Fastest fall.
const TERMINAL_FALL: f32 = 50.0;
/// Fall speed while gliding.
const GLIDE_FALL: f32 = 3.0;
/// Horizontal speed while gliding. With no walk target the glide carries forward along the current
/// facing, as a real DCL glide does.
const GLIDE_SPEED: f32 = 8.0;
/// How long an emote broadcast is declared to run for (ms); `wave` is about 3 s.
const EMOTE_DURATION_MS: u32 = 3000;
/// In-air second jump height (m), a little under DCL's double-jump.
const DOUBLE_JUMP_HEIGHT: f32 = 1.5;
/// Within this horizontal distance of a walk target counts as arrived.
const ARRIVE_RADIUS: f32 = 0.5;
/// This long below 20% of the requested speed counts as blocked.
const BLOCKED_AFTER_SECS: f32 = 0.75;
const POSITION_EVENT_SECS: f32 = 0.1;
const POSITION_KEEPALIVE_SECS: f32 = 2.0;

/// Commands accepted on stdin.
#[derive(Deserialize, Debug)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum SynthCommand {
    /// Walk (walk speed) to a world position through real movement and collisions.
    WalkTo { x: f32, y: Option<f32>, z: f32 },
    /// Same at run speed.
    RunTo { x: f32, y: Option<f32>, z: f32 },
    /// Emanate: walk (or `run`) to a world position along a planned route around buildings, through doorways, up stairs and ramps (`y` picks a level:
    /// a floor or a roof). Events: `nav` planned / failed / arrived / replanning / stuck.
    NavTo {
        x: f32,
        y: Option<f32>,
        z: f32,
        #[serde(default)]
        run: bool,
    },
    /// Emanate: list the spots reachable within `max_cost` metres of walking (`nav_spots` event), for choosing where to hide.
    NavSpots {
        #[serde(default = "default_spot_cost")]
        max_cost: f32,
    },
    /// Abandon the current walk/run target.
    Stop,
    /// Ground: takeoff. In the air: DCL's sequence, first press = double jump, second = open the
    /// glider (so `jump`, `jump`, `jump` = takeoff, double jump, glide).
    Jump,
    /// In-air second jump (explicit).
    DoubleJump,
    /// Start (`on` true, the default) or stop gliding.
    Glide {
        #[serde(default = "yes")]
        on: bool,
    },
    /// Teleport to a parcel (`[x, y]`) in the current realm.
    Teleport { parcel: [i32; 2] },
    /// Instant move to a world position (no travel).
    Move { x: f32, y: f32, z: f32 },
    /// Face a direction; yaw in degrees (0 = looking toward +z in Decentraland space).
    Look {
        yaw: f32,
        /// accepted for the interface; head pitch is not driven yet
        #[allow(dead_code)]
        #[serde(default)]
        pitch: f32,
    },
    /// Play an emote by id or urn (e.g. `wave`).
    Emote { id: String },
    /// Say something in nearby chat.
    Chat { text: String },
    /// Republish the profile with a new display name.
    SayProfile { name: String },
    /// Emanate: set the worn items (base-avatar wearable URNs, free only) and republish the profile.
    Wear { items: Vec<String> },
    /// Start (`on` true, the default) or stop streaming `track` events: every 0.2 s, each visible
    /// player's position and finite-difference velocity as this client receives it over Pulse.
    Track {
        #[serde(default = "yes")]
        on: bool,
    },
    Status,
    /// List the other players this client can see (address, name, position).
    Players,
}

fn yes() -> bool {
    true
}

fn default_spot_cost() -> f32 {
    120.0
}

#[derive(Resource)]
pub struct SynthChannel(Mutex<mpsc::Receiver<SynthCommand>>);

pub fn emit(event: serde_json::Value) {
    println!("@synth {event}");
}

#[derive(Default)]
struct Goal {
    /// bevy-space target
    target: Vec3,
    run: bool,
}

#[derive(Resource, Default)]
struct SynthState {
    goal: Option<Goal>,
    jump_requested: bool,
    double_jump_requested: bool,
    glide_requested: bool,
    glide: bool,
    /// air jumps used since leaving the ground (0 or 1)
    air_jumps: u8,
    emote_count: u32,
    tracking: bool,
    grounded: bool,
    vy: f32,
    /// degrees; applied when there is no walk target
    facing: Option<f32>,
    /// the last orientation written; the engine resets the active movement to 0 each frame, so
    /// it must be written every frame or an idle avatar snaps to face 0
    last_yaw: f32,
    blocked_secs: f32,
    last_pos: Option<Vec3>,
    position_timer: f32,
    last_reported: Option<Vec3>,
}

pub struct SynthPlugin;

impl Plugin for SynthPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SynthChannel(Mutex::new(spawn_stdin_reader())))
            .init_resource::<SynthState>()
            .init_resource::<SynthBroadcastOverride>()
            .init_resource::<Nav>()
            .add_systems(Startup, announce)
            .add_systems(
                Update,
                (release_profile_deploy, drain_commands, nav_plan, nav_follow, drive, emit_chat, track_players)
                    .chain()
                    // after the engine picked the active AvatarMovement for this frame, before the
                    // PostUpdate movement pipeline applies it
                    .after(SceneSets::PostLoop),
            );
    }
}

fn announce() {
    emit(json!({"event": "ready"}));
}

fn spawn_stdin_reader() -> mpsc::Receiver<SynthCommand> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<SynthCommand>(line) {
                Ok(command) => {
                    if tx.send(command).is_err() {
                        break;
                    }
                }
                Err(e) => emit(json!({"event": "error", "error": format!("bad command: {e}")})),
            }
        }
        // the orchestrator went away: exit like the scene server does
        emit(json!({"event": "stdin_closed"}));
        std::process::exit(0);
    });
    rx
}

fn to_bevy(x: f32, y: f32, z: f32) -> Vec3 {
    DclTranslation([x, y, z]).to_bevy_translation()
}

fn dcl_position(p: Vec3) -> [f32; 3] {
    [p.x, p.y, -p.z]
}

fn parcel_of(p: Vec3) -> [i32; 2] {
    [(p.x / 16.0).floor() as i32, ((-p.z) / 16.0).floor() as i32]
}

#[allow(clippy::too_many_arguments)]
fn drain_commands(
    channel: Res<SynthChannel>,
    mut state: ResMut<SynthState>,
    mut commands: Commands,
    mut rpc: EventWriter<RpcCallEvent>,
    player: Query<(Entity, &Transform, Option<&EmoteCommand>), With<PrimaryUser>>,
    mut profile: ResMut<CurrentUserProfile>,
    transports: Query<&Transport>,
    wallet: Res<Wallet>,
    time: Res<Time>,
    mut chat: EventWriter<ChatEvent>,
    player_res: Res<PrimaryPlayerRes>,
    foreign: Query<(&ForeignPlayer, &Transform, Option<&UserProfile>)>,
    mut nav: ResMut<Nav>,
) {
    let Ok(rx) = channel.0.lock() else { return };
    while let Ok(command) = rx.try_recv() {
        let Ok((player_entity, transform, prev_emote)) = player.single() else {
            emit(json!({"event": "error", "error": "player not ready"}));
            continue;
        };
        match command {
            SynthCommand::NavTo { x, y, z, run } => {
                nav.replans = 0;
                nav.pending = Some(NavRequest::To { target: to_bevy(x, y.unwrap_or(0.0), z), level: y.is_some(), run });
            }
            SynthCommand::NavSpots { max_cost } => {
                nav.pending = Some(NavRequest::Spots { max_cost });
            }
            SynthCommand::WalkTo { x, y, z } => {
                nav.active = false;
                state.goal = Some(Goal {
                    target: to_bevy(x, y.unwrap_or(0.0), z),
                    run: false,
                });
                state.facing = None;
                state.blocked_secs = 0.0;
            }
            SynthCommand::RunTo { x, y, z } => {
                nav.active = false;
                state.goal = Some(Goal {
                    target: to_bevy(x, y.unwrap_or(0.0), z),
                    run: true,
                });
                state.facing = None;
                state.blocked_secs = 0.0;
            }
            SynthCommand::Stop => {
                nav.active = false;
                state.goal = None;
            }
            SynthCommand::Jump => {
                if state.grounded {
                    state.jump_requested = true;
                } else if state.air_jumps == 0 {
                    state.double_jump_requested = true;
                } else {
                    state.glide_requested = true;
                }
            }
            SynthCommand::DoubleJump => state.double_jump_requested = true,
            SynthCommand::Glide { on } => {
                if on {
                    state.glide_requested = true;
                } else {
                    state.glide = false;
                    state.glide_requested = false;
                }
            }
            SynthCommand::Teleport { parcel } => {
                rpc.write(RpcCallEvent::engine(RpcCall::TeleportPlayer {
                    to: Some(IVec2::new(parcel[0], parcel[1])),
                    realm: None,
                    response: RpcResultSender::default(),
                }));
                state.goal = None;
                emit(json!({"event": "teleporting", "parcel": parcel}));
            }
            SynthCommand::Move { x, y, z } => {
                rpc.write(RpcCallEvent::engine(RpcCall::MovePlayer {
                    to: to_bevy(x, y, z),
                    looking_at: None,
                    duration: None,
                    camera_rotation: None,
                    response: None,
                }));
                state.goal = None;
            }
            SynthCommand::Look { yaw, pitch: _ } => {
                state.facing = Some(yaw);
                state.goal = None;
            }
            SynthCommand::Emote { id } => {
                // The headless app has no avatar animation systems, so nothing turns the local
                // EmoteCommand below into a Pulse `EmoteStart`; send it directly, as the full
                // client's `broadcast_emote` does (one-shot, with a duration the observers and the
                // Pulse server use to end it). Base emotes go out by plain name (e.g. `wave`).
                state.emote_count += 1;
                broadcast(
                    transports.iter(),
                    BroadcastTarget::PULSE,
                    false,
                    comms::Emote {
                        urn: id.clone(),
                        incremental_id: state.emote_count,
                        timestamp: time.elapsed_secs_f64(),
                        duration_ms: Some(EMOTE_DURATION_MS),
                        stopping: false,
                        mask: EmoteMask::FullBody,
                    },
                );
                emit(json!({"event": "emote", "id": id}));
                commands.entity(player_entity).insert(EmoteCommand {
                    urn: id,
                    timestamp: prev_emote.map(|prev| prev.timestamp + 1).unwrap_or_default(),
                    r#loop: false,
                    mask: EmoteMask::FullBody,
                });
            }
            SynthCommand::Chat { text } => {
                // the same two steps the full client takes: a local ChatEvent (so our own history
                // and scenes see it) and the wire packet to the byte transports that carry nearby
                // chat
                let timestamp = chat_wire_timestamp();
                chat.write(ChatEvent {
                    timestamp: time.elapsed_secs_f64(),
                    sender: player_res.0,
                    channel: "Nearby".to_owned(),
                    message: text.clone(),
                });
                broadcast_to(
                    transports.iter(),
                    BroadcastTarget::WEBSOCKET | BroadcastTarget::LIVEKIT,
                    false,
                    &rfc4::Packet {
                        message: Some(rfc4::packet::Message::Chat(rfc4::Chat {
                            message: text,
                            timestamp,
                        })),
                        protocol_version: 100,
                    },
                );
            }
            SynthCommand::SayProfile { name } => {
                if let Some(p) = profile.profile.as_mut() {
                    p.version += 1;
                    p.content.version = p.version as i64;
                    p.content.name = name.clone();
                    profile.is_deployed = false;
                    emit(json!({"event": "profile", "name": name}));
                } else {
                    emit(json!({"event": "error", "error": "no profile yet"}));
                }
            }
            SynthCommand::Wear { items } => {
                if let Some(p) = profile.profile.as_mut() {
                    p.version += 1;
                    p.content.version = p.version as i64;
                    p.content.avatar.wearables = items.clone();
                    let v = p.version;
                    profile.is_deployed = false;
                    emit(json!({"event": "wear", "items": items, "profile_version": v}));
                } else {
                    emit(json!({"event": "error", "error": "no profile yet"}));
                }
            }
            SynthCommand::Players => {
                let list: Vec<_> = foreign
                    .iter()
                    .map(|(fp, tf, prof)| {
                        json!({
                            "address": format!("{:#x}", fp.address),
                            "name": prof.map(|p| p.content.name.clone()),
                            "position": dcl_position(tf.translation),
                            "parcel": parcel_of(tf.translation),
                            "connected": !fp.transports.is_empty(),
                        })
                    })
                    .collect();
                emit(json!({"event": "players", "players": list}));
            }
            SynthCommand::Track { on } => {
                state.tracking = on;
                emit(json!({"event": "tracking", "on": on}));
            }
            SynthCommand::Status => {
                let p = transform.translation;
                emit(json!({
                    "event": "status",
                    "address": wallet.address().map(|a| format!("{a:#x}")),
                    "name": profile.profile.as_ref().map(|p| p.content.name.clone()),
                    "profile_version": profile.profile.as_ref().map(|p| p.version),
                    "position": dcl_position(p),
                    "parcel": parcel_of(p),
                    "walking": state.goal.is_some(),
                    "gliding": state.glide,
                }));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn drive(
    time: Res<Time>,
    mut state: ResMut<SynthState>,
    mut player: Query<
        (
            &Transform,
            &AvatarDynamicState,
            &mut ActivePlayerComponent<AvatarMovement>,
        ),
        With<PrimaryUser>,
    >,
    mut movement_control: ResMut<EngineMovementControl>,
    mut broadcast: ResMut<SynthBroadcastOverride>,
    config: Res<AppConfig>,
) {
    let Ok((transform, dynamic, mut active)) = player.single_mut() else {
        return;
    };
    let dt = time.delta_secs().max(1e-4);
    let pos = transform.translation;

    // The movement pipeline ignores a movement "initiated" at or before this time, and a default
    // active component reads as initiated at 0.0: with the default 0.0 here that is always
    // "stale". We are the only driver, so accept everything.
    movement_control.accept_movement_after = -1.0;

    // ---- horizontal ----
    let mut horizontal = Vec3::ZERO;
    let mut yaw_deg: Option<f32> = state.facing;
    let mut requested_speed = 0.0;
    let mut arrived = false;
    if let Some(goal) = state.goal.as_ref() {
        let delta = Vec3::new(goal.target.x - pos.x, 0.0, goal.target.z - pos.z);
        let dist = delta.length();
        if dist < ARRIVE_RADIUS {
            arrived = true;
        } else {
            let dir = delta / dist;
            let mut speed = if goal.run {
                config.player_settings.run_speed
            } else {
                config.player_settings.walk_speed
            };
            if state.glide {
                speed = GLIDE_SPEED;
            }
            // don't overshoot in the last frame
            speed = speed.min(dist / dt);
            requested_speed = speed;
            horizontal = dir * speed;
            yaw_deg = Some((-dir.x).atan2(-dir.z).to_degrees());
        }
    }
    if state.glide && state.goal.is_none() {
        // carry forward along the facing (yaw convention: dir = (-sin yaw, 0, -cos yaw))
        let yaw = state.facing.unwrap_or(state.last_yaw).to_radians();
        horizontal = Vec3::new(-yaw.sin(), 0.0, -yaw.cos()) * GLIDE_SPEED;
        requested_speed = 0.0;
    }
    if arrived {
        state.goal = None;
        emit(json!({
            "event": "arrived",
            "position": dcl_position(pos),
            "parcel": parcel_of(pos),
        }));
    }

    // ---- blocked detection (collision stopped us) ----
    if state.goal.is_some() && requested_speed > 0.0 {
        let moved = state
            .last_pos
            .map(|last| Vec3::new(pos.x - last.x, 0.0, pos.z - last.z).length() / dt)
            .unwrap_or(requested_speed);
        if moved < 0.2 * requested_speed {
            state.blocked_secs += dt;
        } else {
            state.blocked_secs = 0.0;
        }
        if state.blocked_secs > BLOCKED_AFTER_SECS {
            state.goal = None;
            state.blocked_secs = 0.0;
            horizontal = Vec3::ZERO;
            emit(json!({
                "event": "blocked",
                "position": dcl_position(pos),
                "parcel": parcel_of(pos),
            }));
        }
    } else {
        state.blocked_secs = 0.0;
    }
    state.last_pos = Some(pos);

    // ---- vertical ----
    // `ground_height` is the height above the nearest ground below (the player's own height
    // when no collider is loaded; the engine floors y at 0)
    let grounded = dynamic.ground_height <= 0.1 || pos.y <= 0.01;
    state.grounded = grounded;
    if grounded {
        state.glide = false; // landing ends a glide
        state.air_jumps = 0;
        state.double_jump_requested = false;
        if state.glide_requested {
            state.glide_requested = false;
            emit(json!({"event": "error", "error": "glide needs to be airborne: jump, then jump again (double jump), then glide"}));
        }
        if state.jump_requested {
            state.vy = (2.0 * GRAVITY * config.player_settings.jump_height).sqrt();
        } else if horizontal != Vec3::ZERO {
            state.vy = -3.0; // stay glued to slopes while walking
        } else {
            state.vy = 0.0;
        }
    } else {
        state.vy -= GRAVITY * dt;
        if state.double_jump_requested {
            state.double_jump_requested = false;
            if state.air_jumps == 0 && !state.glide {
                state.air_jumps = 1;
                state.vy = (2.0 * GRAVITY * DOUBLE_JUMP_HEIGHT).sqrt();
            }
        }
        if state.glide_requested {
            state.glide_requested = false;
            state.glide = true;
            state.air_jumps = state.air_jumps.max(1);
        }
        if state.glide {
            state.vy = state.vy.max(-GLIDE_FALL);
        }
        state.vy = state.vy.max(-TERMINAL_FALL);
    }
    state.jump_requested = false;
    // state the double-jump / glide to other clients (rfc4::Movement jump_count / glide_state)
    broadcast.jump_count = if grounded { 0 } else { 1 + state.air_jumps as i32 };
    broadcast.gliding = state.glide;

    active.component.velocity = Vec3::new(horizontal.x, state.vy, horizontal.z);
    if let Some(yaw) = yaw_deg {
        state.last_yaw = yaw;
    }
    active.component.orientation = state.last_yaw;

    // ---- position events ----
    state.position_timer += dt;
    let moved_since_report = state
        .last_reported
        .map(|last| last.distance(pos) > 0.05)
        .unwrap_or(true);
    let due = if moved_since_report {
        state.position_timer >= POSITION_EVENT_SECS
    } else {
        state.position_timer >= POSITION_KEEPALIVE_SECS
    };
    if due {
        state.position_timer = 0.0;
        state.last_reported = Some(pos);
        let v = dynamic.velocity;
        emit(json!({
            "event": "position",
            "position": dcl_position(pos),
            "velocity": [v.x, v.y, -v.z],
            "yaw": active.component.orientation,
            "grounded": grounded,
            "gliding": state.glide,
            "parcel": parcel_of(pos),
        }));
    }
}

/// While tracking is on, report each visible player's received position and its finite-difference
/// velocity every `TRACK_SECS`. This is what other clients' movement looks like over Pulse: the
/// engine interpolates/extrapolates the foreign transform, so jitter and the server's altitude
/// (200 m) and speed (50 m/s) caps show up here.
const TRACK_SECS: f32 = 0.2;

fn track_players(
    time: Res<Time>,
    state: Res<SynthState>,
    foreign: Query<(&ForeignPlayer, &Transform, Option<&UserProfile>)>,
    mut timer: Local<f32>,
    mut last: Local<std::collections::HashMap<String, (f32, Vec3)>>,
) {
    if !state.tracking {
        last.clear();
        return;
    }
    *timer += time.delta_secs();
    if *timer < TRACK_SECS {
        return;
    }
    *timer = 0.0;
    let now = time.elapsed_secs();
    for (fp, tf, prof) in foreign.iter() {
        let key = format!("{:#x}", fp.address);
        let pos = tf.translation;
        let velocity = last.get(&key).map(|(t, p)| {
            let dt = (now - t).max(1e-3);
            let v = (pos - *p) / dt;
            [v.x, v.y, -v.z]
        });
        last.insert(key.clone(), (now, pos));
        emit(json!({
            "event": "track",
            "t": now,
            "address": key,
            "name": prof.map(|p| p.content.name.clone()),
            "position": dcl_position(pos),
            "velocity": velocity,
            "parcel": parcel_of(pos),
        }));
    }
}

/// The startup profile is created held (`app::setup`); deploy it once the realm's entities
/// endpoint is known, then report the outcome.
fn release_profile_deploy(
    mut profile: ResMut<CurrentUserProfile>,
    ipfas: IpfsAssetServer,
    mut deployed: EventReader<comms::profile::ProfileDeployedEvent>,
) {
    if profile.deploy_held && ipfas.ipfs().entities_endpoint().is_some() {
        profile.deploy_held = false;
        profile.is_deployed = false;
    }
    for ev in deployed.read() {
        emit(json!({
            "event": "profile_deployed",
            "version": ev.version,
            "ok": ev.success,
        }));
    }
}

fn emit_chat(mut events: EventReader<ChatEvent>, players: Query<&ForeignPlayer>) {
    for ev in events.read() {
        if ev.channel != "Nearby" {
            continue;
        }
        // only other players (the sender of our own messages is the primary user, not foreign)
        let Ok(from) = players.get(ev.sender) else {
            continue;
        };
        emit(json!({
            "event": "chat",
            "from": format!("{:#x}", from.address),
            "channel": ev.channel,
            "text": ev.message,
        }));
    }
}
