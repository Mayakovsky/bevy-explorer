//! Emanate round 15: Fly for the Synth (level 3). Pure logic, no engine types beyond `Vec3`, so it is unit-tested without the game.
//!
//! 1. The thrust model: the numbers the players' halo found in ten builds (halo-build04/src/index.ts), applied each frame to the Synth's own velocity.
//!    Option A of EMANATE-SYNTH-FLY-PLAN.md: copy the numbers, then measure live and compare with the players'.
//! 2. The free-space planner: A star on a coarse 3D grid (4 m cells) that a sliced search runs a few thousand nodes a frame, like the spot list.
//!
//! Bevy space: +y up, the thrust direction is a unit "look" vector (what a player's camera would point at).

use crate::nav::{NavWorld, AVATAR_HEIGHT};
use bevy::math::Vec3;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Gravity while falling (m/s^2); the engine multiplies it by `RISE_GRAVITY_FACTOR` while moving up.
pub const GRAVITY: f32 = 10.0;
pub const RISE_GRAVITY_FACTOR: f32 = 4.0;
/// Continuous forces count this much more while the glider is open (GlideWindResponse).
pub const WIND_RESPONSE: f32 = 1.5;
/// The glider's sink while flying level, and while diving (glidingFallingSpeed).
pub const FLY_SINK: f32 = 0.3;
pub const DIVE_SINK: f32 = 20.3;
pub const FY_HOLD: f32 = 7.0;
pub const FY_RISING: f32 = 27.0;
pub const FY_EASE: f32 = 24.0;
pub const CLIMB_FORCE: f32 = 30.0;
pub const CLIMB_MAX: f32 = 14.5;
pub const DIVE_FORCE: f32 = 25.0;
pub const DIVE_MAX: f32 = 16.2;
pub const FWD_FORCE: f32 = 22.0;
/// The ceiling (the most other players can see) and the distance over which lift fades below it.
pub const CEILING: f32 = 200.0;
pub const CEILING_TAPER: f32 = 45.0;
pub const SPEED_CAP: f32 = 24.3;
pub const BRAKE: f32 = 40.0;
pub const H_MAX: f32 = 19.2;
pub const DIVE_BRAKE_START: f32 = 17.8;
pub const DIVE_BRAKE: f32 = 16.0;

/// What the model carries between frames: the velocity it integrates, and the smoothed readings the halo's limiter used.
#[derive(Clone, Copy, Debug, Default)]
pub struct FlightState {
    /// m/s
    pub vel: Vec3,
    pub vel_sm: Vec3,
    pub vy_sm: f32,
}

impl FlightState {
    pub fn start(vel: Vec3) -> Self {
        FlightState { vel, vel_sm: vel, vy_sm: vel.y }
    }
    pub fn h_speed(&self) -> f32 {
        Vec3::new(self.vel.x, 0.0, self.vel.z).length()
    }
}

fn lift_factor(y: f32) -> f32 {
    ((CEILING - y) / CEILING_TAPER).clamp(0.0, 1.0)
}

/// The thrust force (before the wind response) the halo applies for a look direction, given its smoothed readings. Returns (force, diving).
fn boost_force(look: Vec3, h_sm: f32, vy_sm: f32, y: f32) -> (Vec3, bool) {
    let sinp = look.y.clamp(-1.0, 1.0);
    let cosp = (1.0 - sinp * sinp).max(0.0).sqrt();
    let h_len = (look.x * look.x + look.z * look.z).sqrt();
    let h_ramp = ((H_MAX - h_sm) / 3.0).clamp(0.0, 1.0);
    let (fx, fz) = if h_len > 1e-4 { (look.x / h_len * FWD_FORCE * cosp * h_ramp, look.z / h_len * FWD_FORCE * cosp * h_ramp) } else { (0.0, 0.0) };
    let mut fy = FY_HOLD;
    if sinp > 0.0 {
        let w = (sinp / 0.17).clamp(0.0, 1.0);
        let base = FY_HOLD + (FY_RISING - FY_HOLD) * w + CLIMB_FORCE * sinp;
        let target = CLIMB_MAX.min(CLIMB_MAX * sinp / 0.7);
        fy = if vy_sm < target { base } else { FY_EASE };
    } else {
        fy += DIVE_FORCE * sinp * ((DIVE_MAX + vy_sm) / 3.0).clamp(0.0, 1.0);
    }
    if fy > 0.0 {
        fy *= lift_factor(y);
    }
    (Vec3::new(fx, fy, fz), sinp < -0.17)
}

/// One frame of flight. `thrust` false = the glider only (F not held): no force, the glider sinks and the horizontal speed bleeds off.
/// Returns the new velocity (also stored in the state).
pub fn step(st: &mut FlightState, look: Vec3, thrust: bool, y: f32, dt: f32) -> Vec3 {
    let dt = dt.clamp(1e-4, 0.1);
    // the halo's smoothing is per 60 Hz frame (0.85 / 0.9 of the old value); keep it frame-rate independent
    let a_v = 1.0 - 0.85f32.powf(dt * 60.0);
    let a_vel = 1.0 - 0.9f32.powf(dt * 60.0);
    let h_sm = Vec3::new(st.vel_sm.x, 0.0, st.vel_sm.z).length();
    let (mut f, diving) = if thrust { boost_force(look, h_sm, st.vy_sm, y) } else { (Vec3::ZERO, false) };
    // the limits: ONE brake force (total speed over the cap, a dive near its limit, horizontal speed over H_MAX)
    let total = st.vel_sm.length();
    let mut b = Vec3::ZERO;
    if total > SPEED_CAP {
        b -= st.vel_sm / total * BRAKE;
    }
    if total <= SPEED_CAP && st.vy_sm < -DIVE_BRAKE_START {
        b.y += ((-st.vy_sm - DIVE_BRAKE_START) / 4.1).min(1.0) * DIVE_BRAKE;
    }
    if total <= SPEED_CAP && h_sm > H_MAX {
        let k = ((h_sm - H_MAX) / 1.5).min(1.0);
        b.x += -st.vel_sm.x / h_sm * BRAKE * k;
        b.z += -st.vel_sm.z / h_sm * BRAKE * k;
    }
    f += b;
    let mut v = st.vel;
    v.x += f.x * WIND_RESPONSE * dt;
    v.z += f.z * WIND_RESPONSE * dt;
    if !thrust {
        // no thrust: the horizontal speed bleeds off (the engine's drag on external velocity)
        let k = (1.0 - 0.5 * dt).max(0.0);
        v.x *= k;
        v.z *= k;
    }
    let g = GRAVITY * if v.y > 0.0 { RISE_GRAVITY_FACTOR } else { 1.0 };
    v.y += (f.y * WIND_RESPONSE - g) * dt;
    v.y = v.y.max(-(if diving { DIVE_SINK } else { FLY_SINK }));
    st.vel = v;
    st.vy_sm = st.vy_sm * (1.0 - a_v) + v.y * a_v;
    st.vel_sm = st.vel_sm * (1.0 - a_vel) + v * a_vel;
    v
}

/// The look direction toward a target, and how thrust should be used on the way in: the Synth looks at the target; near it the thrust is cut so it
/// does not overshoot. Returns (look, thrust).
pub fn steer(pos: Vec3, vel: Vec3, target: Vec3) -> (Vec3, bool) {
    let d = target - pos;
    let dist = d.length();
    if dist < 1e-3 {
        return (Vec3::Z, false);
    }
    let look = d / dist;
    // cut the thrust when the stopping distance at the horizontal speed (a 2 s coast) is more than the distance left
    let h = Vec3::new(vel.x, 0.0, vel.z).length();
    let thrust = dist > 2.0 * h * 0.5 + 3.0;
    (look, thrust)
}

// ================= the free-space planner =================

pub const FCELL: f32 = 4.0;
/// A cell centre is open when the Synth's body fits there and the ground is at least this far below.
const CLEARANCE: f32 = 1.0;
/// the air that must be clear around a cell centre (m)
const MARGIN: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
pub struct FlyConfig {
    /// horizontal search radius in cells (25 = 100 m)
    pub radius_cells: i32,
    /// the highest cell above the start's altitude, in cells
    pub up_cells: i32,
    pub max_nodes: usize,
    /// the goal counts as reached within this distance (m)
    pub goal_tolerance: f32,
    /// the search never plans above this altitude
    pub ceiling: f32,
}

impl Default for FlyConfig {
    fn default() -> Self {
        FlyConfig { radius_cells: 25, up_cells: 25, max_nodes: 40_000, goal_tolerance: 6.0, ceiling: CEILING - 20.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FNode {
    ix: i32,
    iy: i32,
    iz: i32,
}

#[derive(Debug, PartialEq)]
pub enum FlyPlanError {
    /// no open start cell (the Synth is inside something) or the goal is not in an open cell near its position
    NoStart,
    NoGoal,
    NoPath { expanded: usize },
}

#[derive(Debug)]
pub enum FlyStatus {
    Running,
    Done(Vec<Vec3>),
    Failed(FlyPlanError),
}

struct FOpen {
    f: f32,
    n: FNode,
}
impl PartialEq for FOpen {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f
    }
}
impl Eq for FOpen {}
impl PartialOrd for FOpen {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for FOpen {
    fn cmp(&self, o: &Self) -> Ordering {
        o.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal) // min-heap
    }
}

/// A sliced A star over the free-space grid. Create it, then call `step` each frame with a node budget until it is not `Running`.
pub struct FlightSearch {
    cfg: FlyConfig,
    origin: Vec3,
    start: FNode,
    goal: FNode,
    goal_pos: Vec3,
    open: BinaryHeap<FOpen>,
    g: HashMap<FNode, f32>,
    came: HashMap<FNode, FNode>,
    open_cache: HashMap<FNode, bool>,
    expanded: usize,
    done: bool,
}

fn node_pos(origin: Vec3, n: FNode) -> Vec3 {
    Vec3::new(origin.x + n.ix as f32 * FCELL, origin.y + n.iy as f32 * FCELL, origin.z + n.iz as f32 * FCELL)
}

/// Time cost of flying between two points: horizontal at the top horizontal speed, climbing at the climb rate (the slower of the two limits the leg).
fn leg_cost(a: Vec3, b: Vec3) -> f32 {
    let dh = Vec3::new(b.x - a.x, 0.0, b.z - a.z).length();
    let dy = b.y - a.y;
    let th = dh / H_MAX;
    let tv = if dy > 0.0 { dy / CLIMB_MAX } else { -dy / DIVE_MAX };
    th.max(tv) + 0.25 * th.min(tv)
}

impl FlightSearch {
    pub fn new<W: NavWorld>(world: &mut W, from: Vec3, to: Vec3, cfg: FlyConfig) -> Result<Self, FlyPlanError> {
        let origin = from;
        let mut s = FlightSearch {
            cfg,
            origin,
            start: FNode { ix: 0, iy: 0, iz: 0 },
            goal: FNode { ix: 0, iy: 0, iz: 0 },
            goal_pos: to,
            open: BinaryHeap::new(),
            g: HashMap::new(),
            came: HashMap::new(),
            open_cache: HashMap::new(),
            expanded: 0,
            done: false,
        };
        // the start cell is where the Synth is (it may be standing on a roof: the cell is the Synth's own position, open by definition)
        s.open_cache.insert(s.start, true);
        // the goal: the cell containing it, or the nearest open one within 2 cells
        let gi = FNode { ix: ((to.x - from.x) / FCELL).round() as i32, iy: ((to.y - from.y) / FCELL).round() as i32, iz: ((to.z - from.z) / FCELL).round() as i32 };
        let mut best: Option<(f32, FNode)> = None;
        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    let n = FNode { ix: gi.ix + dx, iy: gi.iy + dy, iz: gi.iz + dz };
                    if s.is_open(world, n) {
                        let d = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                            best = Some((d, n));
                        }
                    }
                }
            }
        }
        s.goal = best.map(|(_, n)| n).ok_or(FlyPlanError::NoGoal)?;
        s.g.insert(s.start, 0.0);
        let h = leg_cost(from, node_pos(origin, s.goal));
        s.open.push(FOpen { f: h, n: s.start });
        Ok(s)
    }

    fn is_open<W: NavWorld>(&mut self, world: &mut W, n: FNode) -> bool {
        if let Some(o) = self.open_cache.get(&n) {
            return *o;
        }
        let p = node_pos(self.origin, n);
        // a margin of MARGIN m on every side: the Synth follows the route at 19 m/s and cuts corners, so a cell is open only when the air around it is clear too
        // the cells right around the start use the plain body (a takeoff from a roof edge or a narrow street must find a way up and out)
        let near_start = n.ix.abs() <= 2 && n.iy.abs() <= 2 && n.iz.abs() <= 2;
        let blocked = if near_start { world.blocked(p.x, p.z, p.y - 0.5, p.y - 0.5 + AVATAR_HEIGHT + 1.0) } else { Self::body_blocked(world, p.x, p.z, p.y) };
        let open = p.y >= 1.0 && p.y <= self.cfg.ceiling && !blocked && {
            // the ground is below the body: the highest surface at or below the cell is at least CLEARANCE under it
            let top = world.surfaces(p.x, p.z, p.y).first().copied();
            top.map(|t| p.y - t >= CLEARANCE).unwrap_or(true)
        };
        self.open_cache.insert(n, open);
        open
    }

    /// The body plus a margin: the centre and four points MARGIN m around it must all be clear.
    fn body_blocked<W: NavWorld>(world: &mut W, x: f32, z: f32, y: f32) -> bool {
        let (y0, y1) = (y - 0.5 - MARGIN * 0.5, y - 0.5 + AVATAR_HEIGHT + 1.0 + MARGIN * 0.5);
        [(0.0, 0.0), (MARGIN, 0.0), (-MARGIN, 0.0), (0.0, MARGIN), (0.0, -MARGIN)].iter().any(|(dx, dz)| world.blocked(x + dx, z + dz, y0, y1))
    }

    /// A move between two adjacent cells is free when the segment's middle is free too (a thin wall between two open cells).
    fn segment_free<W: NavWorld>(&mut self, world: &mut W, a: FNode, b: FNode) -> bool {
        let (pa, pb) = (node_pos(self.origin, a), node_pos(self.origin, b));
        let m = (pa + pb) * 0.5;
        if a.ix.abs() <= 2 && a.iy.abs() <= 2 && a.iz.abs() <= 2 {
            return !world.blocked(m.x, m.z, m.y - 0.5, m.y - 0.5 + AVATAR_HEIGHT + 1.0);
        }
        !Self::body_blocked(world, m.x, m.z, m.y)
    }

    pub fn expanded(&self) -> usize {
        self.expanded
    }

    pub fn step<W: NavWorld>(&mut self, world: &mut W, budget: usize) -> FlyStatus {
        if self.done {
            return FlyStatus::Failed(FlyPlanError::NoPath { expanded: self.expanded });
        }
        let goal_world = node_pos(self.origin, self.goal);
        for _ in 0..budget {
            let Some(FOpen { n, .. }) = self.open.pop() else {
                self.done = true;
                return FlyStatus::Failed(FlyPlanError::NoPath { expanded: self.expanded });
            };
            let gc = *self.g.get(&n).unwrap_or(&f32::MAX);
            let pos = node_pos(self.origin, n);
            if (pos - goal_world).length() <= FCELL * 0.75 || (pos - self.goal_pos).length() <= self.cfg.goal_tolerance.max(FCELL * 0.75) {
                let mut cells = vec![n];
                let mut cur = n;
                while let Some(p) = self.came.get(&cur) {
                    cells.push(*p);
                    cur = *p;
                }
                cells.reverse();
                let mut pts: Vec<Vec3> = cells.iter().map(|c| node_pos(self.origin, *c)).collect();
                pts[0] = self.origin;
                self.done = true;
                let mut path = self.smooth(world, pts);
                // finish on the goal itself (a metre above the surface under it) when the last leg is free, so a flight ends where it was asked to, not up to a cell short
                let end = self.goal_pos + Vec3::new(0.0, 1.0, 0.0);
                if (end - *path.last().unwrap()).length() > 0.5 && Self::line_free(world, *path.last().unwrap(), end) {
                    path.push(end);
                }
                return FlyStatus::Done(path);
            }
            self.expanded += 1;
            if self.expanded > self.cfg.max_nodes {
                self.done = true;
                return FlyStatus::Failed(FlyPlanError::NoPath { expanded: self.expanded });
            }
            for dx in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dz in -1..=1i32 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let m = FNode { ix: n.ix + dx, iy: n.iy + dy, iz: n.iz + dz };
                        if m.ix.abs() > self.cfg.radius_cells || m.iz.abs() > self.cfg.radius_cells || m.iy > self.cfg.up_cells || m.iy < -self.cfg.up_cells {
                            continue;
                        }
                        let pm = node_pos(self.origin, m);
                        let ng = gc + leg_cost(pos, pm);
                        if ng >= *self.g.get(&m).unwrap_or(&f32::MAX) {
                            continue;
                        }
                        if !self.is_open(world, m) || !self.segment_free(world, n, m) {
                            continue;
                        }
                        self.g.insert(m, ng);
                        self.came.insert(m, n);
                        self.open.push(FOpen { f: ng + leg_cost(pm, goal_world), n: m });
                    }
                }
            }
        }
        FlyStatus::Running
    }

    /// String pulling: drop a waypoint when the straight segment from the last kept point to the one after it is free (sampled every 2 m).
    fn smooth<W: NavWorld>(&mut self, world: &mut W, pts: Vec<Vec3>) -> Vec<Vec3> {
        if pts.len() <= 2 {
            return pts;
        }
        let mut out = vec![pts[0]];
        let mut i = 0;
        while i + 1 < pts.len() {
            let mut j = pts.len() - 1;
            while j > i + 1 && !Self::line_free(world, pts[i], pts[j]) {
                j -= 1;
            }
            out.push(pts[j]);
            i = j;
        }
        out
    }

    fn line_free<W: NavWorld>(world: &mut W, a: Vec3, b: Vec3) -> bool {
        let len = (b - a).length();
        let n = (len / 2.0).ceil().max(1.0) as i32;
        for k in 1..n {
            let p = a + (b - a) * (k as f32 / n as f32);
            if Self::body_blocked(world, p.x, p.z, p.y) {
                return false;
            }
            if let Some(t) = world.surfaces(p.x, p.z, p.y).first() {
                if p.y - t < CLEARANCE {
                    return false;
                }
            }
        }
        true
    }
}

/// Is the air straight above a spot clear (with the route margin) from `y_from` up to `y_to`? A hover target is only valid where it is: no building in the way.
pub fn column_clear<W: NavWorld>(world: &mut W, x: f32, z: f32, y_from: f32, y_to: f32) -> bool {
    let mut y = y_from + 2.0;
    while y <= y_to {
        let (y0, y1) = (y - 0.5 - MARGIN * 0.5, y - 0.5 + AVATAR_HEIGHT + 1.0 + MARGIN * 0.5);
        if [(0.0, 0.0), (MARGIN, 0.0), (-MARGIN, 0.0), (0.0, MARGIN), (0.0, -MARGIN)].iter().any(|(dx, dz)| world.blocked(x + dx, z + dz, y0, y1)) {
            return false;
        }
        y += 2.0;
    }
    true
}

#[cfg(test)]
mod tests;
