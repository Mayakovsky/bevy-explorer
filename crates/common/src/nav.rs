//! Emanate: path planning for the Synth over whatever colliders the engine has loaded (scenes plus the landscape).
//!
//! The planner is pure: it asks a [`NavWorld`] two questions (which upward-facing surfaces are under this point, and does the avatar capsule fit
//! here) and runs A* over lazily generated nodes `(cell x, cell z, surface height)`, so one cell can hold several levels (ground, a floor above
//! it, a roof). Moves are the ones a normal player can make: walk, step up at most [`MAX_STEP`], follow a ramp or stairs (a continuous surface
//! rising at most [`MAX_RAMP_RISE`] per metre), and step or drop off a ledge of at most [`MAX_DROP`] (with a cost). Doorways, stairs and
//! roofs are not special cases: a doorway is a gap the capsule fits through, stairs and ramps are surfaces that connect levels.
//! Gliding down from a height is a separate query ([`glide_to_goal`], [`glide_options`]): the Synth runs off a ledge, opens the glider and flies straight to a lower goal.

use bevy::math::Vec3;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

pub const CELL: f32 = 1.0;
pub const AVATAR_RADIUS: f32 = 0.3;
pub const AVATAR_HEIGHT: f32 = 2.0;
/// A step the avatar climbs without a ramp (the engine's `MAX_STEP_HEIGHT`).
pub const MAX_STEP: f32 = 0.5;
/// The steepest continuous slope it walks, as rise per horizontal metre (about 50 degrees; the engine's limit is 60).
pub const MAX_RAMP_RISE: f32 = 1.2;
/// A drop it will take on purpose without gliding.
pub const MAX_DROP: f32 = 3.0;
/// How far the ramp surface may deviate from a straight line between two cell centres (stairs are lumpy).
const RAMP_TOLERANCE: f32 = 0.25;

pub trait NavWorld {
    /// Heights of walkable (upward-facing, not too steep) surfaces under `(x, z)` at or below `from_y`, highest first.
    fn surfaces(&mut self, x: f32, z: f32, from_y: f32) -> Vec<f32>;
    /// Does a vertical cylinder of the avatar's radius at `(x, z)` spanning heights `y0..y1` overlap anything solid? (Callers start the span
    /// [`MAX_STEP`] above the feet, so a riser or a kerb the avatar steps over is not an obstacle.)
    fn blocked(&mut self, x: f32, z: f32, y0: f32, y1: f32) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Node {
    ix: i32,
    iz: i32,
    /// surface height in centimetres
    y: i32,
}

impl Node {
    fn pos(&self) -> Vec3 {
        Vec3::new((self.ix as f32 + 0.5) * CELL, self.y as f32 * 0.01, (self.iz as f32 + 0.5) * CELL)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Move {
    Walk,
    /// a step up or a ramp or stairs
    Climb,
    /// a step or drop down off a ledge
    Drop,
    /// a jump (or a double jump) up onto an adjacent ledge or platform: only planned when the config's `jump_rise` allows it
    Jump,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Waypoint {
    pub pos: Vec3,
    pub how: Move,
}

#[derive(Clone, Debug)]
pub struct PlanConfig {
    /// search is limited to this many cells from the start (a square)
    pub radius_cells: i32,
    pub max_nodes: usize,
    /// reaching within this many metres horizontally of the goal counts
    pub goal_tolerance: f32,
    /// extra cost per metre climbed, per metre dropped (a drop is also a fixed cost)
    pub climb_cost: f32,
    pub drop_cost: f32,
    /// the highest ledge a jump gets up onto (the Synth's power level: normal 1.5, Jump level 8.0 with the double jump); 0 = walking only
    pub jump_rise: f32,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self { radius_cells: 150, max_nodes: 60_000, goal_tolerance: 1.2, climb_cost: 0.6, drop_cost: 1.5, jump_rise: 0.0 }
    }
}

#[derive(Debug, PartialEq)]
pub enum PlanError {
    /// no standable surface at the start
    NoStart,
    /// no standable surface at the goal
    NoGoal,
    /// the search ran out of nodes or space without reaching the goal
    NoPath { expanded: usize },
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub waypoints: Vec<Waypoint>,
    pub cost: f32,
    pub expanded: usize,
}

struct Open {
    f: f32,
    node: Node,
}
impl PartialEq for Open {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f
    }
}
impl Eq for Open {}
impl PartialOrd for Open {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Open {
    fn cmp(&self, o: &Self) -> Ordering {
        o.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal) // min-heap
    }
}

fn cell_of(v: f32) -> i32 {
    (v / CELL).floor() as i32
}

/// The planner keeps a per-cell cache of surfaces; build a fresh one (or call `clear`) when the colliders change.
/// What a planner has learned about the world (surface columns and standability): kept between slices of a long search.
#[derive(Default)]
pub struct NavCache {
    surfaces: HashMap<(i32, i32), Vec<f32>>,
    standable: HashMap<Node, bool>,
}

/// A reachable-spot search that can be run a few hundred nodes at a time, so a game loop keeps going while it works.
pub struct ReachSearch {
    start: Node,
    open: BinaryHeap<Open>,
    g: HashMap<Node, f32>,
    expanded: usize,
    max_cost: f32,
    cfg: PlanConfig,
    pub done: bool,
}

impl ReachSearch {
    /// the nodes reached so far with their cheapest cost
    pub fn spots(&self) -> Vec<(Vec3, f32)> {
        self.g.iter().map(|(n, c)| (n.pos(), *c)).collect()
    }
    pub fn expanded(&self) -> usize {
        self.expanded
    }
}

pub struct Planner<'w, W: NavWorld> {
    world: &'w mut W,
    surfaces: HashMap<(i32, i32), Vec<f32>>,
    standable: HashMap<Node, bool>,
    /// the highest ledge a jump reaches (0 = no jumping moves), from the config of the current query
    jump_rise: f32,
}

impl<'w, W: NavWorld> Planner<'w, W> {
    pub fn new(world: &'w mut W) -> Self {
        Self { world, surfaces: HashMap::new(), standable: HashMap::new(), jump_rise: 0.0 }
    }
    /// a planner that starts with what an earlier slice learned
    pub fn with_cache(world: &'w mut W, cache: NavCache) -> Self {
        Self { world, surfaces: cache.surfaces, standable: cache.standable, jump_rise: 0.0 }
    }
    pub fn into_cache(self) -> NavCache {
        NavCache { surfaces: self.surfaces, standable: self.standable }
    }

    fn cell_surfaces(&mut self, ix: i32, iz: i32, from_y: f32) -> Vec<f32> {
        // cache the full column once (from far above), filter on use
        let cached = self.surfaces.entry((ix, iz)).or_insert_with(Vec::new);
        if cached.is_empty() {
            let col = self.world.surfaces((ix as f32 + 0.5) * CELL, (iz as f32 + 0.5) * CELL, 10_000.0);
            *cached = if col.is_empty() { vec![f32::NEG_INFINITY] } else { col }; // a sentinel marks "looked, nothing there"
        }
        cached.iter().copied().filter(|y| y.is_finite() && *y <= from_y).collect()
    }

    fn can_stand(&mut self, n: Node) -> bool {
        if let Some(v) = self.standable.get(&n) {
            return *v;
        }
        let p = n.pos();
        let v = !self.world.blocked(p.x, p.z, p.y + MAX_STEP, p.y + AVATAR_HEIGHT);
        self.standable.insert(n, v);
        v
    }

    /// the nearest standable node under a point: the highest surface not above `y + MAX_STEP`, or the highest when `y` is None
    fn node_at(&mut self, x: f32, z: f32, y: Option<f32>) -> Option<Node> {
        let (ix, iz) = (cell_of(x), cell_of(z));
        let from = y.map(|y| y + MAX_STEP).unwrap_or(f32::INFINITY);
        let mut best: Option<Node> = None;
        for s in self.cell_surfaces(ix, iz, from) {
            let n = Node { ix, iz, y: (s * 100.0).round() as i32 };
            if self.can_stand(n) {
                best = Some(n);
                break;
            }
        }
        best
    }

    fn neighbours(&mut self, n: Node, out: &mut Vec<(Node, f32, Move)>) {
        out.clear();
        let p = n.pos();
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let (jx, jz) = (n.ix + dx, n.iz + dz);
            let run = ((dx * dx + dz * dz) as f32).sqrt() * CELL;
            // diagonal moves need both orthogonal neighbours open so the capsule does not clip a corner
            if dx != 0 && dz != 0 {
                let (a, b) = (self.node_at((n.ix + dx) as f32 * CELL + 0.5 * CELL, p.z, Some(p.y)), self.node_at(p.x, (n.iz + dz) as f32 * CELL + 0.5 * CELL, Some(p.y)));
                if a.is_none() || b.is_none() {
                    continue;
                }
            }
            for s in self.cell_surfaces(jx, jz, p.y + MAX_STEP + MAX_RAMP_RISE * run) {
                let rise = s - p.y;
                if rise < -MAX_DROP {
                    break; // surfaces are highest first: the rest are deeper
                }
                let m = Node { ix: jx, iz: jz, y: (s * 100.0).round() as i32 };
                let q = m.pos();
                let how = if rise > 0.02 { Move::Climb } else if rise < -MAX_STEP { Move::Drop } else { Move::Walk };
                if rise > MAX_STEP {
                    // above a step: only a continuous ramp or stairs, checked at the midpoint
                    if rise > MAX_RAMP_RISE * run {
                        continue;
                    }
                    let mid = self.world.surfaces((p.x + q.x) * 0.5, (p.z + q.z) * 0.5, p.y + MAX_RAMP_RISE * run);
                    let want = (p.y + q.y) * 0.5;
                    if !mid.iter().any(|y| (y - want).abs() <= RAMP_TOLERANCE) {
                        continue;
                    }
                }
                if !self.can_stand(m) {
                    continue;
                }
                // sample the move for walls the cell centres miss
                let mut ok = true;
                for t in [0.25, 0.5, 0.75] {
                    let (x, z) = (p.x + (q.x - p.x) * t, p.z + (q.z - p.z) * t);
                    // a step or a drop is walked at the higher of the two heights (the avatar lifts over the riser before it reaches it); a ramp follows the line
                    let y = if rise.abs() <= MAX_STEP || how == Move::Drop { p.y.max(q.y) } else { p.y + (q.y - p.y) * t };
                    if self.world.blocked(x, z, y + MAX_STEP, y + AVATAR_HEIGHT) {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                let mut cost = (run * run + rise * rise).sqrt();
                if rise > 0.0 {
                    cost += rise * 0.6;
                } else if how == Move::Drop {
                    cost += 1.5 + (-rise) * 1.0;
                }
                out.push((m, cost, how));
                if how != Move::Drop {
                    break; // walking onto the nearest connected level: do not also list deeper ones
                }
            }
        }
        // jumps: up onto an adjacent ledge or platform no ramp reaches (a normal jump gets about 1.5 m, the Jump level's double jump about 8 m)
        if self.jump_rise > MAX_STEP {
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (jx, jz) = (n.ix + dx, n.iz + dz);
                for s in self.cell_surfaces(jx, jz, p.y + self.jump_rise) {
                    let rise = s - p.y;
                    if rise <= MAX_STEP || rise > self.jump_rise {
                        continue;
                    }
                    let m = Node { ix: jx, iz: jz, y: (s * 100.0).round() as i32 };
                    if out.iter().any(|(o, _, _)| *o == m) || !self.can_stand(m) {
                        continue;
                    }
                    // the air above the takeoff cell must be free up to the ledge (the avatar rises beside the wall before it moves over)
                    if self.world.blocked(p.x, p.z, p.y + MAX_STEP, s + AVATAR_HEIGHT) {
                        continue;
                    }
                    out.push((m, 2.5 + rise * 1.2, Move::Jump));
                }
            }
        }
    }

    /// Plan a walk from `from` to `to` (`to.y` is optional: any level of the goal cell will do, the highest standable one first).
    pub fn plan(&mut self, from: Vec3, to: Vec3, to_level: bool, cfg: &PlanConfig) -> Result<Plan, PlanError> {
        self.jump_rise = cfg.jump_rise;
        let start = self.node_at(from.x, from.z, Some(from.y)).ok_or(PlanError::NoStart)?;
        let level = if to_level { Some(to.y) } else { None };
        // a goal cell that is not standable (a prop or a wall has appeared there since it was listed) snaps to the nearest standable cell within 3 cells
        let goal_node = match self.node_at(to.x, to.z, level) {
            Some(n) => n,
            None => {
                let mut best: Option<(f32, Node)> = None;
                for dx in -3..=3 {
                    for dz in -3..=3 {
                        if dx == 0 && dz == 0 {
                            continue;
                        }
                        if let Some(n) = self.node_at(to.x + dx as f32 * CELL, to.z + dz as f32 * CELL, level) {
                            let d = ((dx * dx + dz * dz) as f32).sqrt();
                            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                                best = Some((d, n));
                            }
                        }
                    }
                }
                best.map(|(_, n)| n).ok_or(PlanError::NoGoal)?
            }
        };
        // a goal that asked for a level (a rooftop, a platform) must end on that level: a route that quietly ends on the ground beside it is a failure the caller can act on
        if to_level && (goal_node.pos().y - to.y).abs() > 1.5 {
            return Err(PlanError::NoGoal);
        }
        let goal_pos = goal_node.pos();
        let h = |n: &Node| -> f32 { (n.pos() - goal_pos).length() };
        let mut open = BinaryHeap::new();
        let mut g: HashMap<Node, f32> = HashMap::new();
        let mut came: HashMap<Node, (Node, Move)> = HashMap::new();
        g.insert(start, 0.0);
        open.push(Open { f: h(&start), node: start });
        let mut expanded = 0;
        let mut buf = Vec::new();
        let mut found: Option<Node> = None;
        while let Some(Open { node, f: _ }) = open.pop() {
            let gc = *g.get(&node).unwrap_or(&f32::MAX);
            let pos = node.pos();
            let near = ((pos.x - goal_pos.x).powi(2) + (pos.z - goal_pos.z).powi(2)).sqrt() <= cfg.goal_tolerance.max(CELL * 0.5);
            if near && (!to_level || (pos.y - goal_pos.y).abs() <= 1.0) {
                found = Some(node);
                break;
            }
            expanded += 1;
            if expanded > cfg.max_nodes {
                break;
            }
            self.neighbours(node, &mut buf);
            for (m, cost, how) in buf.clone() {
                if (m.ix - start.ix).abs() > cfg.radius_cells || (m.iz - start.iz).abs() > cfg.radius_cells {
                    continue;
                }
                let ng = gc + cost;
                if ng < *g.get(&m).unwrap_or(&f32::MAX) {
                    g.insert(m, ng);
                    came.insert(m, (node, how));
                    open.push(Open { f: ng + h(&m), node: m });
                }
            }
        }
        let Some(end) = found else { return Err(PlanError::NoPath { expanded }) };
        let cost = *g.get(&end).unwrap_or(&0.0);
        let mut chain = vec![(end, Move::Walk)];
        let mut cur = end;
        while let Some((prev, how)) = came.get(&cur) {
            chain.last_mut().unwrap().1 = *how;
            chain.push((*prev, Move::Walk));
            cur = *prev;
        }
        chain.reverse();
        // chain[i].1 is how the move INTO chain[i] was made (the start's is Walk)
        let mut wps: Vec<Waypoint> = Vec::new();
        for (i, (n, how)) in chain.iter().enumerate() {
            let w = Waypoint { pos: n.pos(), how: if i == 0 { Move::Walk } else { *how } };
            wps.push(w);
        }
        Ok(Plan { waypoints: simplify(&wps), cost, expanded })
    }

    /// Every standable node reachable from `from` within `max_cost` (cheapest cost to each), for choosing hiding spots: `(position, cost)`.
    pub fn reachable(&mut self, from: Vec3, max_cost: f32, cfg: &PlanConfig) -> Vec<(Vec3, f32)> {
        let Some(mut search) = self.reach_start(from, max_cost, cfg) else { return Vec::new() };
        self.reach_step(&mut search, usize::MAX);
        search.spots()
    }

    /// Begin a reachable-spot search (see [`Planner::reach_step`]); `None` when there is nowhere to stand at `from`.
    pub fn reach_start(&mut self, from: Vec3, max_cost: f32, cfg: &PlanConfig) -> Option<ReachSearch> {
        self.jump_rise = cfg.jump_rise;
        let start = self.node_at(from.x, from.z, Some(from.y))?;
        let mut open = BinaryHeap::new();
        let mut g: HashMap<Node, f32> = HashMap::new();
        g.insert(start, 0.0);
        open.push(Open { f: 0.0, node: start });
        Some(ReachSearch { start, open, g, expanded: 0, max_cost, cfg: cfg.clone(), done: false })
    }

    /// Expand up to `budget` more nodes of a search; `search.done` is set when the search has finished (or hit its node limit).
    pub fn reach_step(&mut self, search: &mut ReachSearch, budget: usize) {
        self.jump_rise = search.cfg.jump_rise;
        let mut buf = Vec::new();
        let mut spent = 0usize;
        while spent < budget {
            let Some(Open { node, f }) = search.open.pop() else {
                search.done = true;
                return;
            };
            if f > *search.g.get(&node).unwrap_or(&f32::MAX) + 1e-4 {
                continue;
            }
            search.expanded += 1;
            spent += 1;
            if search.expanded > search.cfg.max_nodes {
                search.done = true;
                return;
            }
            self.neighbours(node, &mut buf);
            for (m, cost, _) in buf.clone() {
                let ng = f + cost;
                if ng > search.max_cost || (m.ix - search.start.ix).abs() > search.cfg.radius_cells || (m.iz - search.start.iz).abs() > search.cfg.radius_cells {
                    continue;
                }
                if ng < *search.g.get(&m).unwrap_or(&f32::MAX) {
                    search.g.insert(m, ng);
                    search.open.push(Open { f: ng, node: m });
                }
            }
        }
    }
}

/// Drop waypoints that continue straight on at the same slope and move kind.
fn simplify(w: &[Waypoint]) -> Vec<Waypoint> {
    if w.len() <= 2 {
        return w.to_vec();
    }
    let mut out = vec![w[0]];
    for i in 1..w.len() - 1 {
        let a = w[i].pos - out.last().unwrap().pos;
        let b = w[i + 1].pos - w[i].pos;
        let same = w[i].how == w[i + 1].how && w[i].how != Move::Jump && a.normalize_or_zero().dot(b.normalize_or_zero()) > 0.999;
        if !same {
            out.push(w[i]);
        }
    }
    out.push(*w.last().unwrap());
    out
}

/// A glide: run off a ledge at the starting height, open the glider and fly in a straight line to a landing point, closing the glider above it to
/// drop the last part. The glider moves `glide_speed` m/s forward and sinks `sink` m/s, so after `d` metres the avatar is `from.y - sink * d / glide_speed`
/// high; the landing must be a standable surface that the flight reaches from above.
#[derive(Clone, Copy, Debug)]
pub struct GlideChoice {
    /// where the avatar runs off the edge (at the starting height)
    pub ledge: Vec3,
    pub landing: Vec3,
    pub bearing: f32,
}

/// Is the straight glide from `ledge` to `goal` flyable? The flight altitude at each metre must clear everything (the avatar cylinder at flight height overlaps
/// nothing) and must still be at or above the goal surface when it arrives (closing the glider drops the rest).
pub fn glide_clear<W: NavWorld>(world: &mut W, ledge: Vec3, goal: Vec3, glide_speed: f32, sink: f32) -> bool {
    let d = Vec3::new(goal.x - ledge.x, 0.0, goal.z - ledge.z);
    let len = d.length();
    if len < 1.0 {
        return ledge.y - goal.y >= 0.5;
    }
    let dir = d / len;
    let mut run = 1.0;
    while run < len {
        let y = ledge.y - sink * run / glide_speed;
        let (x, z) = (ledge.x + dir.x * run, ledge.z + dir.z * run);
        if world.blocked(x, z, y, y + AVATAR_HEIGHT) {
            return false;
        }
        run += 1.0;
    }
    // it must still be above the landing surface when it gets there (it can close the glider and drop, but not climb)
    ledge.y - sink * len / glide_speed >= goal.y - 0.2
}

/// The first ledge along a bearing from `from`: the distance where the standing surface falls away from the current level (None if there is a wall first
/// or no edge within `max_run`).
pub fn find_ledge<W: NavWorld>(world: &mut W, from: Vec3, bearing: f32, max_run: f32) -> Option<f32> {
    let (dx, dz) = (bearing.cos(), bearing.sin());
    let mut run = 1.0;
    while run <= max_run {
        let (x, z) = (from.x + dx * run, from.z + dz * run);
        if world.blocked(x, z, from.y + MAX_STEP, from.y + AVATAR_HEIGHT) {
            return None; // a wall at this level
        }
        let top = world.surfaces(x, z, from.y + MAX_STEP).first().copied();
        if top.map(|t| from.y - t > MAX_STEP + 0.01).unwrap_or(true) {
            return Some(run);
        }
        run += 1.0;
    }
    None
}

/// Glide options from a high standing point: for each of `directions` bearings, the ledge and every landing at the given `distances` (metres from the
/// ledge) that is a standable surface at least 1 m below the start and reachable by a clear straight glide.
pub fn glide_options<W: NavWorld>(world: &mut W, from: Vec3, directions: usize, distances: &[f32], glide_speed: f32, sink: f32) -> Vec<GlideChoice> {
    let mut out = Vec::new();
    for k in 0..directions {
        let a = k as f32 / directions as f32 * std::f32::consts::TAU;
        let Some(start) = find_ledge(world, from, a, 40.0) else { continue };
        let ledge = Vec3::new(from.x + a.cos() * start, from.y, from.z + a.sin() * start);
        for &dist in distances {
            let (x, z) = (ledge.x + a.cos() * dist, ledge.z + a.sin() * dist);
            let flight_y = from.y - sink * dist / glide_speed;
            // the standable surface the flight lands on: the highest one under the flight height that is also well below the start
            let Some(s) = world.surfaces(x, z, flight_y + 0.2).into_iter().find(|s| from.y - *s >= 1.0 && !world.blocked(x, z, *s + MAX_STEP, *s + AVATAR_HEIGHT)) else { continue };
            let goal = Vec3::new(x, s, z);
            if glide_clear(world, ledge, goal, glide_speed, sink) {
                out.push(GlideChoice { ledge, landing: goal, bearing: a });
            }
        }
    }
    out
}

/// A goal-directed glide (`glide_to`): the Synth is on a high spot and wants to reach `goal`, a lower standable point. Try the bearing towards the goal and
/// a few either side for a ledge from which the straight glide to the goal is clear; None = it cannot get there by gliding (the caller walks instead).
pub fn glide_to_goal<W: NavWorld>(world: &mut W, from: Vec3, goal: Vec3, glide_speed: f32, sink: f32) -> Option<GlideChoice> {
    if from.y - goal.y < 1.0 {
        return None;
    }
    let toward = (goal.z - from.z).atan2(goal.x - from.x);
    let mut tried: Vec<f32> = vec![0.0];
    for deg in [15.0f32, 30.0, 45.0, 70.0, 90.0] {
        tried.push(deg.to_radians());
        tried.push(-deg.to_radians());
    }
    let mut best: Option<(f32, GlideChoice)> = None;
    for off in tried {
        let a = toward + off;
        let Some(run) = find_ledge(world, from, a, 40.0) else { continue };
        let ledge = Vec3::new(from.x + a.cos() * run, from.y, from.z + a.sin() * run);
        if glide_clear(world, ledge, goal, glide_speed, sink) {
            let cost = run + off.abs() * 10.0; // short run, small detour
            if best.map(|(c, _)| cost < c).unwrap_or(true) {
                best = Some((cost, GlideChoice { ledge, landing: goal, bearing: a }));
            }
        }
    }
    best.map(|(_, c)| c)
}

#[cfg(test)]
mod tests;
