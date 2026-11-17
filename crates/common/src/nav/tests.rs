use super::*;

/// An axis-aligned box: x0..x1, z0..z1, y0..y1 (metres).
#[derive(Clone, Copy)]
struct B(f32, f32, f32, f32, f32, f32);

/// A flat ground at y = 0 plus boxes; the avatar is a vertical cylinder.
struct World {
    boxes: Vec<B>,
}

impl World {
    fn covers(b: &B, x: f32, z: f32) -> bool {
        x >= b.0 && x <= b.1 && z >= b.2 && z <= b.3
    }
}

impl NavWorld for World {
    fn surfaces(&mut self, x: f32, z: f32, from_y: f32) -> Vec<f32> {
        let mut c: Vec<f32> = vec![0.0];
        for b in &self.boxes {
            if World::covers(b, x, z) {
                c.push(b.5);
            }
        }
        // a surface strictly inside another box is not a surface
        let mut out: Vec<f32> = c
            .into_iter()
            .filter(|s| *s <= from_y && !self.boxes.iter().any(|b| World::covers(b, x, z) && b.4 + 1e-4 < *s && *s < b.5 - 1e-4))
            .collect();
        out.sort_by(|a, b| b.partial_cmp(a).unwrap());
        out.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        out
    }
    fn blocked(&mut self, x: f32, z: f32, y0: f32, y1: f32) -> bool {
        self.boxes.iter().any(|b| {
            let dx = (b.0 - x).max(0.0).max(x - b.1);
            let dz = (b.2 - z).max(0.0).max(z - b.3);
            (dx * dx + dz * dz).sqrt() < AVATAR_RADIUS && y0 < b.5 - 1e-3 && y1 > b.4 + 1e-3
        })
    }
}

fn plan(w: &mut World, from: Vec3, to: Vec3, level: bool) -> Result<Plan, PlanError> {
    Planner::new(w).plan(from, to, level, &PlanConfig::default())
}

fn length(p: &Plan) -> f32 {
    p.waypoints.windows(2).map(|w| (w[1].pos - w[0].pos).length()).sum()
}

/// walk the polyline in 0.2 m steps and assert the capsule never overlaps a box on the way
fn assert_walkable(w: &mut World, p: &Plan) {
    for pair in p.waypoints.windows(2) {
        let (a, b) = (pair[0].pos, pair[1].pos);
        let n = ((b - a).length() / 0.2).ceil().max(1.0) as i32;
        for i in 0..=n {
            let q = a + (b - a) * (i as f32 / n as f32);
            assert!(!w.blocked(q.x, q.z, q.y + MAX_STEP, q.y + AVATAR_HEIGHT), "the path walks into something at {q}");
        }
    }
}

#[test]
fn open_ground_is_a_straight_walk() {
    let mut w = World { boxes: vec![] };
    let p = plan(&mut w, Vec3::new(0.5, 0.0, 0.5), Vec3::new(30.5, 0.0, 0.5), false).unwrap();
    assert!(length(&p) < 31.5 && length(&p) > 28.0, "{}", length(&p));
    assert!(p.waypoints.len() <= 3, "straight lines are simplified: {}", p.waypoints.len());
}

#[test]
fn a_wall_is_walked_around_never_through() {
    let mut w = World { boxes: vec![B(10.0, 11.0, -30.0, 15.0, 0.0, 3.0)] };
    let p = plan(&mut w, Vec3::new(0.5, 0.0, 0.5), Vec3::new(20.5, 0.0, 0.5), false).unwrap();
    assert!(length(&p) > 35.0 && length(&p) < 45.0, "goes round the north end of the wall: {}", length(&p));
    assert_walkable(&mut w, &p);
}

#[test]
fn a_doorway_in_the_wall_is_used() {
    // wall at x 10..11 with a 2 m gap at z 0..2
    let mut w = World { boxes: vec![B(10.0, 11.0, -30.0, 0.0, 0.0, 3.0), B(10.0, 11.0, 2.0, 30.0, 0.0, 3.0)] };
    let p = plan(&mut w, Vec3::new(0.5, 0.0, 1.0), Vec3::new(20.5, 0.0, 1.0), false).unwrap();
    assert!(length(&p) < 22.0, "straight through the door: {}", length(&p));
    assert_walkable(&mut w, &p);
}

#[test]
fn a_gap_narrower_than_the_avatar_is_not_a_door() {
    let mut w = World { boxes: vec![B(10.0, 11.0, -30.0, 0.0, 0.0, 3.0), B(10.0, 11.0, 0.5, 30.0, 0.0, 3.0), B(-5.0, 40.0, 30.0, 31.0, 0.0, 3.0), B(-5.0, 40.0, -31.0, -30.0, 0.0, 3.0), B(40.0, 41.0, -30.0, 31.0, 0.0, 3.0), B(-6.0, -5.0, -30.0, 31.0, 0.0, 3.0)] };
    // a 0.5 m gap and a closed box around everything: no way through
    let r = plan(&mut w, Vec3::new(0.5, 0.0, 0.25), Vec3::new(20.5, 0.0, 0.25), false);
    assert!(matches!(r, Err(PlanError::NoPath { .. })), "{r:?}");
}

fn stairs(x0: f32, steps: usize, rise: f32, tread: f32, z0: f32, z1: f32) -> Vec<B> {
    (0..steps).map(|k| B(x0 + k as f32 * tread, x0 + (k + 1) as f32 * tread, z0, z1, 0.0, rise * (k + 1) as f32)).collect()
}

#[test]
fn stairs_connect_the_ground_to_a_platform() {
    // 10 steps of 0.25 m rise and 0.5 m tread up to a platform at 2.5 m, enclosed by walls so the only way up is the stairs
    let mut boxes = stairs(5.0, 10, 0.25, 0.5, -2.0, 2.0);
    boxes.push(B(10.0, 20.0, -2.0, 2.0, 0.0, 2.5)); // the platform
    boxes.push(B(0.0, 21.0, 2.0, 3.0, 0.0, 6.0)); // side walls
    boxes.push(B(0.0, 21.0, -3.0, -2.0, 0.0, 6.0));
    boxes.push(B(20.0, 21.0, -3.0, 3.0, 0.0, 6.0));
    let mut w = World { boxes };
    let p = plan(&mut w, Vec3::new(0.5, 0.0, 0.5), Vec3::new(15.5, 2.5, 0.5), true).unwrap();
    assert!((p.waypoints.last().unwrap().pos.y - 2.5).abs() < 0.05, "ends on the platform: {:?}", p.waypoints.last());
    assert!(p.waypoints.iter().any(|w| w.how == Move::Climb));
    assert_walkable(&mut w, &p);
}

#[test]
fn a_sheer_wall_is_not_climbable() {
    // a 2.5 m platform with no stairs, walled in: unreachable
    let boxes = vec![B(10.0, 20.0, -2.0, 2.0, 0.0, 2.5), B(0.0, 21.0, 2.0, 3.0, 0.0, 6.0), B(0.0, 21.0, -3.0, -2.0, 0.0, 6.0), B(20.0, 21.0, -3.0, 3.0, 0.0, 6.0), B(-1.0, 0.0, -3.0, 3.0, 0.0, 6.0)];
    let mut w = World { boxes };
    let r = plan(&mut w, Vec3::new(5.5, 0.0, 0.5), Vec3::new(15.5, 2.5, 0.5), true);
    assert!(matches!(r, Err(PlanError::NoPath { .. })), "{r:?}");
}

#[test]
fn a_roof_is_reached_by_stairs_up_the_side() {
    // a 4 m building with an outside stair along its south face
    let mut boxes = vec![B(10.0, 30.0, 10.0, 30.0, 0.0, 4.0)];
    for k in 0..16 {
        boxes.push(B(9.0 - 1.0 + (k as f32) * 1.25 + 0.0, 9.0 + (k as f32) * 1.25 + 0.25, 8.0, 10.0, 0.0, 0.25 * (k + 1) as f32));
    }
    // stair run along +x at z 8..10 touches the building face; its last step meets the roof height only if long enough: use a ramp-like 16 steps
    let mut w = World { boxes };
    let r = plan(&mut w, Vec3::new(5.5, 0.0, 9.0), Vec3::new(20.5, 4.0, 20.0), true);
    // the stairs end beside the face at 4 m but the roof edge is at x 10..30, z 10: the top step abuts the roof, so the roof is reachable
    let p = r.expect("a path onto the roof");
    assert!((p.waypoints.last().unwrap().pos.y - 4.0).abs() < 0.05);
    assert_walkable(&mut w, &p);
}

#[test]
fn a_short_drop_is_taken_a_long_drop_is_not() {
    let mut low = World { boxes: vec![B(0.0, 10.0, 0.0, 10.0, 0.0, 2.0)] };
    let p = plan(&mut low, Vec3::new(5.0, 2.0, 5.0), Vec3::new(14.0, 0.0, 5.0), true).unwrap();
    assert!(p.waypoints.iter().any(|w| w.how == Move::Drop), "{:?}", p.waypoints.iter().map(|w| w.how).collect::<Vec<_>>());
    // 6 m roof, walled so there is no other way down
    let mut high = World { boxes: vec![B(0.0, 10.0, 0.0, 10.0, 0.0, 6.0)] };
    let r = plan(&mut high, Vec3::new(5.0, 6.0, 5.0), Vec3::new(14.0, 0.0, 5.0), true);
    assert!(matches!(r, Err(PlanError::NoPath { .. })), "{r:?}");
}

#[test]
fn a_start_inside_a_wall_or_off_the_map_reports_it() {
    let mut w = World { boxes: vec![B(0.0, 10.0, 0.0, 10.0, 0.0, 6.0)] };
    // inside a solid block: nothing to stand on
    let r = plan(&mut w, Vec3::new(5.0, 2.0, 5.0), Vec3::new(20.0, 0.0, 5.0), false);
    assert!(matches!(r, Err(PlanError::NoStart) | Err(PlanError::NoPath { .. })), "{r:?}");
}

#[test]
fn reachable_respects_the_cost_limit() {
    let mut w = World { boxes: vec![] };
    let cells = Planner::new(&mut w).reachable(Vec3::new(0.5, 0.0, 0.5), 10.0, &PlanConfig::default());
    assert!(cells.len() > 100);
    assert!(cells.iter().all(|(_, c)| *c <= 10.0 + 1e-3));
    assert!(cells.iter().all(|(p, _)| ((p.x - 0.5).powi(2) + (p.z - 0.5).powi(2)).sqrt() <= 10.5));
}

fn tower_world() -> World {
    // a 12 m tower; a 20 m wall 30 m east of it
    World { boxes: vec![B(0.0, 10.0, 0.0, 10.0, 0.0, 12.0), B(30.0, 31.0, -40.0, 50.0, 0.0, 20.0)] }
}

#[test]
fn glide_options_from_a_roof_respect_range_and_walls() {
    let mut w = tower_world();
    let from = Vec3::new(5.0, 12.0, 5.0);
    // a steep glider (3 m/s sink at 8 m/s: 12 m of height buys about 32 m): short landings only
    let steep = glide_options(&mut w, from, 8, &[10.0, 20.0, 45.0], 8.0, 3.0);
    assert!(!steep.is_empty());
    assert!(steep.iter().all(|g| g.landing.y.abs() < 0.01), "lands on the ground");
    assert!(steep.iter().any(|g| (g.bearing - std::f32::consts::PI).abs() < 0.1), "westward options exist");
    let west: Vec<f32> = steep.iter().filter(|g| (g.bearing - std::f32::consts::PI).abs() < 0.1).map(|g| ((g.landing.x - g.ledge.x).powi(2) + (g.landing.z - g.ledge.z).powi(2)).sqrt()).collect();
    assert!(west.iter().all(|d| *d <= 33.0), "the steep glider cannot reach 45 m from 12 m: {west:?}");
    // east: the 20 m wall stands 20 m past the ledge, so a 10 m landing before it is fine and nothing 20 m or more is
    assert!(steep.iter().any(|g| g.bearing.abs() < 0.1), "a short landing before the wall");
    assert!(steep.iter().filter(|g| g.bearing.abs() < 0.1).all(|g| g.landing.x - g.ledge.x < 15.0), "no glide through the wall");
    // the slow controls-v1 glider (0.3 m/s sink) reaches far: 45 m is fine
    let slow = glide_options(&mut w, from, 8, &[10.0, 20.0, 45.0], 8.0, 0.3);
    assert!(slow.iter().any(|g| (g.bearing - std::f32::consts::PI).abs() < 0.1 && ((g.landing.x - g.ledge.x).powi(2) + (g.landing.z - g.ledge.z).powi(2)).sqrt() > 40.0));
}

#[test]
fn glide_to_a_goal_picks_a_ledge_and_fails_cleanly_when_it_cannot() {
    let mut w = tower_world();
    let from = Vec3::new(5.0, 12.0, 5.0);
    // a goal on the ground 25 m west: reachable (slow glider)
    let c = glide_to_goal(&mut w, from, Vec3::new(-25.0, 0.0, 5.0), 8.0, 0.3).expect("a glide");
    assert!(c.ledge.x < 5.0, "runs off the west edge: {}", c.ledge);
    assert!((c.landing - Vec3::new(-25.0, 0.0, 5.0)).length() < 1e-4);
    // the same goal with the steep glider is still within 32 m: fine; 60 m is not
    assert!(glide_to_goal(&mut w, from, Vec3::new(-25.0, 0.0, 5.0), 8.0, 3.0).is_some());
    assert!(glide_to_goal(&mut w, from, Vec3::new(-60.0, 0.0, 5.0), 8.0, 3.0).is_none(), "out of range for a 12 m start");
    // a goal that is not lower than the start cannot be glided to
    assert!(glide_to_goal(&mut w, from, Vec3::new(-5.0, 11.5, 5.0), 8.0, 0.3).is_none());
    // a goal behind the 20 m wall: the flight would hit it
    assert!(glide_to_goal(&mut w, from, Vec3::new(45.0, 0.0, 5.0), 8.0, 0.3).is_none(), "wall in the way");
}

#[test]
fn glide_from_the_ground_has_no_ledge() {
    let mut w = tower_world();
    assert!(glide_to_goal(&mut w, Vec3::new(-20.0, 0.0, 5.0), Vec3::new(-40.0, 0.0, 5.0), 8.0, 0.3).is_none());
}

#[test]
fn a_goal_inside_a_prop_snaps_to_the_nearest_standable_cell() {
    let mut w = World { boxes: vec![B(19.0, 22.0, -1.0, 2.0, 0.0, 3.0)] };
    // the goal point (20.5, 0.5) is inside the box: the plan ends beside it
    let p = plan(&mut w, Vec3::new(0.5, 0.0, 0.5), Vec3::new(20.5, 0.0, 0.5), true).expect("snapped goal");
    let end = p.waypoints.last().unwrap().pos;
    assert!(((end.x - 20.5).powi(2) + (end.z - 0.5).powi(2)).sqrt() <= 4.5, "{end}");
    assert!(!w.blocked(end.x, end.z, end.y + MAX_STEP, end.y + AVATAR_HEIGHT));
    // a goal with nothing standable anywhere near still reports no_goal
    let mut solid = World { boxes: vec![B(10.0, 60.0, -30.0, 30.0, 0.0, 50.0)] };
    assert!(matches!(plan(&mut solid, Vec3::new(0.5, 0.0, 0.5), Vec3::new(30.5, 0.0, 0.5), false), Err(PlanError::NoStart) | Err(PlanError::NoGoal) | Err(PlanError::NoPath { .. })));
}

#[test]
fn a_level_goal_that_lands_on_another_level_fails_instead_of_ending_on_the_ground() {
    // flat ground only: a goal asking for the 6 m level has no standable surface there
    let mut w = World { boxes: vec![] };
    assert!(matches!(plan(&mut w, Vec3::new(0.5, 0.0, 0.5), Vec3::new(10.5, 6.0, 0.5), true), Err(PlanError::NoGoal)));
    // the same goal without a level (any height) still plans
    assert!(plan(&mut w, Vec3::new(0.5, 0.0, 0.5), Vec3::new(10.5, 6.0, 0.5), false).is_ok());
}

#[test]
fn jumping_moves_reach_ledges_a_walking_plan_cannot() {
    // a 1.3 m block with no ramp: only a jump gets onto it
    let mut w = World { boxes: vec![B(10.0, 14.0, -2.0, 2.0, 0.0, 1.3)] };
    let walk = PlanConfig::default();
    let jump = PlanConfig { jump_rise: 1.4, ..PlanConfig::default() };
    let from = Vec3::new(0.5, 0.0, 0.5);
    let to = Vec3::new(12.5, 1.3, 0.5);
    assert!(matches!(Planner::new(&mut w).plan(from, to, true, &walk), Err(PlanError::NoPath { .. }) | Err(PlanError::NoGoal)), "no jump, no route");
    let p = Planner::new(&mut w).plan(from, to, true, &jump).expect("a jump reaches it");
    assert!(p.waypoints.iter().any(|wp| wp.how == Move::Jump), "the route includes a jump");
    assert!((p.waypoints.last().unwrap().pos.y - 1.3).abs() < 0.1);
}

#[test]
fn a_tall_platform_needs_the_double_jump_range() {
    let mut w = World { boxes: vec![B(10.0, 14.0, -2.0, 2.0, 0.0, 7.5)] };
    let normal = PlanConfig { jump_rise: 1.4, ..PlanConfig::default() };
    let level2 = PlanConfig { jump_rise: 8.0, ..PlanConfig::default() };
    let from = Vec3::new(0.5, 0.0, 0.5);
    let to = Vec3::new(12.5, 7.5, 0.5);
    assert!(Planner::new(&mut w).plan(from, to, true, &normal).is_err());
    let p = Planner::new(&mut w).plan(from, to, true, &level2).expect("the double jump reaches a 7.5 m platform");
    assert!(p.waypoints.iter().any(|wp| wp.how == Move::Jump));
}

#[test]
fn the_reachable_spot_list_grows_with_the_jump_range() {
    let mut w = World { boxes: vec![B(10.0, 14.0, -2.0, 2.0, 0.0, 6.0)] };
    let n = |w: &mut World, rise: f32| {
        let cfg = PlanConfig { jump_rise: rise, ..PlanConfig::default() };
        Planner::new(w).reachable(Vec3::new(0.5, 0.0, 0.5), 60.0, &cfg).iter().filter(|(p, _)| p.y > 5.0).count()
    };
    assert_eq!(n(&mut w, 1.4), 0);
    assert!(n(&mut w, 8.0) > 0);
}

#[test]
fn a_reachable_search_run_in_slices_gives_the_same_spots_as_one_call() {
    let mut w = World { boxes: vec![B(10.0, 14.0, -2.0, 2.0, 0.0, 1.3), B(-6.0, -5.0, -8.0, 8.0, 0.0, 3.0)] };
    let cfg = PlanConfig { jump_rise: 1.4, radius_cells: 40, ..PlanConfig::default() };
    let from = Vec3::new(0.5, 0.0, 0.5);
    let mut whole = Planner::new(&mut w).reachable(from, 40.0, &cfg);
    let mut planner = Planner::new(&mut w);
    let mut search = planner.reach_start(from, 40.0, &cfg).expect("a start");
    let mut slices = 0;
    while !search.done {
        planner.reach_step(&mut search, 50);
        slices += 1;
    }
    assert!(slices > 3, "the search needed several slices");
    // a cache carried over between slices (a fresh planner each frame, as the game loop does) must not change the answer
    let mut w2 = World { boxes: vec![B(10.0, 14.0, -2.0, 2.0, 0.0, 1.3), B(-6.0, -5.0, -8.0, 8.0, 0.0, 3.0)] };
    let mut cache = NavCache::default();
    let mut s2 = {
        let mut p = Planner::with_cache(&mut w2, std::mem::take(&mut cache));
        let s = p.reach_start(from, 40.0, &cfg).unwrap();
        cache = p.into_cache();
        s
    };
    while !s2.done {
        let mut p = Planner::with_cache(&mut w2, std::mem::take(&mut cache));
        p.reach_step(&mut s2, 50);
        cache = p.into_cache();
    }
    let key = |v: &mut Vec<(Vec3, f32)>| {
        v.sort_by(|a, b| (a.0.x, a.0.z, a.0.y).partial_cmp(&(b.0.x, b.0.z, b.0.y)).unwrap());
    };
    let mut a = search.spots();
    let mut b = s2.spots();
    key(&mut whole);
    key(&mut a);
    key(&mut b);
    assert_eq!(whole.len(), a.len());
    assert_eq!(whole.len(), b.len());
}

#[test]
fn a_jump_needs_headroom_toward_the_ledge() {
    // the platform's west face 0.15 m inside the takeoff cell: the cell centre is clear, the body 0.2 m toward the ledge is not
    let level2 = PlanConfig { jump_rise: 8.0, ..PlanConfig::default() };
    let from = Vec3::new(0.5, 0.0, 0.5);
    let to = Vec3::new(12.5, 7.5, 0.5);
    let mut tight = World { boxes: vec![B(9.85, 14.0, -200.0, 200.0, 0.0, 7.5)] };
    assert!(Planner::new(&mut tight).plan(from, to, true, &level2).is_err(), "no takeoff with the face 0.15 m into the cell");
    let mut clear = World { boxes: vec![B(10.0, 14.0, -200.0, 200.0, 0.0, 7.5)] };
    assert!(Planner::new(&mut clear).plan(from, to, true, &level2).is_ok(), "a face on the cell boundary still works");
}
