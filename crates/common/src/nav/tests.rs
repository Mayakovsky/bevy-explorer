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

#[test]
fn glide_from_a_roof_lands_about_2_67_times_the_height_away_and_not_through_walls() {
    // a 12 m tower; a wall 20 m east of it, 20 m high
    let mut w = World { boxes: vec![B(0.0, 10.0, 0.0, 10.0, 0.0, 12.0), B(30.0, 31.0, -40.0, 50.0, 0.0, 20.0)] };
    let from = Vec3::new(5.0, 12.0, 5.0);
    let landings = glide_landings(&mut w, from, 8, 8.0, 3.0);
    assert!(!landings.is_empty());
    for (p, _) in &landings {
        assert!(p.y.abs() < 1.0, "lands on the ground: {p}");
    }
    // west (bearing pi): open, lands near 12 * 8/3 = 32 m away
    let west = landings.iter().find(|(_, a)| (a - std::f32::consts::PI).abs() < 0.1).expect("a westward landing");
    let d = ((west.0.x - 5.0).powi(2) + (west.0.z - 5.0).powi(2)).sqrt();
    // 5.5 m run to the ledge (it starts from the middle of the roof) plus 12 m x 8/3 of glide
    assert!(d > 33.0 && d < 42.0, "{d}");
    // east (bearing 0): the 20 m wall 25 m past the ledge is in the way, so no landing
    assert!(landings.iter().all(|(_, a)| a.abs() > 0.1), "no landing through the wall");
}
