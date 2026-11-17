use super::*;
use bevy::math::Vec3;

#[derive(Clone, Copy)]
struct B(f32, f32, f32, f32, f32, f32); // x0 x1 z0 z1 y0 y1

struct World {
    boxes: Vec<B>,
}
impl NavWorld for World {
    fn surfaces(&mut self, x: f32, z: f32, from_y: f32) -> Vec<f32> {
        let mut c = vec![0.0f32];
        for b in &self.boxes {
            if x >= b.0 && x <= b.1 && z >= b.2 && z <= b.3 {
                c.push(b.5);
            }
        }
        let mut out: Vec<f32> = c.into_iter().filter(|s| *s <= from_y).collect();
        out.sort_by(|a, b| b.partial_cmp(a).unwrap());
        out
    }
    fn blocked(&mut self, x: f32, z: f32, y0: f32, y1: f32) -> bool {
        self.boxes.iter().any(|b| {
            let dx = (b.0 - x).max(0.0).max(x - b.1);
            let dz = (b.2 - z).max(0.0).max(z - b.3);
            (dx * dx + dz * dz).sqrt() < 0.3 && y0 < b.5 && y1 > b.4
        })
    }
}

/// Fly at a fixed look for `secs` at 60 Hz from (0, y0, 0) with an initial velocity; returns (state, peak y, final y).
fn fly(pitch_deg: f32, secs: f32, y0: f32, v0: Vec3) -> (FlightState, f32, f32) {
    let p = pitch_deg.to_radians();
    let look = Vec3::new(p.cos(), p.sin(), 0.0);
    let mut st = FlightState::start(v0);
    let (mut y, mut peak) = (y0, y0);
    for _ in 0..(secs * 60.0) as i32 {
        let v = step(&mut st, look, true, y, 1.0 / 60.0);
        y += v.y / 60.0;
        peak = peak.max(y);
    }
    (st, peak, y)
}

#[test]
fn level_flight_reaches_the_horizontal_top_speed_and_holds_it() {
    let (st, _, y) = fly(0.0, 8.0, 50.0, Vec3::ZERO);
    assert!(st.h_speed() > 18.0 && st.h_speed() <= 20.0, "{}", st.h_speed());
    assert!(st.vel.length() <= 24.3 + 0.5);
    assert!((y - 50.0).abs() < 8.0, "holds height, got {y}");
}

#[test]
fn a_climb_at_45_degrees_reaches_about_the_climb_rate() {
    let (st, _, _) = fly(45.0, 6.0, 30.0, Vec3::ZERO);
    assert!(st.vy_sm > 8.0 && st.vy_sm < 16.5, "{}", st.vy_sm);
    assert!(st.vel.length() <= 24.3 + 1.0);
}

#[test]
fn straight_up_never_passes_the_ceiling() {
    let (_, peak, _) = fly(90.0, 40.0, 30.0, Vec3::ZERO);
    assert!(peak <= CEILING + 1.0, "peak {peak}");
    assert!(peak > 150.0, "it does climb: {peak}");
}

#[test]
fn a_dive_stays_under_the_dive_limit() {
    let (st, _, y) = fly(-60.0, 5.0, 150.0, Vec3::ZERO);
    assert!(st.vel.y < -8.0 && st.vel.y > -23.0, "{}", st.vel.y);
    assert!(y < 140.0);
}

#[test]
fn no_thrust_the_glider_sinks_slowly_and_the_speed_bleeds_off() {
    let mut st = FlightState::start(Vec3::new(15.0, 0.0, 0.0));
    let mut y = 80.0;
    for _ in 0..600 {
        let v = step(&mut st, Vec3::X, false, y, 1.0 / 60.0);
        y += v.y / 60.0;
    }
    assert!((st.vel.y + FLY_SINK).abs() < 0.05, "{}", st.vel.y);
    assert!(st.h_speed() < 15.0);
    assert!(y > 70.0 && y < 80.0);
}

#[test]
fn the_model_is_frame_rate_independent_enough() {
    let run = |hz: f32| {
        let mut st = FlightState::start(Vec3::ZERO);
        let mut y = 50.0;
        for _ in 0..(6.0 * hz) as i32 {
            let v = step(&mut st, Vec3::X, true, y, 1.0 / hz);
            y += v.y / hz;
        }
        st.h_speed()
    };
    assert!((run(30.0) - run(144.0)).abs() < 1.5);
}

#[test]
fn steering_cuts_the_thrust_near_the_target() {
    let (look, thrust) = steer(Vec3::ZERO, Vec3::new(15.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
    assert!(!thrust && look.x > 0.99);
    assert!(steer(Vec3::ZERO, Vec3::new(15.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0)).1);
}

fn run_search(world: &mut World, from: Vec3, to: Vec3) -> Result<Vec<Vec3>, FlyPlanError> {
    let mut s = FlightSearch::new(world, from, to, FlyConfig::default())?;
    loop {
        match s.step(world, 500) {
            FlyStatus::Running => {}
            FlyStatus::Done(p) => return Ok(p),
            FlyStatus::Failed(e) => return Err(e),
        }
    }
}

fn path_is_free(world: &mut World, p: &[Vec3]) -> bool {
    for w in p.windows(2) {
        let n = ((w[1] - w[0]).length() / 1.0).ceil() as i32;
        for k in 0..=n {
            let q = w[0] + (w[1] - w[0]) * (k as f32 / n.max(1) as f32);
            if world.blocked(q.x, q.z, q.y - 0.5, q.y + 2.5) {
                return false;
            }
        }
    }
    true
}

#[test]
fn open_air_is_one_straight_leg() {
    let mut w = World { boxes: vec![] };
    let p = run_search(&mut w, Vec3::new(0.0, 40.0, 0.0), Vec3::new(60.0, 40.0, 0.0)).unwrap();
    assert!(p.len() <= 3, "{} waypoints", p.len());
    assert!((p.last().unwrap().x - 60.0).abs() < 6.5);
}

#[test]
fn it_flies_over_or_around_a_tower() {
    let mut w = World { boxes: vec![B(20.0, 40.0, -20.0, 20.0, 0.0, 60.0)] };
    let p = run_search(&mut w, Vec3::new(0.0, 30.0, 0.0), Vec3::new(70.0, 30.0, 0.0)).unwrap();
    assert!(path_is_free(&mut w, &p), "{:?}", p);
    assert!(p.len() >= 3, "a bend is needed: {:?}", p);
}

#[test]
fn it_climbs_to_a_rooftop_goal() {
    let mut w = World { boxes: vec![B(30.0, 50.0, -10.0, 10.0, 0.0, 25.0)] };
    let p = run_search(&mut w, Vec3::new(0.0, 2.0, 0.0), Vec3::new(40.0, 26.0, 0.0)).unwrap();
    assert!(path_is_free(&mut w, &p));
    let e = p.last().unwrap();
    assert!((e - Vec3::new(40.0, 26.0, 0.0)).length() < 7.0, "{:?}", e);
}

#[test]
fn a_goal_inside_a_building_fails_cleanly() {
    let mut w = World { boxes: vec![B(30.0, 90.0, -40.0, 40.0, 0.0, 300.0)] };
    assert_eq!(run_search(&mut w, Vec3::new(0.0, 30.0, 0.0), Vec3::new(60.0, 30.0, 0.0)).err(), Some(FlyPlanError::NoGoal));
}

#[test]
fn a_closed_pocket_has_no_path_and_stops_at_the_budget() {
    // a wall of towers across the whole search width, taller than the planner's ceiling
    let mut w = World { boxes: vec![B(20.0, 28.0, -150.0, 150.0, 0.0, 400.0)] };
    let r = run_search(&mut w, Vec3::new(0.0, 30.0, 0.0), Vec3::new(60.0, 30.0, 0.0));
    assert!(matches!(r, Err(FlyPlanError::NoPath { .. })));
}

#[test]
fn the_sliced_search_gives_the_same_answer_for_any_budget() {
    let mk = || World { boxes: vec![B(20.0, 40.0, -20.0, 20.0, 0.0, 60.0)] };
    let (a, b) = (Vec3::new(0.0, 30.0, 0.0), Vec3::new(70.0, 30.0, 0.0));
    let mut w1 = mk();
    let mut w2 = mk();
    let mut s1 = FlightSearch::new(&mut w1, a, b, FlyConfig::default()).unwrap();
    let mut s2 = FlightSearch::new(&mut w2, a, b, FlyConfig::default()).unwrap();
    let r1 = loop { if let FlyStatus::Done(p) = s1.step(&mut w1, 7) { break p } };
    let r2 = loop { if let FlyStatus::Done(p) = s2.step(&mut w2, 100000) { break p } };
    assert_eq!(r1, r2);
}

#[test]
fn the_route_ends_on_the_goal() {
    let mut w = World { boxes: vec![B(30.0, 50.0, -10.0, 10.0, 0.0, 25.0)] };
    let g = Vec3::new(41.0, 25.5, 2.0);
    let p = run_search(&mut w, Vec3::new(0.0, 2.0, 0.0), g).unwrap();
    assert!((*p.last().unwrap() - (g + Vec3::Y)).length() < 1.0, "{:?}", p.last());
}

#[test]
fn a_hover_column_is_clear_only_where_nothing_stands_above() {
    let mut w = World { boxes: vec![B(20.0, 40.0, -20.0, 20.0, 0.0, 60.0)] };
    assert!(column_clear(&mut w, 0.0, 0.0, 0.0, 40.0));
    assert!(!column_clear(&mut w, 30.0, 0.0, 0.0, 40.0)); // inside the tower
    assert!(!column_clear(&mut w, 18.5, 0.0, 0.0, 40.0)); // beside it, inside the margin
    assert!(column_clear(&mut w, 30.0, 0.0, 61.0, 100.0)); // above its roof the column is clear
}
