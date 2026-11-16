use super::*;
use crate::terrain;

fn block(min: (i32, i32), max: (i32, i32)) -> HashSet<IVec2> {
    let mut s = HashSet::new();
    for x in min.0..=max.0 {
        for y in min.1..=max.1 {
            s.insert(IVec2::new(x, y));
        }
    }
    s
}

/// two blocks 40 parcels apart: the empty land between them rises away from both
fn two_blocks() -> HashSet<IVec2> {
    let mut a = block((-30, -2), (-26, 2));
    a.extend(block((26, -2), (30, 2)));
    a
}

#[test]
fn empty_set_has_no_terrain() {
    assert!(UnityTerrain::build(&HashSet::new()).is_none());
}

#[test]
fn bounds_are_the_owned_extent_plus_padding() {
    let t = UnityTerrain::build(&two_blocks()).unwrap();
    assert_eq!(t.min_parcel, IVec2::new(-32, -4));
    assert_eq!(t.max_parcel, IVec2::new(32, 4));
    assert!(t.is_occupied(IVec2::new(-28, 0))); // owned
    assert!(t.is_occupied(IVec2::new(40, 0))); // outside the bounds counts as occupied
    assert!(!t.is_occupied(IVec2::new(0, 0))); // free land between
}

#[test]
fn ground_is_flat_on_owned_parcels_and_rises_away_from_them() {
    let t = UnityTerrain::build(&two_blocks()).unwrap();
    assert_eq!(t.height(-28.0 * 16.0 + 8.0, 8.0), 0.0); // inside an owned parcel
    let near = t.height(-24.0 * 16.0 + 8.0, 8.0); // 2 parcels from the block
    let far = t.height(0.0, 8.0); // the middle of the gap
    assert!(far > near, "far {far} near {near}");
    assert!(t.max_steps > 0);
    assert!(far <= t.max_steps as f32 * STEP_HEIGHT + 3.0 * 3.0 + 0.01); // steps plus the 3-octave noise (|noise| <= ~3.9)
}

#[test]
fn heights_match_the_existing_capped_port_when_the_step_count_is_small() {
    // with at most 25 steps the texture step is 10, which is exactly what `terrain::height` assumes
    let t = UnityTerrain::build(&two_blocks()).unwrap();
    assert!(t.max_steps <= 25);
    let texel = |parcel: IVec2| t.texel(parcel.x + t.size / 2, parcel.y + t.size / 2);
    let uv_size = t.size as f32;
    for (x, z) in [(0.0, 8.0), (-200.0, 20.0), (123.4, -45.6), (300.0, 30.0), (-300.0, 3.0), (10.0, 60.0)] {
        let ours = t.height(x, z);
        let theirs = terrain::height(x, z, t.max_steps as f32, uv_size, texel);
        assert!((ours - theirs).abs() < 1e-3, "({x},{z}) ours {ours} theirs {theirs}");
    }
}

#[test]
fn a_single_deployed_block_still_has_a_ring_of_padding_that_slopes_up() {
    let t = UnityTerrain::build(&block((0, 0), (2, 2))).unwrap();
    // padding ring parcels are free; the chamfer treats the outside as black, so the corner of the terrain is low
    assert!(!t.is_occupied(IVec2::new(-1, -1)));
    assert!(t.height(-8.0, -8.0) < 3.5);
}

#[test]
fn heights_never_go_negative_and_are_finite() {
    let t = UnityTerrain::build(&two_blocks()).unwrap();
    for i in -60..60 {
        for j in -10..10 {
            let h = t.height(i as f32 * 9.7, j as f32 * 13.1);
            assert!(h.is_finite() && h >= 0.0);
        }
    }
}
