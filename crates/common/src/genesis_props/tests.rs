use super::*;

fn file() -> Option<Vec<u8>> {
    // the real file is not committed (2.2 MB, Git LFS upstream): EMANATE_GENESIS_TREES or ~/emanate-data/GenesisTrees.bin
    let path = std::env::var("EMANATE_GENESIS_TREES").unwrap_or_else(|_| format!("{}/emanate-data/GenesisTrees.bin", std::env::var("HOME").unwrap_or_default()));
    std::fs::read(path).ok()
}

fn synthetic() -> Vec<u8> {
    // bounds (0,0)..(2,1): two parcels; parcel 0 has 2 instances, parcel 1 has 1
    let mut b = Vec::new();
    for v in [0i32, 0, 2, 1, 0, 2, 3] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&[2, 128, 64, 255, 0, 255]); // prototype, x, z, rotation, scale_xz, scale_y
    b.extend_from_slice(&[7, 0, 0, 0, 10, 10]);
    b.extend_from_slice(&[4, 255, 255, 51, 255, 0]);
    b
}

#[test]
fn parses_the_layout() {
    let g = GenesisProps::parse(&synthetic()).unwrap();
    assert_eq!(g.len(), 3);
    assert_eq!(g.parcel(IVec2::new(0, 0)).len(), 2);
    assert_eq!(g.parcel(IVec2::new(1, 0)).len(), 1);
    assert_eq!(g.parcel(IVec2::new(1, 0))[0], Instance { x: 255, z: 255, prototype: 4, scale_xz: 255, scale_y: 0, rotation_y: 51 });
    assert!(g.parcel(IVec2::new(5, 5)).is_empty()); // outside the file's bounds
}

#[test]
fn rejects_truncated_data() {
    let b = synthetic();
    assert!(GenesisProps::parse(&b[..b.len() - 1]).is_err());
    assert!(GenesisProps::parse(&b[..10]).is_err());
}

#[test]
fn stone_instance_position_scale_and_rotation_match_unity() {
    let g = GenesisProps::parse(&synthetic()).unwrap();
    let i = g.parcel(IVec2::new(0, 0))[0];
    let p = place(IVec2::new(0, 0), &i, &|_| false).unwrap();
    assert!((p.position_xz.x - 128.0 / 255.0 * 16.0).abs() < 1e-4);
    assert!((p.position_xz.y - 64.0 / 255.0 * 16.0).abs() < 1e-4);
    assert!((p.scale.x - 1.0).abs() < 1e-5 && (p.scale.y - 4.0).abs() < 1e-5 && p.scale.x == p.scale.z); // Stone01: lerp(1,4,0)=1 xz, lerp(1,4,1)=4 y
    assert_eq!(p.yaw_degrees, 255.0 * (360.0 / 255.0));
}

#[test]
fn an_instance_in_an_occupied_parcel_is_dropped() {
    let i = Instance { x: 128, z: 128, prototype: 2, scale_xz: 0, scale_y: 0, rotation_y: 0 };
    assert!(place(IVec2::new(3, 3), &i, &|p| p == IVec2::new(3, 3)).is_none());
    assert!(place(IVec2::new(3, 3), &i, &|_| false).is_some());
}

#[test]
fn an_instance_whose_radius_reaches_an_occupied_neighbour_is_dropped() {
    // x = 0 byte: on the west edge of its parcel; Stone radius 1 m at scale 1 reaches the parcel to the west
    let i = Instance { x: 0, z: 128, prototype: 2, scale_xz: 0, scale_y: 0, rotation_y: 0 };
    assert!(place(IVec2::new(3, 3), &i, &|p| p == IVec2::new(2, 3)).is_none());
    // a few metres in: clear of it
    let j = Instance { x: 80, z: 128, ..i };
    assert!(place(IVec2::new(3, 3), &j, &|p| p == IVec2::new(2, 3)).is_some());
}

#[test]
fn out_of_bounds_counts_as_occupied_through_the_callback() {
    let i = Instance { x: 128, z: 128, prototype: 2, scale_xz: 0, scale_y: 0, rotation_y: 0 };
    assert!(place(IVec2::new(0, 0), &i, &|p| p.x < 0 || p.y < 0).is_some());
    assert!(place(IVec2::new(-1, 0), &i, &|p| p.x < 0 || p.y < 0).is_none());
}

#[test]
fn rocks_have_capsules_that_grow_with_the_instance_scale_and_bushes_and_trees_do_not_yet() {
    let small = Placed { prototype: 2, position_xz: Vec2::ZERO, yaw_degrees: 0.0, scale: Vec3::new(1.0, 1.0, 1.0) };
    let big = Placed { scale: Vec3::new(4.0, 4.0, 4.0), ..small };
    let (b0, t0, r0) = rock_capsule(&small, 10.0).unwrap();
    let (b1, t1, r1) = rock_capsule(&big, 10.0).unwrap();
    assert!(r1 > r0 * 3.9 && r1 < r0 * 4.1);
    assert!(t1.y - b1.y >= t0.y - b0.y);
    assert!(((b0.y + t0.y) * 0.5 - 10.0).abs() < 1e-4); // centred on the ground height, like the prefab's capsule
    assert!(rock_capsule(&Placed { prototype: 0, ..small }, 0.0).is_none()); // Bush01: no collider
    assert!(rock_capsule(&Placed { prototype: 5, ..small }, 0.0).is_none()); // Tree01: not ported yet
}

#[test]
fn the_real_file_parses_and_matches_the_known_totals() {
    let Some(bytes) = file() else { eprintln!("GenesisTrees.bin not present: skipped"); return };
    let g = GenesisProps::parse(&bytes).unwrap();
    assert_eq!(g.bounds(), (IVec2::new(-152, -152), IVec2::new(166, 160)));
    assert_eq!(g.len(), 303_020);
    // every instance of every parcel adds up to the total, and every prototype index is one of the eight
    let (min, max) = g.bounds();
    let mut total = 0;
    let mut stones = 0;
    for y in min.y..max.y {
        for x in min.x..max.x {
            for i in g.parcel(IVec2::new(x, y)) {
                assert!((i.prototype as usize) < PROTOTYPES.len());
                if (2..=4).contains(&i.prototype) {
                    stones += 1;
                }
                total += 1;
            }
        }
    }
    assert_eq!(total, 303_020);
    eprintln!("stones in the file: {stones}");
    assert!(stones > 0);
}
