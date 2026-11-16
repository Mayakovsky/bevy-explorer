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

fn placed(prototype: usize, scale: f32) -> Placed {
    Placed { prototype, position_xz: Vec2::new(100.0, 50.0), yaw_degrees: 0.0, scale: Vec3::splat(scale) }
}

#[test]
fn rocks_have_capsules_that_grow_with_the_instance_scale() {
    let small = world_shapes(&placed(2, 1.0), 10.0);
    let big = world_shapes(&placed(2, 4.0), 10.0);
    let (WorldShape::Capsule { a: a0, b: b0, radius: r0 }, WorldShape::Capsule { radius: r1, .. }) = (small[0], big[0]) else { panic!() };
    assert!(r1 > r0 * 3.9 && r1 < r0 * 4.1);
    assert!(((a0.y + b0.y) * 0.5 - 10.0).abs() < 1e-4); // centred on the ground height, like the prefab's capsule
    assert!((r0 - 0.5 * 1.7924199).abs() < 1e-4); // Stone01: radius 0.5 x the larger of the child's x and z scale
}

#[test]
fn bush01_has_no_collider_trees_and_bush02_do() {
    assert!(world_shapes(&placed(0, 1.0), 0.0).is_empty()); // Bush01
    assert_eq!(world_shapes(&placed(1, 1.0), 0.0).len(), 1); // Bush02: one stem capsule
    assert_eq!(world_shapes(&placed(5, 1.0), 0.0).len(), 4); // Tree01: 3 capsules + the crown sphere
    assert_eq!(world_shapes(&placed(6, 1.0), 0.0).len(), 5);
    assert_eq!(world_shapes(&placed(7, 1.0), 0.0).len(), 8);
}

#[test]
fn a_tree_trunk_starts_at_the_ground_and_the_crown_is_high() {
    let shapes = world_shapes(&placed(5, 1.0), 0.0);
    // trunk: the first capsule starts near the instance origin (child y 0.758, horizontal axis X rotated by the child's quaternion to vertical)
    let WorldShape::Capsule { a, b, radius } = shapes[0] else { panic!("first shape is the trunk capsule") };
    assert!((radius - 0.4907274).abs() < 1e-4);
    assert!((b - a).length() > 1.5 && (b - a).y.abs() > 1.5, "the trunk segment is vertical: {a} {b}");
    assert!(a.x > 99.0 && a.x < 101.0 && a.z > 49.0 && a.z < 51.0);
    let crown = shapes.iter().find_map(|s| if let WorldShape::Sphere { centre, radius } = s { Some((*centre, *radius)) } else { None }).unwrap();
    assert!(crown.0.y > 8.0 && (crown.1 - 5.3309493).abs() < 1e-4, "crown at {:?}", crown);
}

#[test]
fn instance_scale_and_yaw_carry_through_the_child_transform() {
    let base = world_shapes(&placed(5, 1.0), 0.0);
    let big = world_shapes(&placed(5, 2.0), 0.0);
    let (WorldShape::Sphere { centre: c1, radius: r1 }, WorldShape::Sphere { centre: c2, radius: r2 }) = (
        base.iter().find(|s| matches!(s, WorldShape::Sphere { .. })).copied().unwrap(),
        big.iter().find(|s| matches!(s, WorldShape::Sphere { .. })).copied().unwrap(),
    ) else { panic!() };
    assert!((r2 / r1 - 2.0).abs() < 1e-4);
    assert!((c2.y / c1.y - 2.0).abs() < 1e-4); // the crown height doubles
    // a 90 degree yaw turns the crown's horizontal offset (Unity: x toward -z)
    let mut turned = placed(5, 1.0);
    turned.yaw_degrees = 90.0;
    let WorldShape::Sphere { centre: c3, .. } = world_shapes(&turned, 0.0).into_iter().find(|s| matches!(s, WorldShape::Sphere { .. })).unwrap() else { panic!() };
    let (dx, dz) = (c1.x - 100.0, c1.z - 50.0);
    assert!((c3.x - (100.0 + dz)).abs() < 1e-3 && (c3.z - (50.0 - dx)).abs() < 1e-3, "{c3} vs d=({dx},{dz})");
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
