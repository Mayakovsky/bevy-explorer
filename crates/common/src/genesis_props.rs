//! Unity's landscape props (rocks, bushes, trees) on empty parcels, read from the desktop client's own
//! `GenesisTrees.bin` (StreamingAssets, 303,020 six-byte instances). The same file works for any realm: an instance is dropped when it
//! overlaps an occupied parcel (`TreeData.OverlapsOccupiedParcel`). This is what players of the desktop app see, so the Synth's
//! colliders are built from it instead of Bevy's own procedural placement.
//!
//! File layout (little endian): `min_parcel x,y: i32`, `max_parcel x,y: i32`, `(max-min)` per-parcel start indices (`i32`, row major by
//! parcel y then x), `count: i32`, then `count` instances of 6 bytes: `prototype, x, z, rotation_y, scale_xz, scale_y` (each a byte, `TreeInstanceData` field order).

use bevy::math::{IVec2, Vec2, Vec3};

pub const PARCEL: f32 = 16.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    pub x: u8,
    pub z: u8,
    pub prototype: u8,
    pub scale_xz: u8,
    pub scale_y: u8,
    pub rotation_y: u8,
}

/// `TerrainData.treeAssets` order: 0 Bush01, 1 Bush02, 2-4 Stone01-03, 5-7 Tree01-03.
pub struct Prototype {
    pub name: &'static str,
    /// Placement radius (metres, before scale): an instance whose radius reaches an occupied parcel is dropped.
    pub radius: f32,
    /// `randomization.randomScale`: the instance scale lerps between these with the stored byte.
    pub scale: (f32, f32),
    pub collider: Collider,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collider {
    None,
    /// A Y-axis capsule of `radius` and total `height` at the prefab's child scale `(sx, sy, sz)`, centred on the instance (the rocks).
    Capsule { radius: f32, height: f32, child_scale: [f32; 3] },
    /// The prefab's collision children (trees: trunk and branch capsules plus crown spheres; Bush02: one stem capsule).
    Shapes(&'static [Shape]),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Capsule,
    Sphere,
}

/// One collider of a prefab, in the prefab's local space (Unity, left-handed): the child's transform and the collider's own parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub kind: Kind,
    pub pos: [f32; 3],
    /// quaternion x, y, z, w
    pub rot: [f32; 4],
    pub scale: [f32; 3],
    pub centre: [f32; 3],
    pub radius: f32,
    pub height: f32,
    /// capsule axis: 0 = X, 1 = Y, 2 = Z
    pub dir: u8,
}

mod shapes;
use shapes::{BUSH02_SHAPES, TREE01_SHAPES, TREE02_SHAPES, TREE03_SHAPES};

pub const PROTOTYPES: [Prototype; 8] = [
    Prototype { name: "Bush01", radius: 2.0, scale: (2.0, 3.0), collider: Collider::None },
    Prototype { name: "Bush02", radius: 2.0, scale: (2.0, 3.0), collider: Collider::Shapes(&BUSH02_SHAPES) },
    Prototype { name: "Stone01", radius: 1.0, scale: (1.0, 4.0), collider: Collider::Capsule { radius: 0.5, height: 2.0, child_scale: [1.71718, 1.01659, 1.7924199] } },
    Prototype { name: "Stone02", radius: 1.0, scale: (1.0, 4.0), collider: Collider::Capsule { radius: 0.5, height: 2.0, child_scale: [1.2314928, 0.96884936, 1.2854519] } },
    Prototype { name: "Stone03", radius: 1.0, scale: (1.0, 4.0), collider: Collider::Capsule { radius: 0.5, height: 2.0, child_scale: [1.5168672, 0.94172555, 1.0701506] } },
    Prototype { name: "Tree01", radius: 10.0, scale: (0.75, 1.25), collider: Collider::Shapes(&TREE01_SHAPES) },
    Prototype { name: "Tree02", radius: 10.0, scale: (0.75, 1.0), collider: Collider::Shapes(&TREE02_SHAPES) },
    Prototype { name: "Tree03", radius: 13.0, scale: (0.75, 1.0), collider: Collider::Shapes(&TREE03_SHAPES) },
];

pub struct GenesisProps {
    min: IVec2,
    max: IVec2,
    /// start index of each parcel's instances
    starts: Vec<i32>,
    instances: Vec<Instance>,
}

impl GenesisProps {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let int = |at: usize| -> Result<i32, String> {
            bytes.get(at..at + 4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| format!("truncated at {at}"))
        };
        let min = IVec2::new(int(0)?, int(4)?);
        let max = IVec2::new(int(8)?, int(12)?);
        let size = max - min;
        if size.x <= 0 || size.y <= 0 || size.x > 4096 || size.y > 4096 {
            return Err(format!("bad bounds {min} {max}"));
        }
        let parcels = (size.x * size.y) as usize;
        let mut starts = Vec::with_capacity(parcels);
        for i in 0..parcels {
            starts.push(int(16 + i * 4)?);
        }
        let count_at = 16 + parcels * 4;
        let count = int(count_at)? as usize;
        let data = bytes.get(count_at + 4..count_at + 4 + count * 6).ok_or("truncated instance data")?;
        let instances = data
            .chunks_exact(6)
            .map(|b| Instance { prototype: b[0], x: b[1], z: b[2], rotation_y: b[3], scale_xz: b[4], scale_y: b[5] })
            .collect();
        Ok(Self { min, max, starts, instances })
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
    pub fn bounds(&self) -> (IVec2, IVec2) {
        (self.min, self.max)
    }

    /// Instances stored for a parcel (empty outside the file's bounds).
    pub fn parcel(&self, parcel: IVec2) -> &[Instance] {
        if parcel.x < self.min.x || parcel.x >= self.max.x || parcel.y < self.min.y || parcel.y >= self.max.y {
            return &[];
        }
        let index = ((parcel.y - self.min.y) * (self.max.x - self.min.x) + parcel.x - self.min.x) as usize;
        let start = self.starts[index] as usize;
        let end = self.starts.get(index + 1).map(|v| *v as usize).unwrap_or(self.instances.len());
        &self.instances[start.min(self.instances.len())..end.min(self.instances.len())]
    }
}

/// A placed instance in Unity coordinates (x east, z north, y up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub prototype: usize,
    pub position_xz: Vec2,
    pub yaw_degrees: f32,
    /// `(sx, sy, sx)`: Unity's `LerpScale(...).xyx`.
    pub scale: Vec3,
}

/// `TreeData.GetTreeTransform` without the height: world x/z, yaw and scale, or `None` when the instance overlaps an occupied parcel.
/// `occupied(parcel)` is true for owned parcels and for anything outside the terrain bounds.
pub fn place(parcel: IVec2, i: &Instance, occupied: &impl Fn(IVec2) -> bool) -> Option<Placed> {
    let proto = PROTOTYPES.get(i.prototype as usize)?;
    let position_xz = Vec2::new((parcel.x as f32 + i.x as f32 * (1.0 / 255.0)) * PARCEL, (parcel.y as f32 + i.z as f32 * (1.0 / 255.0)) * PARCEL);
    let lerp = |t: f32| proto.scale.0 + (proto.scale.1 - proto.scale.0) * t;
    let scale = Vec3::new(lerp(i.scale_xz as f32 * (1.0 / 255.0)), lerp(i.scale_y as f32 * (1.0 / 255.0)), lerp(i.scale_xz as f32 * (1.0 / 255.0)));
    if overlaps_occupied(position_xz, proto.radius * scale.x, occupied) {
        return None;
    }
    Some(Placed { prototype: i.prototype as usize, position_xz, yaw_degrees: i.rotation_y as f32 * (360.0 / 255.0), scale })
}

/// `TreeData.OverlapsOccupiedParcel`, ported branch for branch (including its asymmetries).
pub fn overlaps_occupied(position: Vec2, radius: f32, occupied: &impl Fn(IVec2) -> bool) -> bool {
    let parcel = IVec2::new((position.x / PARCEL).floor() as i32, (position.y / PARCEL).floor() as i32);
    if occupied(parcel) {
        return true;
    }
    let local = position - Vec2::new(parcel.x as f32, parcel.y as f32) * PARCEL;
    let p = parcel;
    if local.x < radius {
        if occupied(IVec2::new(p.x - 1, p.y)) {
            return true;
        }
        if local.y < radius && occupied(IVec2::new(p.x - 1, p.y - 1)) {
            return true;
        }
    }
    if PARCEL - local.x < radius {
        if occupied(IVec2::new(p.x + 1, p.y)) {
            return true;
        }
        if PARCEL - local.y < radius && occupied(IVec2::new(p.x + 1, p.y + 1)) {
            return true;
        }
    }
    if local.y < radius {
        if occupied(IVec2::new(p.x, p.y - 1)) {
            return true;
        }
        if PARCEL - local.x < radius && occupied(IVec2::new(p.x + 1, p.y - 1)) {
            return true;
        }
    }
    if PARCEL - local.y < radius {
        if occupied(IVec2::new(p.x, p.y + 1)) {
            return true;
        }
        if local.x < radius && occupied(IVec2::new(p.x - 1, p.y + 1)) {
            return true;
        }
    }
    false
}

/// A collider in world space (Unity coordinates: x east, y up, z north).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorldShape {
    /// Centre line from `a` to `b`, spheres of `radius` at both ends.
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    Sphere { centre: Vec3, radius: f32 },
}

fn rotate(q: [f32; 4], v: Vec3) -> Vec3 {
    // v' = q v q^-1 (Unity's quaternion maths is the standard formula in its own axes)
    let u = Vec3::new(q[0], q[1], q[2]);
    let w = q[3];
    2.0 * u.dot(v) * u + (w * w - u.dot(u)) * v + 2.0 * w * u.cross(v)
}

/// The prefab's colliders placed for an instance: `M_instance * M_child`, with the instance transform = yaw about Y (Unity's sense: x turns
/// toward -z), scale `(sx, sy, sx)`, translation at the ground height. Radii scale by the largest axis scale perpendicular to the capsule
/// (spheres: the largest), lengths along the axis by the axis scale.
pub fn world_shapes(placed: &Placed, ground_y: f32) -> Vec<WorldShape> {
    let Some(proto) = PROTOTYPES.get(placed.prototype) else {
        return Vec::new();
    };
    let single;
    let shapes: &[Shape] = match proto.collider {
        Collider::None => return Vec::new(),
        Collider::Shapes(list) => list,
        Collider::Capsule { radius, height, child_scale } => {
            single = [Shape { kind: Kind::Capsule, pos: [0.0; 3], rot: [0.0, 0.0, 0.0, 1.0], scale: child_scale, centre: [0.0; 3], radius, height, dir: 1 }];
            &single
        }
    };
    let (sin, cos) = placed.yaw_degrees.to_radians().sin_cos();
    let origin = Vec3::new(placed.position_xz.x, ground_y, placed.position_xz.y);
    let to_world = |local: Vec3| {
        let s = local * placed.scale;
        origin + Vec3::new(s.x * cos + s.z * sin, s.y, -s.x * sin + s.z * cos)
    };
    let mut out = Vec::new();
    for sh in shapes {
        let child_scale = Vec3::from(sh.scale);
        let total = child_scale * placed.scale; // lossy scale, ignoring the child's own rotation
        let at = |local: Vec3| to_world(Vec3::from(sh.pos) + rotate(sh.rot, local * child_scale));
        let centre = Vec3::from(sh.centre);
        match sh.kind {
            Kind::Sphere => out.push(WorldShape::Sphere { centre: at(centre), radius: sh.radius * total.abs().max_element() }),
            Kind::Capsule => {
                let axis = match sh.dir {
                    0 => Vec3::X,
                    1 => Vec3::Y,
                    _ => Vec3::Z,
                };
                let perp = match sh.dir {
                    0 => total.y.abs().max(total.z.abs()),
                    1 => total.x.abs().max(total.z.abs()),
                    _ => total.x.abs().max(total.y.abs()),
                };
                let r = sh.radius * perp;
                // endpoints at +-(height/2 - radius) along the axis in the collider's own frame, carried through the child and instance transforms
                let local_half = (sh.height * 0.5 - sh.radius).max(0.0);
                let (a, b) = (at(centre - axis * local_half), at(centre + axis * local_half));
                out.push(WorldShape::Capsule { a, b, radius: r });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
