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
    /// A Y-axis capsule of `radius` and total `height` at the prefab's child scale `(sx, sy, sz)`, centred on the instance.
    Capsule { radius: f32, height: f32, child_scale: [f32; 3] },
    /// Prefab read but not yet ported (tree and Bush02 trunks).
    Unported,
}

pub const PROTOTYPES: [Prototype; 8] = [
    Prototype { name: "Bush01", radius: 2.0, scale: (2.0, 3.0), collider: Collider::None },
    Prototype { name: "Bush02", radius: 2.0, scale: (2.0, 3.0), collider: Collider::Unported },
    Prototype { name: "Stone01", radius: 1.0, scale: (1.0, 4.0), collider: Collider::Capsule { radius: 0.5, height: 2.0, child_scale: [1.71718, 1.01659, 1.7924199] } },
    Prototype { name: "Stone02", radius: 1.0, scale: (1.0, 4.0), collider: Collider::Capsule { radius: 0.5, height: 2.0, child_scale: [1.2314928, 0.96884936, 1.2854519] } },
    Prototype { name: "Stone03", radius: 1.0, scale: (1.0, 4.0), collider: Collider::Capsule { radius: 0.5, height: 2.0, child_scale: [1.5168672, 0.94172555, 1.0701506] } },
    Prototype { name: "Tree01", radius: 10.0, scale: (0.75, 1.25), collider: Collider::Unported },
    Prototype { name: "Tree02", radius: 10.0, scale: (0.75, 1.0), collider: Collider::Unported },
    Prototype { name: "Tree03", radius: 13.0, scale: (0.75, 1.0), collider: Collider::Unported },
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

/// A rock's collision capsule in world space (Unity coordinates): its centre line runs from `base` to `top` (spheres of `radius` at both ends).
/// Unity scales a capsule's radius by max(|x|, |z|) of the lossy scale and its height by |y|; the instance scale multiplies the prefab's child scale.
pub fn rock_capsule(placed: &Placed, ground_y: f32) -> Option<(Vec3, Vec3, f32)> {
    let Collider::Capsule { radius, height, child_scale } = PROTOTYPES.get(placed.prototype)?.collider else {
        return None;
    };
    let s = Vec3::new(child_scale[0], child_scale[1], child_scale[2]) * placed.scale;
    let r = radius * s.x.abs().max(s.z.abs());
    let h = (height * s.y.abs()).max(2.0 * r);
    // the prefab's CapsuleCollider is centred on its child at the instance origin (centre 0,0,0): the capsule straddles the ground height
    let centre = Vec3::new(placed.position_xz.x, ground_y, placed.position_xz.y);
    let half = (h * 0.5 - r).max(0.0);
    Some((centre - Vec3::Y * half, centre + Vec3::Y * half, r))
}

#[cfg(test)]
mod tests;
