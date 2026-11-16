//! Unity's empty-parcel ground height from a snapshot of occupied parcels: `TerrainGenerator.CreateOccupancyMap` +
//! `WriteInteriorChamferOnWhite` (3-4 chamfer distance field, realm-wide step count) + `GetParcelNoiseHeight`. Unlike
//! `terrain::TerrainTargets` (which caps the step count so heights only depend on a local neighbourhood), this follows Unity exactly:
//! the heights a desktop player stands on. Coordinates are Unity's: x east, z north (negate Bevy z); parcel (x, y) covers z in
//! `[16y, 16y + 16)`.

use bevy::math::IVec2;
use std::collections::HashSet;

use crate::terrain::mountain_noise;

pub const PARCEL: i32 = 16;
pub const STEP_HEIGHT: f32 = 1.8; // TerrainData.stepHeight
pub const BORDER_PADDING: i32 = 2; // TerrainData.borderPadding
const INF: i32 = 1 << 28;
const ORTH: i32 = 3;
const DIAG: i32 = 4;

pub struct UnityTerrain {
    /// Occupancy texture, row major, `size` x `size`, texel (x, y) = parcel (x - half, y - half).
    texels: Vec<u8>,
    size: i32,
    /// `OccupancyFloor` (0..255) and `MaxHeight` (steps * 1.8).
    floor: i32,
    pub max_steps: i32,
    max_height: f32,
    /// Terrain bounds in parcels (inclusive), padding included.
    pub min_parcel: IVec2,
    pub max_parcel: IVec2,
}

impl UnityTerrain {
    pub fn build(owned: &HashSet<IVec2>) -> Option<Self> {
        if owned.is_empty() {
            return None;
        }
        // TerrainModel: bounds of the owned parcels plus the padding, centred as Unity does it
        let min_o = owned.iter().fold(IVec2::MAX, |a, p| a.min(*p));
        let max_o = owned.iter().fold(IVec2::MIN, |a, p| a.max(*p));
        let size_in_parcels = max_o - min_o + IVec2::ONE;
        let with_padding = size_in_parcels + IVec2::splat(2 * BORDER_PADDING);
        let centre = min_o + size_in_parcels / 2;
        let min_parcel = centre - with_padding / 2;
        let max_parcel = min_parcel + (with_padding - IVec2::ONE);

        let abs_max = [min_parcel.x, min_parcel.y, max_parcel.x, max_parcel.y].iter().map(|v| v.abs()).max().unwrap();
        let size = ((abs_max * 2 + 2) as u32).next_power_of_two() as i32;
        let half = size / 2;
        let mut data = vec![0u8; (size * size) as usize];
        // free area: full-width white rows, trimmed by one black column either side
        let free_min = IVec2::new((min_parcel.x).max(-half), (min_parcel.y).max(-half));
        let free_max = IVec2::new((max_parcel.x).min(half - 1), (max_parcel.y).min(half - 1));
        let start_row = half + free_min.y;
        let end_row = half + free_max.y;
        if end_row >= start_row {
            for row in start_row..=end_row {
                let r = (row * size) as usize;
                data[r..r + size as usize].fill(255);
            }
            let (left, right) = (half + free_min.x - 1, half + free_max.x + 1);
            for row in start_row..=end_row {
                for col in [left, right] {
                    if (0..size).contains(&col) {
                        data[(row * size + col) as usize] = 0;
                    }
                }
            }
        }
        for p in owned {
            let (tx, ty) = (p.x + half, p.y + half);
            let idx = ty * size + tx;
            if idx >= 0 && (idx as usize) < data.len() && (0..size).contains(&tx) {
                data[idx as usize] = 0;
            }
        }
        // WriteInteriorChamferOnWhite: working area = the terrain bounds
        let min_x = (min_parcel.x + half).max(0);
        let min_y = (min_parcel.y + half).max(0);
        let max_x = (max_parcel.x + half).min(size - 1);
        let max_y = (max_parcel.y + half).min(size - 1);
        let mut dist = vec![0i32; data.len()];
        let at = |x: i32, y: i32| (y * size + x) as usize;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if data[at(x, y)] > 0 {
                    dist[at(x, y)] = INF;
                }
            }
        }
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let i = at(x, y);
                let mut d = dist[i];
                if d != 0 {
                    d = d.min(if x > min_x { dist[at(x - 1, y)] + ORTH } else { ORTH });
                    d = d.min(if y > min_y { dist[at(x, y - 1)] + ORTH } else { ORTH });
                    d = d.min(if x > min_x && y > min_y { dist[at(x - 1, y - 1)] + DIAG } else { DIAG });
                    d = d.min(if x < max_x && y > min_y { dist[at(x + 1, y - 1)] + DIAG } else { DIAG });
                    dist[i] = d;
                }
            }
        }
        let mut max_d = 0;
        for y in (min_y..=max_y).rev() {
            for x in (min_x..=max_x).rev() {
                let i = at(x, y);
                let mut d = dist[i];
                if d != 0 {
                    d = d.min(if x < max_x { dist[at(x + 1, y)] + ORTH } else { ORTH });
                    d = d.min(if y < max_y { dist[at(x, y + 1)] + ORTH } else { ORTH });
                    d = d.min(if x < max_x && y < max_y { dist[at(x + 1, y + 1)] + DIAG } else { DIAG });
                    d = d.min(if x > min_x && y < max_y { dist[at(x - 1, y + 1)] + DIAG } else { DIAG });
                    dist[i] = d;
                }
                if data[i] != 0 && dist[i] < INF && dist[i] > max_d {
                    max_d = dist[i];
                }
            }
        }
        // ApplyDistanceFieldMapping
        let mut max_pixel = max_d / ORTH - 1;
        let (floor, max_steps) = if max_d == 0 || max_pixel <= 0 {
            (255, 0)
        } else {
            if max_pixel > 255 {
                max_pixel = 255;
            }
            let step_size = if max_pixel <= 25 { 10 } else { ((255 - 1) / max_pixel).max(1) };
            let min_value = 255 - step_size * max_pixel;
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let i = at(x, y);
                    if data[i] == 0 {
                        continue;
                    }
                    let pixel = ((dist[i] / ORTH) - 1).min(max_pixel);
                    data[i] = (min_value + pixel * step_size).clamp(min_value, 255) as u8;
                }
            }
            (min_value, max_pixel)
        };
        Some(Self { texels: data, size, floor, max_steps, max_height: max_steps as f32 * STEP_HEIGHT, min_parcel, max_parcel })
    }

    /// Texel value of a parcel (Unity's clamped texture reads).
    fn texel(&self, x: i32, y: i32) -> f32 {
        f32::from(self.texels[(y.clamp(0, self.size - 1) * self.size + x.clamp(0, self.size - 1)) as usize])
    }

    /// `IsParcelOccupied` as `TreeData` uses it: black texel, or outside the terrain bounds.
    pub fn is_occupied(&self, parcel: IVec2) -> bool {
        if parcel.x < self.min_parcel.x || parcel.y < self.min_parcel.y || parcel.x > self.max_parcel.x || parcel.y > self.max_parcel.y {
            return true;
        }
        let half = self.size / 2;
        self.texels[((parcel.y + half) * self.size + parcel.x + half) as usize] == 0
    }

    /// `GetParcelNoiseHeight(x, z)` in Unity coordinates.
    pub fn height(&self, x: f32, unity_z: f32) -> f32 {
        let size = self.size as f32;
        let uv = [(x / PARCEL as f32 + size * 0.5) / size, (unity_z / PARCEL as f32 + size * 0.5) / size];
        let pixel = uv.map(|v| v * size - 0.5);
        let base = pixel.map(|v| v.floor() as i32);
        let f = [pixel[0] - pixel[0].floor(), pixel[1] - pixel[1].floor()];
        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
        // SampleBilinearClamp: `top` is row y, `bottom` is row y + 1 (Unity's index vector), then lerp by the fractional y
        let top = lerp(self.texel(base[0], base[1]), self.texel(base[0] + 1, base[1]), f[0]);
        let bottom = lerp(self.texel(base[0], base[1] + 1), self.texel(base[0] + 1, base[1] + 1), f[0]);
        let occupancy = lerp(top, bottom, f[1]) * (1.0 / 255.0);
        let min_value = self.floor as f32 / 255.0;
        if occupancy <= min_value {
            return 0.0;
        }
        let normalized = (occupancy - min_value) / (1.0 - min_value);
        let y = normalized * self.max_height + mountain_noise(x, unity_z) * (normalized * 20.0).clamp(0.0, 1.0);
        y.max(0.0)
    }
}

#[cfg(test)]
mod tests;
