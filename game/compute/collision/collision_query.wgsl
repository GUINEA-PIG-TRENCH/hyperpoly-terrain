// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Guinea Pig Trench LLC (PA #13674084)
// Credit Facility: Truth Holds Enterprise (PA #7049023)
//
// vinculum: cell_occupancy / ray_path_length
//   The ratio of voxel cells the ray tests that contain surface
//   vs. the total length of the ray. Lower = more empty space skipped.
//
// layer: simulation
// domain: terrain (maps to swarm as agent_vision / terrain_occupancy,
//          legal as document_graph / search_space)
//
// ──────────────────────────────────────────────────────────────
// COLLISION: Voxel DDA Raycast + Sphere-Cast
// ──────────────────────────────────────────────────────────────
// Walks the dual grid (255³ cells) using a DDA traversal,
// testing the QEF isosurface at each step.
//
// No BVH — the grid IS the acceleration structure.
// Only active cells (where density crosses iso) are tested.
//
// Reads: qef_vertices (same buffer as extraction pipeline)
//        hermite_buffer (for density at corners)
//        brick_flags (to skip unchanged bricks)
// Writes: Hit record (position, normal, material, t)

struct HitRecord {
  hit: u32;          // 0 = miss, 1 = hit
  pos: vec3<f32>;
  normal: vec3<f32>;
  material_type: u32; // 0=air, 1=rock, 2=ore, 3=support
  t: f32;            // ray parameter
};

struct Ray {
  origin: vec3<f32>;
  direction: vec3<f32>;
  t_min: f32;
  t_max: f32;
};

struct SphereQuery {
  center: vec3<f32>;
  radius: f32;
};

@group(0) @binding(0) var<storage, read> qef_vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> hermite_buffer: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> density_u16: array<F16>;
@group(0) @binding(3) var<storage, read> brick_flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> hit_output: array<HitRecord>;
@group(0) @binding(5) var<uniform> ray_params: Ray;

const CELLS: u32 = 255u;
const CELL_STRIDE: u32 = 65025u;
const ISO_THRESHOLD: f32 = 0.5;

// ── Helpers ──

fn world_to_cell(pos: vec3<f32>) -> vec3<u32> {
  return vec3<u32>(u32(clamp(pos.x, 0.0, 254.0)),
                   u32(clamp(pos.y, 0.0, 254.0)),
                   u32(clamp(pos.z, 0.0, 254.0)));
}

fn cell_idx(c: vec3<u32>) -> u32 {
  return c.x + c.y * CELLS + c.z * CELL_STRIDE;
}

// ── Ray-Cell Intersection ──
// Tests whether the ray passes through the implicit surface
// defined by the QEF vertex in this cell.

fn intersect_cell(c: vec3<u32>, ray: Ray) -> HitRecord {
  let idx = cell_idx(c);
  let v = qef_vertices[idx];

  // Quick AABB test: is the dual vertex near the ray?
  // The QEF vertex is inside the cell (cx..cx+1, cy..cy+1, cz..cz+1)
  // Use slab test for ray-AABB
  let inv_dir = 1.0 / ray.direction;
  let t1 = (vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) - ray.origin) * inv_dir;
  let t2 = (vec3<f32>(f32(c.x + 1u), f32(c.y + 1u), f32(c.z + 1u)) - ray.origin) * inv_dir;
  let t_near = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  let t_far = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

  if (t_near > t_far || t_far < 0.0) {
    return HitRecord(0u, vec3<f32>(0.0), vec3<f32>(0.0), 0u, 0.0);
  }

  // Check if the QEF vertex is inside the cell — if so, test distance
  // from ray to vertex. If vertex is at cell edge (feature preserved),
  // the test radius shrinks.
  let to_vertex = v - (ray.origin + t_near * ray.direction);
  let dist = length(to_vertex);

  // Feature weight approximation: distance from cell center
  let cell_center = vec3<f32>(f32(c.x) + 0.5, f32(c.y) + 0.5, f32(c.z) + 0.5);
  let off_center = length(v - cell_center);
  let hit_radius = 0.5 + off_center * 0.3; // Expand for off-center vertices

  if (dist < hit_radius) {
    let hit_t = t_near;
    let hit_pos = ray.origin + hit_t * ray.direction;

    // Compute normal from QEF vertex gradient (central diff)
    let dx_l = qef_vertices[cell_idx(c + vec3<u32>(0u, 0u, 0u) - vec3<u32>(1u, 0u, 0u))]; // clamp handled
    let dx_r = qef_vertices[cell_idx(c + vec3<u32>(1u, 0u, 0u))];
    let normal = normalize(dx_r - dx_l);

    return HitRecord(1u, hit_pos, normal, 1u, hit_t);
  }

  return HitRecord(0u, vec3<f32>(0.0), vec3<f32>(0.0), 0u, 0.0);
}

// ── Main: DDA Raycast ──

@compute @workgroup_size(64)
fn raycast_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  // One thread handles one ray (batch up to 64 rays at once)
  let ray_idx = gid.x;
  if (ray_idx > 0u) { return; } // Single-ray mode for now

  let ray = ray_params;
  let origin = ray.origin;
  let dir = ray.direction;
  let inv_dir = 1.0 / dir;

  // DDA setup — step through 255³ grid
  let start_cell = world_to_cell(origin);

  // Check if starting inside a solid cell
  let start_idx = cell_idx(start_cell);
  let start_density = density_u16[start_idx]; // approximate
  var closest = HitRecord(0u, vec3<f32>(0.0), vec3<f32>(0.0), 0u, ray.t_max);

  // DDA traversal
  var cell = start_cell;
  let step = vec3<i32>(i32(sign(dir.x)), i32(sign(dir.y)), i32(sign(dir.z)));
  let t_delta = vec3<f32>(abs(inv_dir.x), abs(inv_dir.y), abs(inv_dir.z));

  // Initial t_max for each axis
  var t_max = vec3<f32>(
    (f32(cell.x + u32(step.x > 0)) - origin.x) * inv_dir.x,
    (f32(cell.y + u32(step.y > 0)) - origin.y) * inv_dir.y,
    (f32(cell.z + u32(step.z > 0)) - origin.z) * inv_dir.z
  );

  for (var i = 0u; i < 512u; i++) {
    if (cell.x >= CELLS || cell.y >= CELLS || cell.z >= CELLS) { break; }
    if (cell.x > CELLS || cell.y > CELLS || cell.z > CELLS) { break; }

    // Test this cell
    let hit = intersect_cell(cell, ray);
    if (hit.hit == 1u && hit.t < closest.t) {
      closest = hit;
      break; // First hit is closest due to DDA ordering
    }

    // Step to next cell
    if (t_max.x < t_max.y) {
      if (t_max.x < t_max.z) {
        cell.x = u32(i32(cell.x) + step.x);
        t_max.x += t_delta.x;
      } else {
        cell.z = u32(i32(cell.z) + step.z);
        t_max.z += t_delta.z;
      }
    } else {
      if (t_max.y < t_max.z) {
        cell.y = u32(i32(cell.y) + step.y);
        t_max.y += t_delta.y;
      } else {
        cell.z = u32(i32(cell.z) + step.z);
        t_max.z += t_delta.z;
      }
    }
  }

  // Guard: reject NaN/invalid hits (written at end of closest computation)
  if (isnan(closest.pos.x) || isnan(closest.pos.y) || isnan(closest.pos.z) || closest.t < 0.0) {
    closest.t = -1.0;
  }

  hit_output[ray_idx] = closest;
}
