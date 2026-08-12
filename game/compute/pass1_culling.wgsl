// vinculum: culling_efficiency / visible_fraction
//   Ratio of active brick cells vs total grid cells after visibility culling.
//   Drives downstream dispatch sizing.
//   Also maps to swarm as search_coverage / explored_fraction,
//   and evacuation as safe_zone_ratio / total_area.
// vinculum: depth_resolution / subvolume_span
//   Ratio of leaf-node octree cells vs parent-cell spans at each LOD level.
//   Enables spatial-indexed dispatch: only bricks within frustum+LOD range execute.
// layer: compute
// domain: terrain
//
struct BrickMeta {
  min: f32, scale: f32, moisture: f32, stability: f32,
  lod: u32, // 0=leaf (16³), 1=parent (32³), 2=grandparent (64³)
  pad: u32,
};

struct DispatchArgs { count_x: u32, count_y: u32, count_z: u32 }

@group(0) @binding(0) var<storage, read> meta_buffer: array<BrickMeta>;
@group(0) @binding(1) var<storage, read_write> brick_state: array<u32>;
@group(0) @binding(2) var<storage, read_write> raw_queue: array<u32>;
@group(0) @binding(3) var<storage, read_write> queue_count: atomic<u32>;
@group(0) @binding(4) var<uniform> sched_params: vec4<f32>;
@group(0) @binding(5) var<uniform> camera_pos: vec3<f32>;

@compute @workgroup_size(64)
fn culling_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let brick_idx = gid.x;
  if (brick_idx >= 4096u) { return; }

  let meta = meta_buffer[brick_idx];
  let moisture = meta.moisture;
  let stability = meta.stability;

  // ── LOD-based activation ──
  // Convert brick_idx to world-space center
  // For 256³ grid with 16³ bricks: 16 bricks per axis
  let bx = brick_idx % 16u;
  let by = (brick_idx / 16u) % 16u;
  let bz = brick_idx / 256u;
  let world_center = vec3(f32(bx) * 16.0 + 8.0, f32(by) * 16.0 + 8.0, f32(bz) * 16.0 + 8.0);

  // Distance from camera for LOD selection
  let dist_to_camera = distance(world_center, camera_pos);

  // LOD selection: leaf (16³) within 40 units, parent (32³) within 80, grandparent beyond
  let selected_lod = select(2u, select(1u, 0u, dist_to_camera < 40.0), dist_to_camera < 80.0);

  // Only activate if this brick's current LOD matches or is finer than selected
  let lod_valid = meta.lod >= selected_lod;

  // EMA smoothing using brick_state high bits
  let last_moisture = f32((brick_state[brick_idx] >> 8u) & 0xFFu) / 255.0;
  let alpha = sched_params.z;
  let smooth_moisture = mix(last_moisture, moisture, alpha);

  // Hysteresis deadband
  let was_active = (brick_state[brick_idx] & 1u) == 1u;
  let activate = lod_valid && ((smooth_moisture > sched_params.x) || (stability < sched_params.y));
  let deactivate = !lod_valid || ((smooth_moisture < sched_params.x - sched_params.w) && (stability > sched_params.y + 0.1));
  let active = was_active ? !deactivate : activate;

  // Update persistent state
  var state = brick_state[brick_idx];
  state = (state & 0xFFFFFF00u) | (u32(clamp(smooth_moisture * 255.0, 0.0, 255.0)) << 8u);
  state = (state & 0xFFFFFFFEu) | (select(0u, 1u, active));
  brick_state[brick_idx] = state;

  // Enqueue if active
  if (active) {
    let slot = atomicAdd(&queue_count, 1u);
    if (slot < 4096u) { raw_queue[slot] = brick_idx; }
  }
}
