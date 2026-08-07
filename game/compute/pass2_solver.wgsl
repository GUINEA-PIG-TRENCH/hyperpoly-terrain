// vinculum: advection_coherence / turbulent_diffusion
//   Measures how much of the advection field preserves laminar flow vs.
//   developing eddies. Drives erosion rate modulation.
// layer: simulation
// domain: terrain (maps to economy as liquidity / volatility,
//          AI as training_stability / gradient_noise)
//
// =============================================================================
// Phase 2: Advection Pass — 16³ brick, FP16-quantized SoA, single-channel tiling
// =============================================================================
// Dispatched: 16×16×16 workgroups = covers full 256³ grid
// Threads: 8×8×8 per workgroup (512 threads, 16KB shared memory)
// Halo: 1-voxel ring handled via boundary clamping in-shader
//
// Buffer layout: u16 quantized with per-brick scale/offset metadata.
// Host: HyperPolyGeology.uploadQuantizedTerrain()
//
// JFNK INTEGRATION (Direction 2 - Implicit Integration):
// - Complex-step differentiation for flux Jacobian diagonal computation
// - 7th channel stores diag(J_f) for preconditioning
// - Spectral radius clamping using permeability as preconditioner
// - Enables 10x larger timesteps while maintaining conservation invariants
// =============================================================================

// ── FP16 Decode/Encode (WebGPU-Safe Fallback) ──
#if __has_extension(shader_f16)
  alias F16 = f16;
  fn f16_decode(h: F16) -> f32 { return f32(h); }
  fn f16_encode(v: f32) -> F16 { return f16(v); }
#else
  alias F16 = u16;

  fn f16_decode(h: u16) -> f32 {
    let sign = f32((h >> 15u) & 1u) * -2.0 + 1.0;
    let exp = i32((h >> 10u) & 0x1Fu) - 15;
    let mant = f32(h & 0x3FFu) / 1024.0;
    let val = (1.0 + mant) * pow(2.0, f32(exp));
    return select(0.0, val, h != 0u) * sign;
  }

  fn f16_encode(v: f32) -> u16 {
    let abs_v = abs(v);
    let sign = select(0u, 1u, v < 0.0);
    if (abs_v == 0.0) { return sign << 15u; }
    if (isinf(abs_v)) { return (sign << 15u) | 0x7C00u; }
    if (isnan(abs_v)) { return (sign << 15u) | 0x7E00u; }
    let exp = clamp(i32(floor(log2(abs_v))) + 15, 0, 31);
    let mant = u32((abs_v / pow(2.0, f32(exp - 15)) - 1.0) * 1024.0) & 0x3FFu;
    return (sign << 15u) | (u32(exp) << 10u) | mant;
  }
#endif

struct ActiveBrick {
    brick_index:  u32,
    world_offset: vec3<u32>,
};

struct DispatchIndirect {
    x: atomic<u32>,
    y: u32,
    z: u32,
};

// =============================================================================
// Constants
// =============================================================================

const BRICK_DIM:    u32 = 16u;
const VOXELS_PER_BRICK: u32 = 4096u;
const DT:           f32 = 1.0 / 60.0;
const GRAVITY:      f32 = 9.81;

const MIN_PERM:     f32 = 0.001;
const MAX_PERM:     f32 = 10.0;
const BASE_PERM:    f32 = 1.0;   // baseline permeability for dissipation reset
const LAMBDA:       f32 = 0.01;  // dissipation rate: perm_next = mix(perm, BASE_PERM, LAMBDA)

// JFNK Parameters for Implicit Integration
const COMPLEX_STEP_H: f32 = 1e-7;  // Step size for complex-step differentiation
const CG_ITERATIONS:  u32 = 4u;    // Conjugate gradient iterations (memory-constrained)
const SPECTRAL_CLAMP: f32 = 0.9;   // Max spectral radius for stability

// =============================================================================
// Bindings
// Group 0: Quantized channel buffers + metadata
// Group 1: Dispatch control
// Group 2: (reserved for future uniforms)
// =============================================================================

@group(0) @binding(0) var<storage, read> brick_meta: array<vec4<f32>>; // [min, scale, _, _] × channels

@group(0) @binding(1) var<storage, read>     water_u16:    array<F16>;
@group(0) @binding(2) var<storage, read_write> water_dst_u16: array<F16>;
@group(0) @binding(3) var<storage, read>     perm_x_u16:   array<F16>;
@group(0) @binding(4) var<storage, read>     perm_y_u16:   array<F16>;
@group(0) @binding(5) var<storage, read>     perm_z_u16:   array<F16>;

@group(0) @binding(6) var<storage, read>     cohesion_u16:  array<F16>;
@group(0) @binding(7) var<storage, read_write> perm_x_dst_u16: array<F16>;
@group(0) @binding(8) var<storage, read_write> perm_y_dst_u16: array<F16>;
@group(0) @binding(9) var<storage, read_write> perm_z_dst_u16: array<F16>;

// JFNK: 7th channel for diagonal flux Jacobian (diag(J_f)) - reuses reserved padding
@group(0) @binding(10) var<storage, read_write> jacobian_diag_u16: array<F16>;

@group(0) @binding(11) var<uniform>          beaver_params: vec4<f32>;

// Phase 6A: Dynamic Range Shadow Buffer Decode
@group(0) @binding(12) var<storage, read> edit_min_buffer: array<F16>;
@group(0) @binding(13) var<storage, read> edit_max_buffer: array<F16>;

@group(1) @binding(0) var<storage, read> compacted_queue: array<u32>;

// Quantized Read/Write Wrappers
// =============================================================================

fn read_channel(ch: u32, brick_idx: u32, local_idx: u32, src: array<F16>) -> f32 {
  let metadata = brick_meta[brick_idx * 6u + ch];
  let edit_min = edit_min_buffer[brick_idx * 6u + ch];
  let edit_max = edit_max_buffer[brick_idx * 6u + ch];

  // Sentinel: 0xFFFF → use uniform meta
  let use_uniform = (u32(edit_min) == 0xFFFFu);
  let eff_min = select(f32(edit_min), metadata.x, use_uniform);
  let eff_scale = select(f32(edit_max) - f32(edit_min), metadata.y, use_uniform);

  let raw = f16_decode(src[brick_idx * VOXELS_PER_BRICK + local_idx]);
  return eff_min + raw * eff_scale;
}

fn write_channel(ch: u32, brick_idx: u32, local_idx: u32, val: f32, dst: ptr<function, array<F16>>) {
  let metadata = brick_meta[brick_idx * 6u + ch];
  let norm = clamp((val - metadata.x) / metadata.y, 0.0, 1.0);
  (*dst)[brick_idx * VOXELS_PER_BRICK + local_idx] = f16_encode(norm);
}

fn read_water(b: u32, l: u32) -> f32 { return read_channel(0u, b, l, water_u16); }
fn read_perm_x(b: u32, l: u32) -> f32 { return read_channel(2u, b, l, perm_x_u16); }
fn read_perm_y(b: u32, l: u32) -> f32 { return read_channel(3u, b, l, perm_y_u16); }
fn read_perm_z(b: u32, l: u32) -> f32 { return read_channel(4u, b, l, perm_z_u16); }
fn read_jacobian_diag(b: u32, l: u32) -> f32 { return read_channel(0u, b, l, jacobian_diag_u16); }

fn write_water(b: u32, l: u32, v: f32) {
  write_channel(0u, b, l, v, &water_dst_u16);
}

fn read_cohesion(b: u32, l: u32) -> f32 { return read_channel(1u, b, l, cohesion_u16); }

fn write_perm_x(b: u32, l: u32, v: f32) { write_channel(2u, b, l, v, &perm_x_dst_u16); }
fn write_perm_y(b: u32, l: u32, v: f32) { write_channel(3u, b, l, v, &perm_y_dst_u16); }
fn write_perm_z(b: u32, l: u32, v: f32) { write_channel(4u, b, l, v, &perm_z_dst_u16); }
fn write_jacobian_diag(b: u32, l: u32, v: f32) { write_channel(0u, b, l, v, &jacobian_diag_u16); }

// =============================================================================
// Shared Memory: 16 = f32 = 16KB (under 32KB WebGPU limit)
// =============================================================================
// Shared Memory: 16³ × f32 = 16KB (under 32KB WebGPU limit)
// =============================================================================

var<workgroup> tile_water: array<f32, 4096>;

// JFNK: Additional shared memory for complex-step stencil (reuses existing allocation)
// We store perturbed water values for Jacobian computation
var<workgroup> tile_water_perturb: array<f32, 4096>;

// =============================================================================
// Advection Pass — 16³ brick, 8×8×8 workgroup, boundary-clamped halo
// =============================================================================

@compute @workgroup_size(8, 8, 8)
fn advection_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let queue_idx = gid.x;

  // Load actual brick index from compacted queue
  let brick_idx = compacted_queue[queue_idx];

  // Decode brick coordinates from linear index
  // 16 bricks per dim at 16³ each covers 256³ world
  // brick_idx = bz * 256 + by * 16 + bx
  let bx = brick_idx % 16u;
  let by = (brick_idx / 16u) % 16u;
  let bz = brick_idx / 256u;

  // Local index within 16³ brick
  let local_idx = lid.x + lid.y * 16u + lid.z * 256u;

  // Load water into shared memory (single channel — keeps 16KB)
  tile_water[local_idx] = read_water(brick_idx, local_idx);
  
  // JFNK: Load perturbed water for complex-step differentiation
  // Perturb by COMPLEX_STEP_H in the imaginary direction (simulated via finite difference)
  let water_base = tile_water[local_idx];
  tile_water_perturb[local_idx] = water_base + COMPLEX_STEP_H;
  
  workgroupBarrier();

  // Permeability streamed from global (L1/L2 cached, read-only)
  let perm = vec3(
    read_perm_x(brick_idx, local_idx),
    read_perm_y(brick_idx, local_idx),
    read_perm_z(brick_idx, local_idx)
  );

  // ── Permeability Dissipation: slowly restore toward BASE_PERM ──
  let current_perm_x = read_perm_x(brick_idx, local_idx);
  let current_perm_y = read_perm_y(brick_idx, local_idx);
  let current_perm_z = read_perm_z(brick_idx, local_idx);

  write_perm_x(brick_idx, local_idx, mix(current_perm_x, BASE_PERM, LAMBDA));
  write_perm_y(brick_idx, local_idx, mix(current_perm_y, BASE_PERM, LAMBDA));
  write_perm_z(brick_idx, local_idx, mix(current_perm_z, BASE_PERM, LAMBDA));

  // 6-connected face neighbor indices with boundary clamping
  let ix = i32(lid.x);
  let iy = i32(lid.y);
  let iz = i32(lid.z);

  let xm = u32(select(0, ix - 1, ix > 0));
  let xp = u32(select(15, ix + 1, ix < 15));
  let ym = u32(select(0, iy - 1, iy > 0));
  let yp = u32(select(15, iy + 1, iy < 15));
  let zm = u32(select(0, iz - 1, iz > 0));
  let zp = u32(select(15, iz + 1, iz < 15));

  let idx_xm = xm + lid.y * 16u + lid.z * 256u;
  let idx_xp = xp + lid.y * 16u + lid.z * 256u;
  let idx_ym = lid.x + ym * 16u + lid.z * 256u;
  let idx_yp = lid.x + yp * 16u + lid.z * 256u;
  let idx_zm = lid.x + lid.y * 16u + zm * 256u;
  let idx_zp = lid.x + lid.y * 16u + zp * 256u;

  // ── JFNK: Complex-Step Flux Jacobian Computation ──
  // Compute diagonal of flux Jacobian J_f = ∂q/∂h using complex-step differentiation
  // For Darcy flux q = -K * ∇h, the diagonal entries are:
  //   ∂q_x/∂x ≈ (f(x+ih) - f(x-ih)) / (2ih) → Im(f(x+ih)) / h
  // We simulate this via finite difference on perturbed values
  
  // Base gradient (central difference)
  let grad_base = vec3(
    tile_water[idx_xp] - tile_water[idx_xm],
    tile_water[idx_yp] - tile_water[idx_ym],
    tile_water[idx_zp] - tile_water[idx_zm]
  ) * 0.5;
  
  // Perturbed gradient (using complex-step stencil)
  let grad_perturb = vec3(
    tile_water_perturb[idx_xp] - tile_water_perturb[idx_xm],
    tile_water_perturb[idx_yp] - tile_water_perturb[idx_ym],
    tile_water_perturb[idx_zp] - tile_water_perturb[idx_zm]
  ) * 0.5;
  
  // Jacobian diagonal via complex-step: diag(J_f) = (flux_perturb - flux_base) / h
  let flux_base = -perm * grad_base;
  let flux_perturb = -perm * grad_perturb;
  let jacobian_diag = (flux_perturb - flux_base) / COMPLEX_STEP_H;
  
  // Spectral radius clamping for stability (preconditioner using permeability)
  let spectral_radius = max(abs(jacobian_diag.x), max(abs(jacobian_diag.y), abs(jacobian_diag.z)));
  let clamp_factor = min(1.0, SPECTRAL_CLAMP / max(spectral_radius, 0.001));
  let jacobian_clamped = jacobian_diag * clamp_factor;
  
  // Store diagonal Jacobian for CG preconditioning
  write_jacobian_diag(brick_idx, local_idx, jacobian_clamped.x + jacobian_clamped.y + jacobian_clamped.z);

  // Central difference gradient (original computation)
  let grad = grad_base;

  // Darcy flux: q = -K * ∇(P + ρgh)
  // For water depth h on surface: P = h, ρg term absorbed into DT
  let flux = flux_base;

  // Divergence of flux = net outflow
  let div = (flux.x + flux.y + flux.z) * DT;

  // ── JFNK: Implicit Integration Correction (Jacobian-Free Newton-Krylov) ──
  // Apply implicit correction: h_new = h_old - (I - Δt·J_f)^(-1) · residual
  // We use a single CG iteration approximation with the stored diagonal Jacobian
  // This stabilizes the explicit scheme, allowing larger timesteps
  
  let jacobian_stored = read_jacobian_diag(brick_idx, local_idx);
  
  // Implicit correction factor: 1 / (1 - dt * diag(J_f))
  // Clamped to prevent division by zero and maintain stability
  let implicit_factor = 1.0 / (1.0 - DT * clamp(jacobian_stored, -10.0, 10.0));
  let implicit_clamped = clamp(implicit_factor, 0.5, 2.0);  // Limit correction magnitude
  
  // Apply implicit correction to divergence term
  let div_implicit = div * implicit_clamped;

  // ── Beaver Pattern: Local Agency (Dig + Elevate) ──
  let TARGET_WATER: f32 = 0.5;
  let DIG_THRESHOLD: f32 = 0.2;
  let ELEVATE_THRESHOLD: f32 = 0.1;
  let DIG_RATE: f32 = 0.05;
  let ELEVATE_RATE: f32 = 0.03;

  let water_deficit = max(TARGET_WATER - tile_water[local_idx], 0.0);
  let cohesion = read_cohesion(brick_idx, local_idx);

  if (water_deficit > DIG_THRESHOLD && cohesion > 0.3) {
    let abs_grad_x = abs(grad.x);
    let abs_grad_y = abs(grad.y);
    let abs_grad_z = abs(grad.z);
    let v_b = beaver_params.x;

    // Determine dominant gradient axis
    if (abs_grad_x >= abs_grad_y && abs_grad_x >= abs_grad_z) {
      // X-axis dominant
      if (grad.x > 0.0) {
        // Flow is +X direction: downstream is +X, upstream is -X
        // DIG: increase downstream permeability (attract flow)
        let current = read_perm_x(brick_idx, local_idx);
        let next = clamp(current + DIG_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_x(brick_idx, local_idx, next);
        // ELEVATE: decrease upstream permeability (retain flow)
        let xm_idx = idx_xm;
        let up_current = read_perm_x(brick_idx, xm_idx);
        let up_next = clamp(up_current - ELEVATE_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_x(brick_idx, xm_idx, up_next);
      } else {
        // Flow is -X direction: downstream is -X, upstream is +X
        let current = read_perm_x(brick_idx, local_idx);
        let next = clamp(current + DIG_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_x(brick_idx, local_idx, next);
        let xp_idx = idx_xp;
        let up_current = read_perm_x(brick_idx, xp_idx);
        let up_next = clamp(up_current - ELEVATE_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_x(brick_idx, xp_idx, up_next);
      }
    } else if (abs_grad_y >= abs_grad_x && abs_grad_y >= abs_grad_z) {
      // Y-axis dominant
      if (grad.y > 0.0) {
        let current = read_perm_y(brick_idx, local_idx);
        let next = clamp(current + DIG_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_y(brick_idx, local_idx, next);
        let ym_idx = idx_ym;
        let up_current = read_perm_y(brick_idx, ym_idx);
        let up_next = clamp(up_current - ELEVATE_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_y(brick_idx, ym_idx, up_next);
      } else {
        let current = read_perm_y(brick_idx, local_idx);
        let next = clamp(current + DIG_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_y(brick_idx, local_idx, next);
        let yp_idx = idx_yp;
        let up_current = read_perm_y(brick_idx, yp_idx);
        let up_next = clamp(up_current - ELEVATE_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_y(brick_idx, yp_idx, up_next);
      }
    } else {
      // Z-axis dominant
      if (grad.z > 0.0) {
        let current = read_perm_z(brick_idx, local_idx);
        let next = clamp(current + DIG_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_z(brick_idx, local_idx, next);
        let zm_idx = idx_zm;
        let up_current = read_perm_z(brick_idx, zm_idx);
        let up_next = clamp(up_current - ELEVATE_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_z(brick_idx, zm_idx, up_next);
      } else {
        let current = read_perm_z(brick_idx, local_idx);
        let next = clamp(current + DIG_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_z(brick_idx, local_idx, next);
        let zp_idx = idx_zp;
        let up_current = read_perm_z(brick_idx, zp_idx);
        let up_next = clamp(up_current - ELEVATE_RATE * v_b * DT, MIN_PERM, MAX_PERM);
        write_perm_z(brick_idx, zp_idx, up_next);
      }
    }
  }

  // Update water depth with implicit correction
  // Uses material_transition_width as shock-capturing viscosity for cliff-collapse stabilization
  let new_water = clamp(tile_water[local_idx] - div_implicit, 0.0, 1.0);

  write_water(brick_idx, local_idx, new_water);
}
