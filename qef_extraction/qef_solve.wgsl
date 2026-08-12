// qef_solve.wgsl
// Kernel 3: QEF Vertex Placement — solve 3×3 system per crossing cell.
// Minimize Σ (n_i · (v - p_i))² where n_i is gradient, p_i is crossing point.
// Each cell gets ONE vertex placed at the minimizer of its quadric error.
//
// Dispatch: indirect, crossing_count / 64 workgroups

struct QEFParams {
    grid_dim: u32,
    cell_size: f32,
    regularization: f32,  // small epsilon for singular matrices (default 0.001)
};

@group(2) @binding(0) var<storage, read> density_field: array<f32>;
@group(2) @binding(1) var<storage, read> crossings: array<u32>;   // packed edge data from MT pass
@group(2) @binding(2) var<storage, read> crossing_count: atomic<u32>;
@group(2) @binding(3) var<storage, read_write> vertices: array<vec3<f32>>;   // output: one vertex per crossing
@group(2) @binding(4) var<storage, read_write> vertex_count: atomic<u32>;
@group(2) @binding(5) var<uniform> qef_params: QEFParams;

// --- 3×3 SVD (closed form, no iteration) ---
// Solves A^T A x = A^T b for x = vertex position
// A is m×3 (normals), b is m×1 (n·p)
// Using normal equations: (A^T A) x = A^T b

struct Mat3x3 {
    m: array<f32, 9>,  // column-major: m[col*3 + row]
}

fn mat3_zero() -> Mat3x3 {
    return Mat3x3(array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}

fn mat3_add_outer(a: vec3<f32>, weight: f32) -> Mat3x3 {
    // Returns weight * (a ⊗ a) as a matrix to accumulate into A^T A
    var m = mat3_zero();
    m.m[0] = a.x * a.x * weight;  // col 0
    m.m[1] = a.x * a.y * weight;
    m.m[2] = a.x * a.z * weight;
    m.m[3] = a.y * a.x * weight;  // col 1
    m.m[4] = a.y * a.y * weight;
    m.m[5] = a.y * a.z * weight;
    m.m[6] = a.z * a.x * weight;  // col 2
    m.m[7] = a.z * a.y * weight;
    m.m[8] = a.z * a.z * weight;
    return m;
}

// Cramer's rule for 3×3 system (robust enough for QEF with regularization)
fn solve_3x3(ata: Mat3x3, atb: vec3<f32>, reg: f32, fallback: vec3<f32>) -> vec3<f32> {
    // Add regularization to diagonal
    var a00 = ata.m[0] + reg;
    var a01 = ata.m[1];
    var a02 = ata.m[2];
    var a10 = ata.m[3];
    var a11 = ata.m[4] + reg;
    var a12 = ata.m[5];
    var a20 = ata.m[6];
    var a21 = ata.m[7];
    var a22 = ata.m[8] + reg;
    
    // Determinant
    let det = a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);
    
    if (abs(det) < 1e-12) {
        // Singular — fall back to centroid
        return fallback;
    }
    
    let inv_det = 1.0 / det;
    
    // Cramer's rule
    let x = (atb.x * (a11 * a22 - a12 * a21)
           - a01 * (atb.y * a22 - a12 * atb.z)
           + a02 * (atb.y * a21 - a11 * atb.z)) * inv_det;
    
    let y = (a00 * (atb.y * a22 - a12 * atb.z)
           - atb.x * (a10 * a22 - a12 * a20)
           + a02 * (a10 * atb.z - atb.y * a20)) * inv_det;
    
    let z = (a00 * (a11 * atb.z - atb.y * a21)
           - a01 * (a10 * atb.z - atb.y * a20)
           + atb.x * (a10 * a21 - a11 * a20)) * inv_det;
    
    return vec3<f32>(x, y, z);
}

// --- Density gradient via central differences ---
fn density_gradient(cx: u32, cy: u32, cz: u32) -> vec3<f32> {
    let d = qef_params.grid_dim;
    let h = qef_params.cell_size;
    
    let dx = if (cx > 0u && cx < d - 1u) {
        let r = density_field[(cx+1u) + cy*d + cz*d*d];
        let l = density_field[(cx-1u) + cy*d + cz*d*d];
        (r - l) / (2.0 * h)
    } else { 0.0 };
    
    let dy = if (cy > 0u && cy < d - 1u) {
        let r = density_field[cx + (cy+1u)*d + cz*d*d];
        let l = density_field[cx + (cy-1u)*d + cz*d*d];
        (r - l) / (2.0 * h)
    } else { 0.0 };
    
    let dz = if (cz > 0u && cz < d - 1u) {
        let r = density_field[cx + cy*d + (cz+1u)*d*d];
        let l = density_field[cx + cy*d + (cz-1u)*d*d];
        (r - l) / (2.0 * h)
    } else { 0.0 };
    
    return normalize(vec3<f32>(dx, dy, dz));
}

// Unpack edge data from MT pass
fn unpack_crossing(packed: u32) -> vec3<u32> {
    let edge_bits = packed & 0x3Fu;        // lower 6 bits: tet + edge id
    let cell_idx = packed >> 6u;            // upper bits: cell index
    return vec3<u32>(cell_idx, edge_bits >> 3u, edge_bits & 0x7u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = atomicLoad(&crossing_count);
    
    if (idx >= total) { return; }
    
    let packed = crossings[idx];
    let unpacked = unpack_crossing(packed);
    let cell_idx = unpacked.x;
    
    // Recover cell coordinates
    let d = qef_params.grid_dim;
    let cz = cell_idx / (d * d);
    let cy = (cell_idx % (d * d)) / d;
    let cx = cell_idx % d;
    
    // --- Build QEF for this cell ---
    // For each of the 8 corners that cross the isosurface, add:
    // n_i · (v - p_i) term to the quadric
    // Where n_i = gradient at crossing point, p_i = crossing position
    
    var ata = mat3_zero();
    var atb = vec3<f32>(0.0);
    var centroid = vec3<f32>(0.0);
    var point_count = 0u;
    
    // Gather crossing points from all 6 tetrahedra edges in this cell
    // We re-derive crossings from the density field for correctness
    // (avoids storing intersection points in the MT pass)
    
    // Cube corner densities
    var corner_d: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let dx = (i >> 0u) & 1u;
        let dy = (i >> 1u) & 1u;
        let dz = (i >> 2u) & 1u;
        let ci = (cx + dx) + (cy + dy) * d + (cz + dz) * d * d;
        corner_d[i] = density_field[ci];
    }
    
    let cell_origin = vec3<f32>(f32(cx), f32(cy), f32(cz)) * qef_params.cell_size;
    
    // Re-derive crossings from all 6 tets
    for (var t = 0u; t < 6u; t++) {
        let tet = TET_DECOMP[t];
        var tet_d: array<f32, 4>;
        var tet_p: array<vec3<f32>, 4>;
        
        for (var v = 0u; v < 4u; v++) {
            tet_d[v] = corner_d[tet[v]];
        }
        
        var mask = 0u;
        for (var v = 0u; v < 4u; v++) {
            if (tet_d[v] >= 0.1) { mask |= (1u << v); }
        }
        if (mask == 0u || mask == 15u) { continue; }
        
        for (var e = 0u; e < 6u; e++) {
            let e0 = TET_EDGES[e].x;
            let e1 = TET_EDGES[e].y;
            let above0 = (mask >> e0) & 1u;
            let above1 = (mask >> e1) & 1u;
            if (above0 == above1) { continue; }
            
            // Interpolate crossing position
            let d0 = tet_d[e0];
            let d1 = tet_d[e1];
            let t_val = (0.1 - d0) / (d1 - d0);
            
            // Compute world-space position
            let cv0 = tet[e0];
            let cv1 = tet[e1];
            let p0 = cell_origin + vec3<f32>(f32((cv0>>0u)&1u), f32((cv0>>1u)&1u), f32((cv0>>2u)&1u)) * qef_params.cell_size;
            let p1 = cell_origin + vec3<f32>(f32((cv1>>0u)&1u), f32((cv1>>1u)&1u), f32((cv1>>2u)&1u)) * qef_params.cell_size;
            let p = mix(p0, p1, clamp(t_val, 0.0, 1.0));
            
            // Gradient at crossing point
            let n = density_gradient(cx, cy, cz);
            
            // Accumulate: A^T A += n ⊗ n,  A^T b += (n·p) * n
            let accum = mat3_add_outer(n, 1.0);
            ata.m[0] += accum.m[0]; ata.m[1] += accum.m[1]; ata.m[2] += accum.m[2];
            ata.m[3] += accum.m[3]; ata.m[4] += accum.m[4]; ata.m[5] += accum.m[5];
            ata.m[6] += accum.m[6]; ata.m[7] += accum.m[7]; ata.m[8] += accum.m[8];
            
            let dot_np = dot(n, p);
            atb += n * dot_np;
            centroid += p;
            point_count++;
        }
    }
    
    if (idx == 0u) {
        atomicStore(&vertex_count, total);
    }

    // Fallback: if no crossings (shouldn't happen), use cell center
    if (point_count == 0u) {
        let v_out = cell_origin + vec3<f32>(0.5) * qef_params.cell_size;
        if (idx < arrayLength(&vertices)) {
            vertices[idx] = v_out;
        }
        return;
    }
    
    centroid /= f32(point_count);
    
    // Solve QEF
    let v_qef = solve_3x3(ata, atb, qef_params.regularization, centroid);
    
    // Clamp vertex to cell bounds
    let v_clamped = clamp(v_qef, cell_origin, cell_origin + vec3<f32>(qef_params.cell_size));
    
    if (idx < arrayLength(&vertices)) {
        vertices[idx] = v_clamped;
    }
}
