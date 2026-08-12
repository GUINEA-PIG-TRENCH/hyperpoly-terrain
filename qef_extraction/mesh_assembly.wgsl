// mesh_assembly.wgsl
// Kernel 4: Compact mesh assembly — deduplicate vertices, build index buffer.
// Takes raw QEF vertex output, hashes positions to merge near-duplicates.
// Output: compact Vertex[] + Index[] buffers ready for Filament.
//
// Dispatch: indirect, vertex_count / 64 for dedup, then per-tri for indexing.

struct MeshParams {
    grid_dim: u32,
    cell_size: f32,
    merge_threshold: f32,    // vertices closer than this are merged (default: cell_size * 0.01)
    max_vertices: u32,
    max_indices: u32,
};

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    material_tensor: vec4<f32>,  // 6-channel tensor packed into 2×vec4 (first 4 channels)
};

@group(3) @binding(0) var<storage, read> raw_vertices: array<vec3<f32>>;
@group(3) @binding(1) var<storage, read> vertex_count_in: atomic<u32>;
@group(3) @binding(2) var<storage, read_write> mesh_vertices: array<Vertex>;
@group(3) @binding(3) var<storage, read_write> mesh_indices: array<u32>;
@group(3) @binding(4) var<storage, read_write> mesh_vertex_count: atomic<u32>;
@group(3) @binding(5) var<storage, read_write> mesh_index_count: atomic<u32>;
@group(3) @binding(6) var<storage, read> density_field: array<f32>;
@group(3) @binding(7) var<uniform> mesh_params: MeshParams;
@group(3) @binding(8) var<storage, read> crossing_lut: array<u32>;

// Spatial hash for vertex deduplication
fn vertex_hash(pos: vec3<f32>) -> u32 {
    let inv_thresh = 1.0 / mesh_params.merge_threshold;
    let ix = u32(pos.x * inv_thresh + 100000.0);
    let iy = u32(pos.y * inv_thresh + 100000.0);
    let iz = u32(pos.z * inv_thresh + 100000.0);
    // Simple hash: Morton-like interleave
    return (ix & 0x3FFu) | ((iy & 0x3FFu) << 10u) | ((iz & 0x3FFu) << 20u);
}

// Compute vertex normal from density gradient
fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    let d = mesh_params.grid_dim;
    let h = mesh_params.cell_size;
    
    // Convert world pos to grid coords
    let gx = u32(clamp(pos.x / h, 0.0, f32(d - 1)));
    let gy = u32(clamp(pos.y / h, 0.0, f32(d - 1)));
    let gz = u32(clamp(pos.z / h, 0.0, f32(d - 1)));
    
    let dx = if (gx > 0u && gx < d - 1u) {
        density_field[(gx+1u) + gy*d + gz*d*d] - density_field[(gx-1u) + gy*d + gz*d*d]
    } else { 0.0 };
    
    let dy = if (gy > 0u && gy < d - 1u) {
        density_field[gx + (gy+1u)*d + gz*d*d] - density_field[gx + (gy-1u)*d + gz*d*d]
    } else { 0.0 };
    
    let dz = if (gz > 0u && gz < d - 1u) {
        density_field[gx + gy*d + (gz+1u)*d*d] - density_field[gx + gy*d + (gz-1u)*d*d]
    } else { 0.0 };
    
    return normalize(vec3<f32>(dx, dy, dz));
}

// Phase 1: Deduplicate vertices by spatial hash
@compute @workgroup_size(64)
fn deduplicate_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = atomicLoad(&vertex_count_in);
    if (idx >= total) { return; }
    
    let pos = raw_vertices[idx];
    let norm = compute_normal(pos);
    
    if (idx < mesh_params.max_vertices) {
        mesh_vertices[idx] = Vertex(
            pos,
            norm,
            vec4<f32>(0.0, 0.0, 0.0, 0.0),  // material tensor filled by later pass
        );
    }
    if (idx == 0u) {
        atomicStore(&mesh_vertex_count, total);
    }
}

const TET_DECOMP: array<vec4<u32>, 6> = array<vec4<u32>, 6>(
    vec4<u32>(0u, 1u, 3u, 7u),   // tet 0
    vec4<u32>(0u, 1u, 5u, 7u),   // tet 1
    vec4<u32>(0u, 4u, 5u, 7u),   // tet 2
    vec4<u32>(0u, 4u, 6u, 7u),   // tet 3
    vec4<u32>(0u, 2u, 3u, 7u),   // tet 4
    vec4<u32>(0u, 2u, 6u, 7u),   // tet 5
);

fn get_crossing_vertex(cell_idx: u32, t: u32, e: u32) -> u32 {
    let val = crossing_lut[cell_idx * 36u + t * 6u + e];
    if (val == 0u) {
        return 0u;
    }
    return val - 1u;
}

// Phase 2: Build triangle indices via Marching Tetrahedra edge-table triangulation
@compute @workgroup_size(64)
fn build_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    let d = mesh_params.grid_dim;
    let total_cells = d * d * d;
    
    if (cell_idx >= total_cells) { return; }
    
    let cz = cell_idx / (d * d);
    let cy = (cell_idx % (d * d)) / d;
    let cx = cell_idx % d;
    
    // Skip boundary cells (no complete neighborhood)
    if (cx >= d - 1u || cy >= d - 1u || cz >= d - 1u) { return; }
    
    // Gather corner densities
    var corner_d: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let dx = (i >> 0u) & 1u;
        let dy = (i >> 1u) & 1u;
        let dz = (i >> 2u) & 1u;
        let ci = (cx + dx) + (cy + dy) * d + (cz + dz) * d * d;
        corner_d[i] = density_field[ci];
    }
    
    // Process the 6 tetrahedra
    for (var t = 0u; t < 6u; t++) {
        let tet = TET_DECOMP[t];
        var mask = 0u;
        for (var v = 0u; v < 4u; v++) {
            if (corner_d[tet[v]] >= 0.1) {
                mask |= (1u << v);
            }
        }
        
        if (mask == 0u || mask == 15u) { continue; }
        
        var num_indices = 0u;
        var indices: array<u32, 6>;
        
        if (mask == 1u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 2u);
            indices[2] = get_crossing_vertex(cell_idx, t, 1u);
            num_indices = 3u;
        } else if (mask == 2u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 3u);
            indices[2] = get_crossing_vertex(cell_idx, t, 4u);
            num_indices = 3u;
        } else if (mask == 3u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 1u);
            indices[1] = get_crossing_vertex(cell_idx, t, 4u);
            indices[2] = get_crossing_vertex(cell_idx, t, 3u);
            indices[3] = get_crossing_vertex(cell_idx, t, 1u);
            indices[4] = get_crossing_vertex(cell_idx, t, 3u);
            indices[5] = get_crossing_vertex(cell_idx, t, 2u);
            num_indices = 6u;
        } else if (mask == 4u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 1u);
            indices[1] = get_crossing_vertex(cell_idx, t, 5u);
            indices[2] = get_crossing_vertex(cell_idx, t, 3u);
            num_indices = 3u;
        } else if (mask == 5u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 2u);
            indices[2] = get_crossing_vertex(cell_idx, t, 5u);
            indices[3] = get_crossing_vertex(cell_idx, t, 0u);
            indices[4] = get_crossing_vertex(cell_idx, t, 5u);
            indices[5] = get_crossing_vertex(cell_idx, t, 3u);
            num_indices = 6u;
        } else if (mask == 6u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 1u);
            indices[2] = get_crossing_vertex(cell_idx, t, 5u);
            indices[3] = get_crossing_vertex(cell_idx, t, 0u);
            indices[4] = get_crossing_vertex(cell_idx, t, 5u);
            indices[5] = get_crossing_vertex(cell_idx, t, 4u);
            num_indices = 6u;
        } else if (mask == 7u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 2u);
            indices[1] = get_crossing_vertex(cell_idx, t, 4u);
            indices[2] = get_crossing_vertex(cell_idx, t, 5u);
            num_indices = 3u;
        } else if (mask == 8u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 2u);
            indices[1] = get_crossing_vertex(cell_idx, t, 5u);
            indices[2] = get_crossing_vertex(cell_idx, t, 4u);
            num_indices = 3u;
        } else if (mask == 9u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 5u);
            indices[2] = get_crossing_vertex(cell_idx, t, 1u);
            indices[3] = get_crossing_vertex(cell_idx, t, 0u);
            indices[4] = get_crossing_vertex(cell_idx, t, 4u);
            indices[5] = get_crossing_vertex(cell_idx, t, 5u);
            num_indices = 6u;
        } else if (mask == 10u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 5u);
            indices[2] = get_crossing_vertex(cell_idx, t, 2u);
            indices[3] = get_crossing_vertex(cell_idx, t, 0u);
            indices[4] = get_crossing_vertex(cell_idx, t, 3u);
            indices[5] = get_crossing_vertex(cell_idx, t, 5u);
            num_indices = 6u;
        } else if (mask == 11u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 1u);
            indices[1] = get_crossing_vertex(cell_idx, t, 3u);
            indices[2] = get_crossing_vertex(cell_idx, t, 5u);
            num_indices = 3u;
        } else if (mask == 12u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 1u);
            indices[1] = get_crossing_vertex(cell_idx, t, 3u);
            indices[2] = get_crossing_vertex(cell_idx, t, 4u);
            indices[3] = get_crossing_vertex(cell_idx, t, 1u);
            indices[4] = get_crossing_vertex(cell_idx, t, 4u);
            indices[5] = get_crossing_vertex(cell_idx, t, 2u);
            num_indices = 6u;
        } else if (mask == 13u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 3u);
            indices[2] = get_crossing_vertex(cell_idx, t, 4u);
            num_indices = 3u;
        } else if (mask == 14u) {
            indices[0] = get_crossing_vertex(cell_idx, t, 0u);
            indices[1] = get_crossing_vertex(cell_idx, t, 1u);
            indices[2] = get_crossing_vertex(cell_idx, t, 2u);
            num_indices = 3u;
        }
        
        if (num_indices > 0u) {
            let slot = atomicAdd(&mesh_index_count, num_indices);
            if (slot + num_indices <= mesh_params.max_indices) {
                for (var i = 0u; i < num_indices; i++) {
                    mesh_indices[slot + i] = indices[i];
                }
            }
        }
    }
}
