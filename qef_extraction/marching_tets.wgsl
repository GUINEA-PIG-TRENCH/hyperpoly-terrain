// marching_tets.wgsl
// Kernel 2: Marching Tetrahedra — classify edges, emit crossings.
// Each grid cell decomposed into 6 tetrahedra.
// Output: per-cell crossing mask + edge intersection points.
//
// Dispatch: (64, 64, 1) workgroups of (1, 1, 64)

struct MTParams {
    grid_dim: u32,        // 64
    cell_size: f32,
    isosurface: f32,      // threshold, typically 0.1
};

// Cube vertex indices in [0,7] for the 8 corners.
// Corner ordering: (x,y,z) where bit 0=x, bit 1=y, bit 2=z
//
//   z=1:  4---5     z=0:  0---1
//         |\  |\          |\  |\
//         7---6            3---2
//
// 6 tetrahedra decomposing a cube (vertex indices into cube corners [0..7]):
// Each tet shares the cube diagonal (0,7) for consistency.
const TET_DECOMP: array<vec4<u32>, 6> = array<vec4<u32>, 6>(
    vec4<u32>(0u, 1u, 3u, 7u),   // tet 0
    vec4<u32>(0u, 1u, 5u, 7u),   // tet 1
    vec4<u32>(0u, 4u, 5u, 7u),   // tet 2
    vec4<u32>(0u, 4u, 6u, 7u),   // tet 3
    vec4<u32>(0u, 2u, 3u, 7u),   // tet 4
    vec4<u32>(0u, 2u, 6u, 7u),   // tet 5
);

// Tetrahedron edges: pairs of vertex indices (6 edges per tet).
// Indices are into the tet's 4 vertices [0..3].
const TET_EDGES: array<vec2<u32>, 6> = array<vec2<u32>, 6>(
    vec2<u32>(0u, 1u), vec2<u32>(0u, 2u), vec2<u32>(0u, 3u),
    vec2<u32>(1u, 2u), vec2<u32>(1u, 3u), vec2<u32>(2u, 3u),
);

// Cube corner positions (unit cube [0,1]³)
fn cube_corner(idx: u32) -> vec3<f32> {
    return vec3<f32>(
        f32((idx >> 0u) & 1u),
        f32((idx >> 1u) & 1u),
        f32((idx >> 2u) & 1u),
    );
}

@group(1) @binding(0) var<storage, read> density_field: array<f32>;
@group(1) @binding(1) var<storage, read_write> crossing_count: atomic<u32>;  // total crossings
@group(1) @binding(2) var<storage, read_write> crossings: array<u32>;          // packed: cell_idx | edge_data
@group(1) @binding(3) var<uniform> mt_params: MTParams;
@group(1) @binding(4) var<storage, read_write> crossing_lut: array<u32>;

// Read density at a grid corner, with bounds check
fn density_at(cx: u32, cy: u32, cz: u32) -> f32 {
    if (cx >= mt_params.grid_dim || cy >= mt_params.grid_dim || cz >= mt_params.grid_dim) {
        return 0.0;
    }
    let idx = cx + cy * mt_params.grid_dim + cz * mt_params.grid_dim * mt_params.grid_dim;
    return density_field[idx];
}

// Read density at cube corner (0..7) given cell origin
fn cube_corner_density(cx: u32, cy: u32, cz: u32, corner: u32) -> f32 {
    let dx = (corner >> 0u) & 1u;
    let dy = (corner >> 1u) & 1u;
    let dz = (corner >> 2u) & 1u;
    return density_at(cx + dx, cy + dy, cz + dz);
}

// Linear interpolation along edge to find crossing point
fn edge_interp(d0: f32, d1: f32, p0: vec3<f32>, p1: vec3<f32>) -> vec3<f32> {
    let diff = d1 - d0;
    let sign_diff = if (diff >= 0.0) { 1.0 } else { -1.0 };
    let safe_diff = sign_diff * max(abs(diff), 1e-6);
    let t = (mt_params.isosurface - d0) / safe_diff;
    return mix(p0, p1, clamp(t, 0.001, 0.999));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    let cz = gid.z;
    
    if (cx >= mt_params.grid_dim - 1u || cy >= mt_params.grid_dim - 1u || cz >= mt_params.grid_dim - 1u) {
        return;
    }
    
    let cell_base = cx + cy * mt_params.grid_dim + cz * mt_params.grid_dim * mt_params.grid_dim;
    
    // Read corner densities
    var corner_d: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        corner_d[i] = cube_corner_density(cx, cy, cz, i);
    }
    
    var cell_crossings = 0u;
    
    // Process 6 tetrahedra
    for (var t = 0u; t < 6u; t++) {
        let tet = TET_DECOMP[t];
        var tet_d: array<f32, 4>;
        var tet_p: array<vec3<f32>, 4>;
        
        for (var v = 0u; v < 4u; v++) {
            let cv = tet[v];
            tet_d[v] = corner_d[cv];
            tet_p[v] = cube_corner(cv);
        }
        
        // Classify vertices: above or below isosurface
        var mask = 0u;
        for (var v = 0u; v < 4u; v++) {
            if (tet_d[v] >= mt_params.isosurface) {
                mask |= (1u << v);
            }
        }
        
        // Degenerate cases: all above or all below → no crossing
        if (mask == 0u || mask == 15u) { continue; }
        
        // Check each of 6 edges for crossing
        for (var e = 0u; e < 6u; e++) {
            let e0 = TET_EDGES[e].x;
            let e1 = TET_EDGES[e].y;
            
            let above0 = (mask >> e0) & 1u;
            let above1 = (mask >> e1) & 1u;
            
            if (above0 == above1) { continue; }  // no sign change
            
            // Crossing found — compute intersection point
            let d0 = tet_d[e0];
            let d1 = tet_d[e1];
            let p0 = tet_p[e0];
            let p1 = tet_p[e1];
            
            let isect = edge_interp(d0, d1, p0, p1);
            
            // Pack: cell_base (18 bits) | t (3 bits) | e (3 bits) — fits in u32
            let edge_data = cell_base;
            let packed = (edge_data << 6u) | (t << 3u) | e;
            
            // Atomic append to crossings buffer
            let slot = atomicAdd(&crossing_count, 1u);
            if (slot < arrayLength(&crossings)) {
                crossings[slot] = packed;
                let lut_idx = cell_base * 36u + t * 6u + e;
                if (lut_idx < arrayLength(&crossing_lut)) {
                    crossing_lut[lut_idx] = slot + 1u;
                }
            }
            
            cell_crossings++;
        }
    }
}
