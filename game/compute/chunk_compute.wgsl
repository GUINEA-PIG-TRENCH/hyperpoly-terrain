// ──────────────────────────────────────────────────────────────
// CHUNK COMPUTE SHADER — GPU-Driven Chunk Visibility & LOD
// 
// This shader runs each frame to:
//   1. Cull chunks outside camera frustum
//   2. Select LOD level based on distance
//   3. Build indirect draw buffer for GPU-rendered chunks
//   4. Sample 6-channel terrain tensor for material data
//
// Integration with hyperpoly-terrain:
//   - Reads 6-channel storage buffer (density, cohesion, permeability, water, sediment, oxidation)
//   - Brick-aligned chunk boundaries (16³ voxels per brick)
//   - Zero CPU sync: all culling/LOD decisions on GPU
// ──────────────────────────────────────────────────────────────

struct ChunkUniforms {
    camera_pos: vec3<f32>,
    _pad0: f32,
    camera_dir: vec3<f32>,
    _pad1: f32,
    prefetch_radius: f32,
    lod_levels: f32,
    _reserved: vec2<f32>,
};

@group(0) @binding(0) var<uniform> chunk_params: ChunkUniforms;

// 6-channel terrain tensor from HyperPolyTerrain
// Layout: [density, cohesion, permeability, water, sediment, oxidation]
// Each channel is u16 quantized, accessed via byte offset calculation
@group(0) @binding(1) var<storage, read> terrain_tensor: array<u16>;

// Metadata buffer: min/scale per channel per brick
struct BrickMeta {
    min_val: f32,
    scale: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(2) var<storage, read> brick_meta: array<BrickMeta>;

// Indirect draw buffer output
// Format: vertex_count, instance_count, first_vertex, base_vertex, first_instance
struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    base_vertex: u32,
    first_instance: u32,
};
@group(0) @binding(3) var<storage, read_write> indirect_draws: array<DrawIndirect>;

// Visible chunk count (atomic increment)
@group(0) @binding(4) var<storage, read_write> visible_count: atomic<u32>;

// Constants
const BRICK_SIZE: u32 = 16u;
const BRICKS_PER_DIM: u32 = 16u;
const TOTAL_BRICKS: u32 = BRICKS_PER_DIM * BRICKS_PER_DIM * BRICKS_PER_DIM; // 4096
const CHANNELS: u32 = 6u;
const VOXELS_PER_BRICK: u32 = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE; // 4096

// Chunk data structure (computed per invocation)
struct ChunkData {
    chunk_id: u32,
    coord_x: u32,
    coord_y: u32,
    coord_z: u32,
    world_pos: vec3<f32>,
    distance_to_camera: f32,
    lod_level: u32,
    visible: bool,
    in_frustum: bool,
};

// ── Helper Functions ───────────────────────────────────────────

/// Convert chunk ID to 3D coordinates
fn chunkIdToCoord(chunk_id: u32) -> vec3<u32> {
    let bz = chunk_id / (BRICKS_PER_DIM * BRICKS_PER_DIM);
    let remainder = chunk_id % (BRICKS_PER_DIM * BRICKS_PER_DIM);
    let by = remainder / BRICKS_PER_DIM;
    let bx = remainder % BRICKS_PER_DIM;
    return vec3<u32>(bx, by, bz);
}

/// Get world position of chunk center
fn getChunkWorldPos(coord: vec3<u32>) -> vec3<f32> {
    let offset = vec3<f32>(coord) * f32(BRICK_SIZE);
    return offset + vec3<f32>(f32(BRICK_SIZE) * 0.5);
}

/// Calculate distance-based LOD level
fn calculateLOD(distance: f32, max_lod: u32) -> u32 {
    // LOD thresholds (tunable parameters)
    let lod_distances = array<f32, 4>(20.0, 50.0, 100.0, 200.0);
    
    var lod: u32 = 0u;
    for (var i: u32 = 0u; i < u32(max_lod) && i < 4u; i++) {
        if (distance > lod_distances[i]) {
            lod = i + 1u;
        }
    }
    return min(lod, max_lod - 1u);
}

/// Simple sphere frustum culling (can be extended to full frustum planes)
fn isInFrustum(world_pos: vec3<f32>, chunk_radius: f32) -> bool {
    let cam_pos = chunk_params.camera_pos;
    let cam_dir = normalize(chunk_params.camera_dir);
    
    // Vector from camera to chunk
    let to_chunk = world_pos - cam_pos;
    let dist = length(to_chunk);
    
    // Direction to chunk
    let dir_to_chunk = normalize(to_chunk);
    
    // Dot product test: is chunk within ~60° FOV cone?
    let dot_prod = dot(cam_dir, dir_to_chunk);
    let fov_threshold = cos(0.52); // ~60° horizontal FOV
    
    // Near/far plane culling
    let near_plane = 1.0;
    let far_plane = 500.0;
    
    return (dot_prod > fov_threshold) && 
           (dist > near_plane - chunk_radius) && 
           (dist < far_plane + chunk_radius);
}

/// Sample a specific channel from the 6-channel terrain tensor
/// Returns normalized value in [0, 1] range
fn sampleChannel(brick_idx: u32, voxel_idx: u32, channel: u32) -> f32 {
    // Validate inputs
    if (brick_idx >= TOTAL_BRICKS || channel >= CHANNELS) {
        return 0.0;
    }
    
    // Calculate flat index into the tensor buffer
    // Layout: all bricks for channel 0, then all bricks for channel 1, etc.
    let channel_offset = u32(channel) * TOTAL_BRICKS * VOXELS_PER_BRICK;
    let brick_offset = u32(brick_idx) * VOXELS_PER_BRICK;
    let flat_idx = channel_offset + brick_offset + voxel_idx;
    
    // Read quantized u16 value
    let quantized_val = terrain_tensor[flat_idx];
    
    // Get metadata for dequantization
    let meta_idx = u32(brick_idx) * CHANNELS + channel;
    let meta = brick_meta[meta_idx];
    
    // Dequantize: value = min + (quantized / 65535) * scale
    let normalized = f32(quantized_val) / 65535.0;
    return meta.min_val + normalized * meta.scale;
}

/// Get average density/cohesion for a chunk (for LOD biasing)
fn getChunkMaterialProperties(coord: vec3<u32>) -> vec2<f32> {
    let brick_idx = coord.z * BRICKS_PER_DIM * BRICKS_PER_DIM + 
                    coord.y * BRICKS_PER_DIM + 
                    coord.x;
    
    // Sample center voxel of the brick
    let center_voxel = VOXELS_PER_BRICK / 2u;
    
    let density = sampleChannel(brick_idx, center_voxel, 0u); // density channel
    let cohesion = sampleChannel(brick_idx, center_voxel, 5u); // cohesion channel
    
    return vec2<f32>(density, cohesion);
}

// ── Main Compute Shader ────────────────────────────────────────

@compute @workgroup_size(64)
fn chunk_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chunk_id = gid.x;
    
    // Early exit if beyond total chunks
    if (chunk_id >= TOTAL_BRICKS) {
        return;
    }
    
    // Convert chunk ID to coordinates
    let coord = chunkIdToCoord(chunk_id);
    
    // Get world position
    let world_pos = getChunkWorldPos(coord);
    
    // Calculate distance to camera
    let dist = distance(world_pos, chunk_params.camera_pos);
    
    // Frustum culling
    let chunk_radius = f32(BRICK_SIZE) * 0.866; // Half-diagonal of 16³ cube
    let visible = isInFrustum(world_pos, chunk_radius);
    
    // Calculate LOD level
    let lod = calculateLOD(dist, u32(chunk_params.lod_levels));
    
    // Prefetch radius check: always include chunks near camera
    let in_prefetch = dist < f32(chunk_params.prefetch_radius) * f32(BRICK_SIZE) * 2.0;
    
    // Final visibility decision
    let is_visible = visible || in_prefetch;
    
    if (is_visible) {
        // Atomically increment visible count and get index
        let idx = atomicAdd(&visible_count, 1u);
        
        // Write indirect draw command
        // In production: this would reference per-chunk mesh buffers
        // For now, we write placeholder data
        indirect_draws[idx] = DrawIndirect(
            36u,           // 12 triangles × 3 vertices (placeholder)
            1u,            // One instance per chunk
            idx * 36u,     // First vertex offset
            0u,            // Base vertex
            chunk_id       // Instance ID = chunk ID for shader lookup
        );
    }
}

// ── Utility Entry Points ───────────────────────────────────────

/// Reset visible count before each frame's chunk update
@compute @workgroup_size(1)
fn reset_visible_count() {
    atomicStore(&visible_count, 0u);
}

/// Query visible chunk count (for debugging/stats)
@compute @workgroup_size(1)
fn query_visible_count() {
    // This could write to a query buffer for CPU readback
    // Omitted for brevity — use GPU debug tools instead
}
