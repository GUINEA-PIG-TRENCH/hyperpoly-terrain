# Hyperpoly-Terrain × Trench-Builder Integration Guide

This document describes how to integrate **hyperpoly-terrain**'s 6-channel GPU voxel/tensor substrate with **trench-builder**'s open-world chunk streaming pipeline using shared WebGPU buffers.

## 🏗️ Integration Architecture

```
[ hyperpoly-terrain ] (6-Channel WGSL Compute)
       │
       ├── Voxel/Tensor Buffers (StorageBuffer / Uniforms)
       │
       ▼
[ ChunkManager / Fractype Bridge ] ◄── Prefetch Radius & LOD Culling
       │
       ▼
[ trench-builder ] (Three.js WebGPU Renderer & Infinite Geometry)
```

## 📦 Channel Layout

The 6-channel tensor substrate stores the following data per voxel:

| Channel | Name         | Data Type | Usage                                      |
|---------|--------------|-----------|--------------------------------------------|
| 0       | density      | u16       | Iso-surface extraction (QEF / Marching Cubes) |
| 1       | cohesion     | u16       | Rock vs. loose soil structural stiffness   |
| 2       | permeability | u16       | Vegetation density mask                    |
| 3       | water        | u16       | Specular wetness / darkening factor        |
| 4       | sediment     | u16       | Surface color gradient (sand/silt layer)   |
| 5       | oxidation    | u16       | Weathering and rust/iron mineral tinting   |

## 🔑 Key Integration Steps

### 1. Initialize HyperPolyTerrainCompute

```typescript
import { HyperpolyTerrainCompute } from './game/compute/hyperpoly_terrain_compute.js';

const device = /* your WebGPU device */;

const terrainCompute = new HyperpolyTerrainCompute(device, {
  gridSize: 256,
});

await terrainCompute.init({
  erosion: erosionWGSL,
  darcyFlow: darcyFlowWGSL,
  materialUpdate: materialUpdateWGSL,
  brickActivation: brickActivationWGSL
});

// Upload initial terrain data
await terrainCompute.uploadQuantizedTerrain(shellModelGenerator);
```

### 2. Pass WebGPU Storage Buffers to ChunkManager

Rather than reading volume data back to the CPU, pass the WebGPU `GPUBuffer` reference holding the 6-channel tensor data straight into `ChunkManager`:

```typescript
import { ChunkManager } from './game/compute/chunk_manager.js';

const chunkManager = new ChunkManager(device, {
  chunkSize: 16,      // Must match brick size
  prefetchRadius: 3,  // Synchronized with terrain active brick radius
  lodLevels: 4
});

// Bind 6-channel storage buffer directly to chunk manager
await chunkManager.init(
  { chunkCompute: chunkComputeWGSL },
  terrainCompute.getTensorBuffer(),  // GPUBuffer - zero CPU sync
  terrainCompute.getMetaBuffer()     // GPUBuffer - min/scale metadata
);
```

### 3. Configure Dynamic Prefetch Radius

Match the active brick dispatcher queue from `hyperpoly-terrain` to `trench-builder`'s prefetch radius to eliminate mesh popping during rapid camera movement or erosion updates:

```typescript
// Sync tick updates in your main loop
function onSimulateTick(delta: number) {
  // Run terrain simulation (erosion, Darcy flow, etc.)
  terrainCompute.dispatch(delta);
  
  // Get active brick indices from terrain compute
  const activeBricks = terrainCompute.getActiveBrickIndices();
  
  // Update chunk manager with camera position and active bricks
  chunkManager.updateActiveChunks(
    activeBricks,
    camera.position,    // Float32Array [x, y, z]
    camera.direction    // Float32Array [x, y, z]
  );
}
```

### 4. Use Biome-Aware 6-Channel Material Shaders

In your renderer, sample the material tensor channels in the vertex and fragment WebGPU shaders to drive:

- **Vertex displacement** (density channel via QEF)
- **Moisture darkening** (water channel)
- **Sediment color blending** (sediment + oxidation channels)
- **Vegetation masks** (permeability channel)
- **Structural stiffness visualization** (cohesion channel)

Example shader binding pattern:

```wgsl
// In your render shader
@group(0) @binding(1) var<storage, read> terrain_tensor: array<u16>;
@group(0) @binding(2) var<storage, read> brick_meta: array<BrickMeta>;

fn getMaterialProperties(world_pos: vec3<f32>) -> vec4<f32> {
    let brick_idx = world_to_brick_index(world_pos);
    let voxel_idx = world_to_voxel_index(world_pos);
    
    let density = sampleChannel(brick_idx, voxel_idx, 0u);
    let cohesion = sampleChannel(brick_idx, voxel_idx, 5u);
    let water = sampleChannel(brick_idx, voxel_idx, 3u);
    let sediment = sampleChannel(brick_idx, voxel_idx, 4u);
    
    return vec4<f32>(density, cohesion, water, sediment);
}
```

## ⚠️ Common Pitfalls to Avoid

### Zero CPU-GPU Sync

**❌ WRONG:** Do not call `mapAsync()` or `copyBufferToBuffer` to read back terrain data to the CPU on every frame.

```typescript
// DON'T DO THIS
async function badPattern() {
  await terrainBuffer.mapAsync(GPUMapMode.READ);
  const data = new Uint16Array(terrainBuffer.getMappedRange());
  // ... process on CPU ...
  terrainBuffer.unmap();
}
```

**✅ CORRECT:** Keep the compute-to-render loop entirely on the GPU.

```typescript
// DO THIS
chunkManager.init(wgslSources, terrainCompute.getTensorBuffer(), terrainCompute.getMetaBuffer());
// All data stays on GPU - no mapAsync, no readback
```

### Brick Alignment

Ensure `trench-builder` chunk bounds align with `hyperpoly-terrain` brick boundaries:

```typescript
const BRICK_SIZE = 16;  // 16×16×16 voxels per brick

// ChunkManager must use same brick size
const chunkManager = new ChunkManager(device, {
  chunkSize: BRICK_SIZE,  // ✓ Correct
  // chunkSize: 32,       // ✗ Wrong - misaligned
});
```

### Prefetch Radius Synchronization

The prefetch radius should be synchronized between terrain simulation and chunk streaming:

```typescript
// When terrain simulation modifies bricks within radius R
const SIMULATION_RADIUS = 3;

// ChunkManager prefetch should match or exceed this
chunkManager.setPrefetchRadius(SIMULATION_RADIUS);

// Update active chunks with terrain's modified brick indices
chunkManager.updateActiveChunks(
  terrainCompute.getActiveBrickIndices(),
  cameraPos,
  cameraDir
);
```

## 📊 Performance Considerations

### VRAM Usage

For a 256³ grid with 16³ bricks:

| Buffer Type          | Size Calculation                | Approximate Size |
|----------------------|---------------------------------|------------------|
| Channel Buffers (6×) | 6 × 4096 bricks × 4096 voxels × 2B | ~192 MB         |
| Metadata Buffer      | 4096 bricks × 6 channels × 16B     | ~0.4 MB         |
| **Total**            |                                  | **~192.4 MB**   |

### Dispatch Patterns

- **Full grid dispatch**: 16×16×16 workgroups for brick-level operations
- **Tiled extraction**: Process bricks in tiles to reduce staging buffer VRAM
- **Incremental updates**: Use delta buffers for modified bricks only

## 📁 File Reference

| File                                      | Purpose                                          |
|-------------------------------------------|--------------------------------------------------|
| `game/compute/hyperpoly_terrain_compute.js` | Main 6-channel terrain compute class            |
| `game/compute/chunk_manager.js`           | Open-world chunk streaming manager              |
| `game/compute/chunk_compute.wgsl`         | GPU-driven chunk visibility & LOD shader        |
| `game/compute/hyperpoly_geology.js`       | Base geology buffer management                  |
| `game/render/terrain_renderer.js`         | WebGPU terrain mesh renderer                    |
| `game/render/render_terrain.wgsl`         | Terrain rendering shader with material coloring |

## 🔗 Related Documentation

- See `ARCHITECTURE-COMPARISON.md` for system overview
- See `AAA-GAP-ANALYSIS.md` for feature completeness analysis
- See `ROADMAP.md` for upcoming integration milestones
