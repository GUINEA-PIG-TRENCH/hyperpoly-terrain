# hyperpoly-terrain: Zero-Sync WebGPU Benchmark

## Overview
A minimal, self-contained Proof of Concept demonstrating the core architecture of hyperpoly-terrain: a GPU-native terrain engine that eliminates CPU bottlenecks by maintaining all simulation state exclusively on the GPU through 6-channel material tensors.

## Key Features Demonstrated

### Zero-Sync Architecture
- **No CPU Simulation Loop**: All physics computations execute via WebGPU compute shaders
- **Direct Render Pipeline**: Rendering reads directly from compute buffers with zero CPU round-trips
- **6-Channel Material Tensors**: Each grid cell stores height, velocity (X/Y), sediment, moisture, and material type

### Technical Specifications
- **Grid Resolution**: 512×512 (262,144 cells)
- **Memory Footprint**: ~6.3 MB GPU memory
- **Workgroup Size**: 8×8 threads (optimized for most GPUs)
- **Compute Dispatches**: ~4,096 workgroups per frame

### WGSL Shaders Included
1. **Compute Shader**: Hydraulic erosion, fluid advection, sediment transport
2. **Render Shader**: Height-based terrain coloring with flow visualization

## Running the Demo

### Requirements
- Chrome 113+ or Edge 113+ (WebGPU enabled by default)
- Firefox Nightly (with `dom.webgpu.enabled` in about:config)
- A discrete GPU recommended for best performance

### Instructions
1. Open `index.html` in a WebGPU-capable browser
2. Observe the FPS counter and real-time statistics
3. Use the control buttons to:
   - Toggle hydraulic erosion on/off
   - Add fluid pulses to the terrain
   - Reset the simulation

### Expected Performance
- **Target FPS**: 60 FPS locked on modern GPUs
- **Frame Time**: <16ms per frame
- **CPU Usage**: Minimal (<5% single core)

## Architecture Highlights

### Material Tensor Layout
```
Channel 0: Height (elevation 0.0-1.0)
Channel 1: Velocity X (fluid flow direction)
Channel 2: Velocity Y (fluid flow direction)  
Channel 3: Sediment (erosion material carried)
Channel 4: Moisture (water saturation level)
Channel 5: Material Type (rock/soil/sand bitmask)
```

### Simulation Pipeline
1. **Initialize**: CPU creates GPU buffers with procedural terrain
2. **Compute Pass**: WGSL shader updates all tensor channels in parallel
3. **Render Pass**: Vertex shader reads directly from compute buffer
4. **Repeat**: No CPU intervention required during simulation

### Physics Implemented
- Shallow water approximation for fluid flow
- Semi-Lagrangian advection for mass conservation
- Hydraulic erosion with sediment transport
- Sediment deposition based on flow velocity
- Boundary condition handling

## Benchmark Metrics

The demo displays real-time statistics:
- **FPS**: Frames per second (target: 60)
- **Frame Time**: Milliseconds per frame (target: <16ms)
- **Grid Size**: Terrain resolution
- **Compute Dispatches**: Total simulation steps executed
- **Simulation Time**: Elapsed simulation time
- **GPU Memory**: Buffer allocation size

## Next Steps for Production

To extend this POC for production use:

1. **Multi-pass Compute**: Separate erosion, fluid, and deposition passes
2. **Sparse Streaming**: Only update active regions for larger terrains
3. **Texture Sampling**: Use storage textures instead of buffers for better cache locality
4. **Instanced Rendering**: Convert point-list to triangle mesh for smoother visuals
5. **Interaction Layer**: Add mouse/touch input via compute buffer writes
6. **LOD System**: Implement level-of-detail for planetary-scale rendering

## License

Proprietary - Part of hyperpoly-terrain Engine

---

*This benchmark demonstrates the viability of zero-sync WebGPU terrain simulation for enterprise applications in Web-CAD, Digital Twins, and GIS visualization.*
