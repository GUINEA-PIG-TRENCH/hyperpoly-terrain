# HyperPoly Terrain: Zero-Sync WebGPU Benchmark

**Interactive POC demonstrating GPU-native terrain simulation with zero CPU-GPU synchronization.**

## 🎯 What This Demonstrates

This single-file benchmark proves the core `hyperpoly-terrain` architecture:
- **6-Channel Material Tensors**: Height, velocity (X/Y), water depth, sediment load, material type
- **Zero-Sync Compute Pipeline**: All physics (fluid flow, erosion, deposition) execute on GPU via WGSL compute shaders
- **Direct Render Pass**: Fragment shader reads directly from compute buffers—no CPU round-trips
- **Real-Time Interaction**: Mouse-driven water injection and terrain carving with immediate visual feedback

## 🚀 Quick Start

1. **Open in Browser**: Requires Chrome 113+, Edge 113+, or any WebGPU-enabled browser
2. **Navigate to**: `examples/raw-webgpu/index.html`
3. **Interact**:
   - **Left-Click + Drag**: Inject water pulse at cursor location
   - **Right-Click + Drag**: Carve terrain (decrease height, add sediment)
   - **Reset Button**: Regenerate initial terrain noise

## 📊 Benchmark Metrics

The HUD displays real-time performance data:
- **FPS**: Target 60 FPS locked
- **Frame Time**: Should stay under 16.67ms
- **Grid Resolution**: 512×512 (262,144 cells)
- **GPU Memory**: ~6.3 MB for tensor storage
- **Sync Status**: ZERO-SYNC (no CPU readback during simulation)

## 🏗️ Architecture Overview

### Compute Shader (`computePhysics`)
```wgsl
struct MaterialTensor {
    height : f32,      // Channel 0: Terrain elevation
    velocityX : f32,   // Channel 1: Fluid velocity X
    velocityY : f32,   // Channel 2: Fluid velocity Y
    water : f32,       // Channel 3: Water depth
    sediment : f32,    // Channel 4: Suspended sediment
    material : f32     // Channel 5: Material classification
};
```

**Physics Steps per Frame:**
1. **Mouse Interaction**: Inject water or carve terrain based on cursor position
2. **Fluid Advection**: Shallow water approximation with neighbor sampling
3. **Velocity Integration**: Gravity-driven flow with viscosity damping
4. **Erosion/Deposition**: Sediment capacity model based on flow speed

### Fragment Shader (`fsMain`)
Maps tensor channels to RGB colors:
- **Height**: Sand → Grass → Rock → Snow gradient
- **Water**: Semi-transparent blue overlay (depth-based)
- **Sediment**: Brown tint for high sediment load

## 🎨 Visual Enhancements (Phase 1 Complete)

- **Dynamic Color Grading**: Biome-aware terrain coloring
- **Water Transparency**: Depth-based alpha blending
- **Sediment Visualization**: Visible deposition trails
- **Cyber-Styled HUD**: Real-time metrics overlay

## 🖱️ Mouse Interaction (Phase 1 Complete)

- **Raycast-to-Grid**: Screen coordinates → tensor indices
- **Brush System**: Radial influence falloff for natural interaction
- **Dual-Mode**: Water injection (left-click) vs. carving (right-click)
- **Live Feedback**: Cursor info panel shows grid coordinates and mode

## 📈 Production Roadmap

| Feature | Status | Priority |
|---------|--------|----------|
| 512×512 Grid | ✅ Complete | — |
| 6-Channel Tensors | ✅ Complete | — |
| Zero-Sync Pipeline | ✅ Complete | — |
| Mouse Interaction | ✅ Complete | — |
| Dynamic Coloring | ✅ Complete | — |
| 1024×1024 Grid | 🔲 Pending | High |
| Multi-Layer Erosion | 🔲 Pending | High |
| Texture Splattering | 🔲 Pending | Medium |
| Compute-Driven Mesh | 🔲 Pending | Medium |
| Post-Processing Bloom | 🔲 Pending | Low |

## 🔧 Technical Specifications

- **API**: Pure WebGPU (no Three.js, Babylon, or other libraries)
- **Shaders**: WGSL (WebGPU Shading Language)
- **Workgroup Size**: 8×8 (64 threads per dispatch)
- **Dispatch Count**: 4,096 workgroups per frame (512×512 / 8×8)
- **Buffer Layout**: Structured Storage Buffer (SSBO)
- **Render Topology**: Triangle Strip (full-screen quad)

## 📹 Recording the 30-Second Benchmark Video

**Recommended Sequence:**
1. **0:00–0:05**: Show idle terrain with FPS counter at 60
2. **0:05–0:15**: Left-click water injection—demonstrate fluid flow and erosion
3. **0:15–0:20**: Right-click carving—show immediate terrain deformation
4. **0:20–0:25**: Toggle between modes rapidly—prove zero-latency response
5. **0:25–0:30**: Zoom out (if implemented) or reset—end on stable 60 FPS

**Pro Tip**: Record with OBS at 60 FPS, include the HUD in-frame to prove performance claims.

## 📄 License

Proprietary — Guinea Pig Trench LLC. All rights reserved.

---

**Next Steps**: After recording the benchmark video, pivot to **Strategy B (Commercial)** by creating enterprise-facing materials targeting LiDAR calibration, drone swarm coordination, and digital twin applications.
