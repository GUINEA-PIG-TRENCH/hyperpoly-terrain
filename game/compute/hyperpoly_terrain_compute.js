/**
 * HYPERPOLY-TERRAIN COMPUTE — 6-Channel Voxel/Tensor Substrate
 * 
 * This module provides the GPU compute backend for hyperpoly-terrain,
 * exposing the 6-channel tensor buffers for integration with trench-builder.
 * 
 * Channels:
 *   0. density     - Iso-surface extraction (QEF / Marching Cubes)
 *   1. cohesion    - Rock vs. soil structural stiffness
 *   2. permeability - Vegetation density mask
 *   3. water       - Specular wetness / darkening factor
 *   4. sediment    - Surface color gradient (sand/silt layer)
 *   5. oxidation   - Weathering and rust/iron mineral tinting
 * 
 * Integration Pattern:
 *   const terrain = new HyperpolyTerrainCompute(device);
 *   await terrain.init(wgslSources);
 *   
 *   // Pass buffers directly to ChunkManager (zero CPU sync)
 *   chunkManager.init(wgslSources, terrain.getTensorBuffer(), terrain.getMetaBuffer());
 */

import { HyperPolyGeology } from './hyperpoly_geology.js';

export class HyperpolyTerrainCompute extends HyperPolyGeology {
  constructor(device, options = {}) {
    super(device, options.gridSize || 256);
    
    this.brickSize = 16;
    this.bricksPerDim = this.gridSize / this.brickSize;
    this.totalBricks = this.bricksPerDim ** 3;
    
    // Active brick tracking (for synchronization with ChunkManager)
    this.activeBrickIndices = new Uint32Array(this.totalBricks * 3);
    this.activeBrickCount = 0;
    
    // Simulation state
    this.simulationTime = 0;
    this.lastDispatchTime = 0;
    
    console.log(`HyperpolyTerrainCompute initialized: ${this.gridSize}³ grid, ${this.totalBricks} bricks`);
  }
  
  /**
   * Initialize compute pipelines for erosion, Darcy flow, and material updates
   * @param {Object} wgslSources - Shader source code object
   */
  async init(wgslSources) {
    const device = this.device;
    
    // Erosion compute pipeline (hydraulic + thermal)
    this.erosionPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: wgslSources.erosion }),
        entryPoint: 'erosion_step'
      }
    });
    
    // Darcy flow pipeline (groundwater simulation)
    this.darcyPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: wgslSources.darcyFlow }),
        entryPoint: 'darcy_flow'
      }
    });
    
    // Material update pipeline (sediment transport, oxidation)
    this.materialPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: wgslSources.materialUpdate }),
        entryPoint: 'material_update'
      }
    });
    
    // Brick activation pipeline (tracks modified bricks)
    this.brickActivationPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: wgslSources.brickActivation }),
        entryPoint: 'brick_activation'
      }
    });
    
    // Create bind groups
    this._createBindGroups();
    
    console.log('HyperpolyTerrainCompute: Compute pipelines initialized');
  }
  
  _createBindGroups() {
    const device = this.device;
    
    // Erosion bind group
    this.erosionBindGroup = device.createBindGroup({
      layout: this.erosionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.metaBuffer } },
        { binding: 1, resource: { buffer: this.channelBuffers[0] } }, // density
        { binding: 2, resource: { buffer: this.channelBuffers[5] } }, // cohesion
        { binding: 3, resource: { buffer: this.channelBuffers[2] } }, // permeability
        { binding: 4, resource: { buffer: this.channelBuffers[3] } }, // water
      ]
    });
    
    // Darcy flow bind group
    this.darcyBindGroup = device.createBindGroup({
      layout: this.darcyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.metaBuffer } },
        { binding: 1, resource: { buffer: this.channelBuffers[2] } }, // permeability
        { binding: 2, resource: { buffer: this.channelBuffers[3] } }, // water
        { binding: 3, resource: { buffer: this.channelBuffers[4] } }, // sediment
      ]
    });
    
    // Material update bind group
    this.materialBindGroup = device.createBindGroup({
      layout: this.materialPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.metaBuffer } },
        { binding: 1, resource: { buffer: this.channelBuffers[0] } }, // density
        { binding: 2, resource: { buffer: this.channelBuffers[4] } }, // sediment
        { binding: 3, resource: { buffer: this.channelBuffers[5] } }, // oxidation
      ]
    });
    
    // Brick activation bind group
    this.brickActivationBindGroup = device.createBindGroup({
      layout: this.brickActivationPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.metaBuffer } },
        { binding: 1, resource: { buffer: this.channelBuffers[0] } }, // density
        { binding: 2, resource: { buffer: this.activeBrickBuffer } },
      ]
    });
  }
  
  /**
   * Dispatch all compute passes for one simulation step
   * @param {number} delta - Time delta in seconds
   */
  dispatch(delta = 0.016) {
    const device = this.device;
    const encoder = device.createCommandEncoder();
    
    // Update time uniform (if needed by shaders)
    this.simulationTime += delta;
    
    // Pass 1: Erosion (hydraulic + thermal weathering)
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.erosionPipeline);
      pass.setBindGroup(0, this.erosionBindGroup);
      pass.dispatchWorkgroups(this.bricksPerDim, this.bricksPerDim, this.bricksPerDim);
      pass.end();
    }
    
    // Pass 2: Darcy flow (groundwater percolation)
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.darcyPipeline);
      pass.setBindGroup(0, this.darcyBindGroup);
      pass.dispatchWorkgroups(this.bricksPerDim, this.bricksPerDim, this.bricksPerDim);
      pass.end();
    }
    
    // Pass 3: Material updates (sediment transport, oxidation)
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.materialPipeline);
      pass.setBindGroup(0, this.materialBindGroup);
      pass.dispatchWorkgroups(this.bricksPerDim, this.bricksPerDim, this.bricksPerDim);
      pass.end();
    }
    
    // Pass 4: Brick activation tracking
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.brickActivationPipeline);
      pass.setBindGroup(0, this.brickActivationBindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.totalBricks / 64), 1, 1);
      pass.end();
    }
    
    device.queue.submit([encoder.finish()]);
    this.lastDispatchTime = performance.now();
  }
  
  /**
   * Get active brick indices after simulation step
   * Used by ChunkManager to synchronize prefetch radius
   * @returns {Uint32Array} Flat array of [bx, by, bz, bx, by, bz, ...]
   */
  getActiveBrickIndices() {
    // In production: read back from GPU via mapping or query buffer
    // For now, return cached indices (would be updated by brick_activation shader)
    return this.activeBrickIndices.subarray(0, this.activeBrickCount * 3);
  }
  
  /**
   * Get the 6-channel tensor storage buffer
   * Pass this directly to ChunkManager or renderer (zero CPU sync)
   * @returns {GPUBuffer}
   */
  getTensorBuffer() {
    // Return first channel buffer as representative
    // In production: create a combined view or use texture storage
    return this.channelBuffers[0];
  }
  
  /**
   * Get individual channel buffer by index
   * @param {number} channelIndex - 0-5
   * @returns {GPUBuffer}
   */
  getChannelBuffer(channelIndex) {
    if (channelIndex < 0 || channelIndex >= this.channels) {
      throw new Error(`Invalid channel index: ${channelIndex}`);
    }
    return this.channelBuffers[channelIndex];
  }
  
  /**
   * Get metadata buffer (min/scale per channel per brick)
   * @returns {GPUBuffer}
   */
  getMetaBuffer() {
    return this.metaBuffer;
  }
  
  /**
   * Get simulation statistics
   * @returns {Object}
   */
  getStats() {
    return {
      gridSize: this.gridSize,
      totalBricks: this.totalBricks,
      activeBricks: this.activeBrickCount,
      simulationTime: this.simulationTime,
      lastDispatchTime: this.lastDispatchTime,
      vramUsage: this._calculateVRAMUsage()
    };
  }
  
  _calculateVRAMUsage() {
    // Channel buffers: 6 × (totalBricks × voxelsPerBrick × 2 bytes)
    const channelSize = this.totalBricks * (this.brickSize ** 3) * 2;
    const totalChannelBytes = this.channels * channelSize;
    
    // Meta buffer: totalBricks × channels × 16 bytes
    const metaBytes = this.totalBricks * this.channels * 16;
    
    const totalBytes = totalChannelBytes + metaBytes;
    return {
      channels: totalChannelBytes / 1024 / 1024,
      metadata: metaBytes / 1024 / 1024,
      total: totalBytes / 1024 / 1024
    };
  }
  
  destroy() {
    // Destroy all buffers
    for (const buf of this.channelBuffers) {
      buf.destroy();
    }
    this.metaBuffer.destroy();
    
    console.log('HyperpolyTerrainCompute: Destroyed');
  }
}
