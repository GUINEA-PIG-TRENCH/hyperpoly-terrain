/**
 * CHUNK MANAGER — Open-World Chunk Streaming for Trench-Builder
 * 
 * Integrates hyperpoly-terrain's 6-channel GPU voxel/tensor substrate
 * with trench-builder's infinite geometry streaming pipeline.
 * 
 * Architecture:
 *   [HyperPolyTerrain Compute] → [6-Channel Storage Buffers]
 *           ↓
 *   [ChunkManager] — Prefetch radius & LOD culling
 *           ↓
 *   [Trench-Builder Renderer] — Three.js WebGPU / Custom WebGPU
 * 
 * Key Features:
 *   - Zero CPU-GPU sync: StorageBuffer passed directly to shaders
 *   - Brick-aligned chunks (16³ voxels per brick)
 *   - Dynamic prefetch radius synchronized with active brick dispatcher
 *   - Biome-aware 6-channel material sampling
 */

export class ChunkManager {
  constructor(device, options = {}) {
    this.device = device;
    
    // Chunk configuration
    this.chunkSize = options.chunkSize || 16;        // voxels per chunk dimension
    this.prefetchRadius = options.prefetchRadius || 3; // bricks ahead of camera
    this.lodLevels = options.lodLevels || 4;
    
    // Grid configuration (must match hyperpoly-terrain)
    this.gridSize = options.gridSize || 256;
    this.brickSize = 16;
    this.bricksPerDim = this.gridSize / this.brickSize; // 16
    
    // Active chunks tracking
    this.activeChunks = new Map(); // key: chunkId, value: ChunkData
    this.visibleChunks = [];
    this.pendingLoads = new Set();
    
    // Camera tracking for frustum/prefetch
    this.cameraPosition = [0, 10, 20];
    this.cameraDirection = [0, 0, -1];
    
    // Statistics
    this.stats = {
      activeChunks: 0,
      visibleChunks: 0,
      prefetchedBricks: 0,
      lastUpdateTime: 0
    };
    
    console.log(`ChunkManager initialized: chunkSize=${this.chunkSize}, prefetchRadius=${this.prefetchRadius}, lodLevels=${this.lodLevels}`);
  }
  
  /**
   * Initialize the chunk manager with WGSL shader modules
   * @param {Object} wgslSources - Shader source code object
   * @param {GPUBuffer} tensorBuffer - 6-channel storage buffer from HyperPolyTerrain
   * @param {GPUBuffer} metaBuffer - Metadata buffer (min/scale per channel per brick)
   */
  async init(wgslSources, tensorBuffer, metaBuffer) {
    const device = this.device;
    
    // Create chunk compute pipeline for LOD selection and frustum culling
    this.chunkComputePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: wgslSources.chunkCompute }),
        entryPoint: 'chunk_update'
      }
    });
    
    // Create indirect draw buffer for chunk rendering
    this.indirectDrawBufferSize = 10000 * 20; // Max 10K chunks × 5 u32 per draw
    this.indirectDrawBuffer = device.createBuffer({
      size: this.indirectDrawBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT
    });
    
    // Visible chunk count buffer (for indirect draw)
    this.visibleCountBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    // Uniform buffer for chunk update params
    this.chunkUniformBuffer = device.createBuffer({
      size: 64, // camera pos (12) + direction (12) + prefetch radius (4) + padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Store references to terrain compute buffers
    this.tensorBuffer = tensorBuffer;
    this.metaBuffer = metaBuffer;
    
    // Create bind group for chunk updates
    this.chunkBindGroup = device.createBindGroup({
      layout: this.chunkComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.chunkUniformBuffer } },
        { binding: 1, resource: { buffer: this.tensorBuffer } },
        { binding: 2, resource: { buffer: this.metaBuffer } },
        { binding: 3, resource: { buffer: this.indirectDrawBuffer } },
        { binding: 4, resource: { buffer: this.visibleCountBuffer } },
      ]
    });
    
    console.log('ChunkManager: WebGPU pipelines initialized');
  }
  
  /**
   * Update active chunks based on camera position and hyperpoly-terrain brick indices
   * @param {Uint32Array} activeBrickIndices - From HyperPolyTerrain.getActiveBrickIndices()
   * @param {Float32Array} cameraPos - Current camera world position
   * @param {Float32Array} cameraDir - Normalized camera forward vector
   */
  updateActiveChunks(activeBrickIndices, cameraPos, cameraDir) {
    const now = performance.now();
    
    // Update camera tracking
    this.cameraPosition = Array.from(cameraPos);
    this.cameraDirection = Array.from(cameraDir);
    
    // Calculate which chunks should be active based on:
    // 1. Camera frustum
    // 2. Prefetch radius in movement direction
    // 3. Active bricks from terrain simulation
    
    const newActiveSet = new Set();
    const camChunk = this.worldToChunkCoord(cameraPos[0], cameraPos[1], cameraPos[2]);
    
    // Add chunks within prefetch radius
    for (let dz = -this.prefetchRadius; dz <= this.prefetchRadius; dz++) {
      for (let dy = -this.prefetchRadius; dy <= this.prefetchRadius; dy++) {
        for (let dx = -this.prefetchRadius; dx <= this.prefetchRadius; dx++) {
          const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (dist > this.prefetchRadius) continue;
          
          const cx = camChunk[0] + dx;
          const cy = camChunk[1] + dy;
          const cz = camChunk[2] + dz;
          
          // Clamp to world bounds
          if (cx < 0 || cx >= this.bricksPerDim ||
              cy < 0 || cy >= this.bricksPerDim ||
              cz < 0 || cz >= this.bricksPerDim) continue;
          
          const chunkId = this.chunkIdFromCoord(cx, cy, cz);
          newActiveSet.add(chunkId);
        }
      }
    }
    
    // Merge with active bricks from terrain simulation
    if (activeBrickIndices) {
      const numBricks = activeBrickIndices.length / 3;
      for (let i = 0; i < numBricks; i++) {
        const bx = activeBrickIndices[i * 3];
        const by = activeBrickIndices[i * 3 + 1];
        const bz = activeBrickIndices[i * 3 + 2];
        const chunkId = this.chunkIdFromCoord(bx, by, bz);
        newActiveSet.add(chunkId);
      }
    }
    
    // Remove chunks that are no longer needed
    for (const [chunkId, chunkData] of this.activeChunks) {
      if (!newActiveSet.has(chunkId)) {
        this._unloadChunk(chunkId);
      }
    }
    
    // Load new chunks
    for (const chunkId of newActiveSet) {
      if (!this.activeChunks.has(chunkId) && !this.pendingLoads.has(chunkId)) {
        this._loadChunk(chunkId);
      }
    }
    
    // Update statistics
    this.stats.activeChunks = this.activeChunks.size;
    this.stats.visibleChunks = this.visibleChunks.length;
    this.stats.lastUpdateTime = now;
  }
  
  /**
   * Dispatch chunk compute shader to update visibility and LOD
   * Called each frame before rendering
   */
  dispatchChunkUpdate(encoder) {
    // Update uniform buffer with camera data
    const uniformData = new Float32Array(16);
    uniformData.set(this.cameraPosition, 0);
    uniformData.set(this.cameraDirection, 4);
    uniformData[8] = this.prefetchRadius;
    uniformData[9] = this.lodLevels;
    this.device.queue.writeBuffer(this.chunkUniformBuffer, 0, uniformData);
    
    // Dispatch chunk compute pass
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.chunkComputePipeline);
    pass.setBindGroup(0, this.chunkBindGroup);
    
    // Dispatch workgroups: one per potential chunk (16³ = 4096 max)
    const maxChunks = this.bricksPerDim ** 3;
    const workgroupSize = 64;
    const workgroupCount = Math.ceil(maxChunks / workgroupSize);
    pass.dispatchWorkgroups(workgroupCount, 1, 1);
    pass.end();
  }
  
  /**
   * Get the list of visible chunks for rendering
   * @returns {Array} Array of chunk data objects
   */
  getVisibleChunks() {
    return this.visibleChunks;
  }
  
  /**
   * Get indirect draw buffer for GPU-driven chunk rendering
   * @returns {GPUBuffer}
   */
  getIndirectDrawBuffer() {
    return this.indirectDrawBuffer;
  }
  
  /**
   * Synchronize prefetch radius with hyperpoly-terrain active brick radius
   * @param {number} radius - New prefetch radius in bricks
   */
  setPrefetchRadius(radius) {
    this.prefetchRadius = Math.max(1, Math.min(8, radius));
    console.log(`ChunkManager: prefetch radius updated to ${this.prefetchRadius}`);
  }
  
  /**
   * Get chunk statistics
   * @returns {Object} Stats object
   */
  getStats() {
    return { ...this.stats };
  }
  
  // ── Private Methods ───────────────────────────────────────────
  
  _loadChunk(chunkId) {
    const [cx, cy, cz] = this.chunkCoordFromId(chunkId);
    
    // Mark as pending
    this.pendingLoads.add(chunkId);
    
    // Create chunk data structure
    const chunkData = {
      id: chunkId,
      coord: [cx, cy, cz],
      worldOffset: [
        cx * this.brickSize,
        cy * this.brickSize,
        cz * this.brickSize
      ],
      loaded: false,
      visible: false,
      lod: 0,
      meshBuffer: null,
      loadTime: performance.now()
    };
    
    // In a real implementation, this would trigger async mesh extraction
    // For GPU-only approach, we just register the chunk
    this.activeChunks.set(chunkId, chunkData);
    this.pendingLoads.delete(chunkId);
    chunkData.loaded = true;
    
    console.log(`ChunkManager: Loaded chunk (${cx},${cy},${cz})`);
  }
  
  _unloadChunk(chunkId) {
    const chunkData = this.activeChunks.get(chunkId);
    if (chunkData) {
      // Clean up GPU resources
      if (chunkData.meshBuffer) {
        chunkData.meshBuffer.destroy();
      }
      
      this.activeChunks.delete(chunkId);
      const idx = this.visibleChunks.indexOf(chunkData);
      if (idx !== -1) {
        this.visibleChunks.splice(idx, 1);
      }
      
      console.log(`ChunkManager: Unloaded chunk ${chunkId}`);
    }
  }
  
  chunkIdFromCoord(x, y, z) {
    return `${x},${y},${z}`;
  }
  
  chunkCoordFromId(id) {
    return id.split(',').map(Number);
  }
  
  worldToChunkCoord(wx, wy, wz) {
    const vx = Math.floor(wx / this.brickSize);
    const vy = Math.floor(wy / this.brickSize);
    const vz = Math.floor(wz / this.brickSize);
    return [
      Math.max(0, Math.min(this.bricksPerDim - 1, vx)),
      Math.max(0, Math.min(this.bricksPerDim - 1, vy)),
      Math.max(0, Math.min(this.bricksPerDim - 1, vz))
    ];
  }
  
  destroy() {
    // Unload all chunks
    for (const chunkId of Array.from(this.activeChunks.keys())) {
      this._unloadChunk(chunkId);
    }
    
    // Destroy GPU buffers
    if (this.indirectDrawBuffer) this.indirectDrawBuffer.destroy();
    if (this.visibleCountBuffer) this.visibleCountBuffer.destroy();
    if (this.chunkUniformBuffer) this.chunkUniformBuffer.destroy();
    
    console.log('ChunkManager: Destroyed');
  }
}
