import { quantizeBrick } from './geology_quantizer.js';

export class HyperPolyGeology {
  constructor(device, gridSize = 256) {
    this.device = device;
    this.gridSize = gridSize;
    this.brickSize = 16;
    this.bricksPerDim = gridSize / this.brickSize; // 16
    this.totalBricks = this.bricksPerDim ** 3;     // 4096
    this.channels = 6;
    this.voxelsPerBrick = this.brickSize ** 3;     // 4096

    this._createBuffers();
  }

  _createBuffers() {
    const channelByteSize = this.totalBricks * this.voxelsPerBrick * 2; // u16 = 2B

    // 1. SoA channel buffers (quantized u16)
    this.channelBuffers = Array.from({ length: this.channels }, () =>
      this.device.createBuffer({
        size: channelByteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      })
    );

    // 2. Metadata buffer: array<vec4<f32>>[totalBricks * channels]
    this.metaBuffer = this.device.createBuffer({
      size: this.totalBricks * this.channels * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // 3. Brick flags buffer: u32 per brick for active/visible state
    this.brickFlagsBuffer = this.device.createBuffer({
      size: this.totalBricks * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
  }

  async   stepSimulation(encoder) {
    // Stub: dispatch passes that would normally advance simulation state.
    // The edit manager handles terrain modifications; future passes
    // will handle hydraulic/thermal simulation.
    const pass = encoder.beginComputePass();
    pass.end();
  }

  async uploadQuantizedTerrain(shellModelGenerator) {
    // shellModelGenerator(bx, by, bz) -> Float32Array[6]
    const metaHost = new Float32Array(this.totalBricks * this.channels * 4);
    const queue = this.device.queue;

    for (let bz = 0; bz < this.bricksPerDim; bz++) {
      for (let by = 0; by < this.bricksPerDim; by++) {
        for (let bx = 0; bx < this.bricksPerDim; bx++) {
          const brickIdx = (bz * this.bricksPerDim * this.bricksPerDim) +
                           (by * this.bricksPerDim) + bx;
          
          const srcChannels = shellModelGenerator(bx, by, bz);
          const { buffers, meta } = quantizeBrick(srcChannels, this.brickSize);

          // Upload channels (TypedArray direct write)
          for (let c = 0; c < this.channels; c++) {
            const offset = brickIdx * this.voxelsPerBrick * 2;
            queue.writeBuffer(this.channelBuffers[c], offset, buffers[c]);
          }

          // Pack metadata
          for (let c = 0; c < this.channels; c++) {
            const baseIdx = (brickIdx * this.channels + c) * 4;
            metaHost[baseIdx]     = meta[c * 2];     // min
            metaHost[baseIdx + 1] = meta[c * 2 + 1]; // scale
            metaHost[baseIdx + 2] = 0.0;
            metaHost[baseIdx + 3] = 0.0;
          }
        }
      }
    }

    queue.writeBuffer(this.metaBuffer, 0, metaHost);
    await queue.onSubmittedWorkDone();
    console.log(`Quantized terrain uploaded. ${this.totalBricks} bricks, ~${(this.totalBricks * this.channels * this.voxelsPerBrick * 2 / 1024 / 1024).toFixed(0)}MB VRAM`);
  }
}
