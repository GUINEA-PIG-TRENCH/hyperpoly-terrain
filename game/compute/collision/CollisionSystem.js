/**
 * COLLISION SYSTEM — Host integration for HyperPolyEngine
 * 
 * Manages raycast and sphere-cast queries against the dual mesh.
 * Reads directly from the QEF vertex buffer — no separate BVH.
 * Queries execute as GPU compute dispatches, no CPU readback.
 */

import { readGPUBuffer } from '../examples/utils/buffer-sync.js';

export class CollisionSystem {
  constructor(device) {
    this.device = device;

    // Hit output buffer: up to 64 concurrent queries
    // HitRecord stride = 64 bytes (WGSL 16-byte vec3 alignment)
    this.hitBuffer = device.createBuffer({
      size: 64 * 64, // 64 HitRecords × 64 bytes each
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Ray uniform buffer — padded for WGSL vec3 16-byte alignment
    this.rayBuffer = device.createBuffer({
      size: 10 * 4, // origin(4 floats) + direction(4 floats) + tMin(1) + tMax(1)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Sphere uniform buffer — padded for WGSL vec3 16-byte alignment
    this.sphereBuffer = device.createBuffer({
      size: 11 * 4, // center(4 floats) + radius(1) + pad(3) + velocity(4 floats)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Readback staging for results
    this.readbackBuffer = device.createBuffer({
      size: 64, // Single HitRecord (64 bytes)
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.pipelines = {};
    this.bindGroups = {};
    this.isInitialized = false;
  }

  async init(wgslSrcQuery, wgslSrcSphere, qefBuffer, hermiteBuffer, densityBuffer, brickFlagsBuffer) {
    const device = this.device;

    // Raycast pipeline
    this.pipelines.raycast = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: wgslSrcQuery }),
        entryPoint: 'raycast_pass',
      },
    });

    // Sphere-cast pipeline
    this.pipelines.spherecast = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: wgslSrcSphere }),
        entryPoint: 'sphere_cast_pass',
      },
    });

    // Raycast bind group (group 0)
    this.bindGroups.raycast = device.createBindGroup({
      layout: this.pipelines.raycast.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qefBuffer } },
        { binding: 1, resource: { buffer: hermiteBuffer } },
        { binding: 2, resource: { buffer: densityBuffer } },
        { binding: 3, resource: { buffer: brickFlagsBuffer } },
        { binding: 4, resource: { buffer: this.hitBuffer } },
        { binding: 5, resource: { buffer: this.rayBuffer } },
      ],
    });

    // Sphere-cast bind group (group 1)
    this.bindGroups.spherecast = device.createBindGroup({
      layout: this.pipelines.spherecast.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: qefBuffer } },
        { binding: 1, resource: { buffer: hermiteBuffer } },
        { binding: 2, resource: { buffer: densityBuffer } },
        { binding: 3, resource: { buffer: this.hitBuffer } },
        { binding: 4, resource: { buffer: this.sphereBuffer } },
      ],
    });

    this.isInitialized = true;
  }

  /**
   * Raycast from origin in direction. Returns first hit.
   * Dispatches GPU compute, reads back result.
   */
  async raycast(origin, direction, tMin = 0.0, tMax = 500.0) {
    // Write ray params with WGSL vec3 alignment padding
    const buf = new ArrayBuffer(10 * 4); // 40 bytes
    const f32 = new Float32Array(buf);
    f32[0] = origin[0]; f32[1] = origin[1]; f32[2] = origin[2]; f32[3] = 0; // pad
    f32[4] = direction[0]; f32[5] = direction[1]; f32[6] = direction[2]; f32[7] = 0; // pad
    f32[8] = tMin; f32[9] = tMax;
    this.device.queue.writeBuffer(this.rayBuffer, 0, buf);

    // Dispatch
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.raycast);
    pass.setBindGroup(0, this.bindGroups.raycast);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();

    // Copy hit to readback
    encoder.copyBufferToBuffer(this.hitBuffer, 0, this.readbackBuffer, 0, 64);
    this.device.queue.submit([encoder.finish()]);

    // Readback
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuffer.getMappedRange().slice(0, 64));
    this.readbackBuffer.unmap();

    const hit = {
      hit: data[0] !== 0,
      pos: [data[4], data[5], data[6]],
      normal: [data[8], data[9], data[10]],
      materialType: data[12],
      t: data[13],
    };

    return hit;
  }

  /**
   * Sphere-cast: sweep a sphere through the grid along a velocity vector.
   * Returns the first obstruction and push-out vector.
   */
  async sphereCast(center, radius, velocity) {
    // Write sphere params with WGSL vec3 alignment padding
    const buf = new ArrayBuffer(11 * 4); // 44 bytes
    const f32 = new Float32Array(buf);
    f32[0] = center[0]; f32[1] = center[1]; f32[2] = center[2]; f32[3] = 0; // pad
    f32[4] = radius;
    f32[5] = 0; f32[6] = 0; f32[7] = 0; // pad to 16-byte alignment
    f32[8] = velocity[0]; f32[9] = velocity[1]; f32[10] = velocity[2]; // f32[11] = 0 implicit
    this.device.queue.writeBuffer(this.sphereBuffer, 0, buf);

    // Dispatch
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.spherecast);
    pass.setBindGroup(0, this.bindGroups.spherecast);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();

    encoder.copyBufferToBuffer(this.hitBuffer, 0, this.readbackBuffer, 0, 64);
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuffer.getMappedRange().slice(0, 64));
    this.readbackBuffer.unmap();

    return {
      hit: data[0] !== 0,
      pushOut: [data[4], data[5], data[6]],
      normal: [data[8], data[9], data[10]],
      penetrationDepth: data[13],
    };
  }

  destroy() {
    const bufs = ['hitBuffer', 'rayBuffer', 'sphereBuffer', 'readbackBuffer'];
    for (const key of bufs) {
      if (this[key]) this[key].destroy();
    }
  }
}
