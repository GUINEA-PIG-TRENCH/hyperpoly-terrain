/**
 * TERRAIN RENDERER — WebGPU Render Pipeline for Dual-Contour Mesh
 * 
 * Consumes vertex/index buffers produced by Phase5Extractor + Phase5Draw
 * and renders them through a WGSL shader with material coloring.
 * 
 * Usage:
 *   const renderer = new TerrainRenderer(device, context);
 *   await renderer.init(renderWGSL);
 *   
 *   // Each frame:
 *   renderer.render(encoder, extractor.getVertexBuffer(), draw, viewProjMatrix);
 */

export class TerrainRenderer {
  constructor(device, gpuContext, depthTexture) {
    this.device = device;
    this.context = gpuContext;
    this.depthTexture = depthTexture;
    this.pipeline = null;
    this.uniformBuffer = null;
    this.bindGroup = null;
    this.cohesionBuffer = null;
  }

  /**
   * Initialize the render pipeline and uniform buffer.
   * @param {string} renderWGSL - Source code for render_terrain.wgsl
   * @param {GPUBuffer} [cohesionBuffer] - Optional storage buffer with per-vertex cohesion values
   */
  async init(renderWGSL, cohesionBuffer = null) {
    const device = this.device;
    const format = navigator.gpu.getPreferredCanvasFormat();

    // ── Uniform buffer: 
    //    view_proj (64B) + sun_direction (12B + 4B pad)
    //    + ambient_color (12B) + sun_color (12B) 
    //    + camera_pos (12B) + time (4B)
    //    Total: 128B
    this.uniformBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Initial uniforms
    const initialData = new Float32Array(32);
    initialData[16] = -0.5; initialData[17] = 0.7; initialData[18] = -0.5; // sun_direction
    initialData[20] = 0.15; initialData[21] = 0.18; initialData[22] = 0.25; // ambient
    initialData[24] = 1.0; initialData[25] = 0.95; initialData[26] = 0.85;  // sun color
    initialData[28] = 0.0; initialData[29] = 10.0; initialData[30] = 20.0;  // camera_pos
    initialData[31] = 0.0; // time
    device.queue.writeBuffer(this.uniformBuffer, 0, initialData);

    // ── Cohesion buffer — if not provided, create a stub
    this.cohesionBuffer = cohesionBuffer;
    if (!this.cohesionBuffer) {
      this.cohesionBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
      });
    }

    // ── Shader module ──
    const shaderModule = device.createShaderModule({ code: renderWGSL });

    // ── Render pipeline ──
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 12,  // vec3<f32> = 12 bytes
          attributes: [{
            format: 'float32x3',
            offset: 0,
            shaderLocation: 0,
          }],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs',
        targets: [{
          format: format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
        frontFace: 'ccw',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    // ── Bind group ──
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.cohesionBuffer } },
      ],
    });

    console.log('TerrainRenderer initialized');
  }

  /**
   * Update uniforms each frame using AI Lighting params.
   * @param {Float32Array} viewProjMatrix - 16 floats column-major
   * @param {object} aiLight - {sunDirection, ambientColor, sunColor, fogDensity, fogColor}
   */
  setViewProjAI(viewProjMatrix, aiLight) {
    const device = this.device;
    const data = new Float32Array(32);
    data.set(viewProjMatrix, 0);
    // sun_direction (from AI or procedural)
    data[16] = aiLight.sunDirection[0];
    data[17] = aiLight.sunDirection[1];
    data[18] = aiLight.sunDirection[2];
    // ambient_color
    data[20] = aiLight.ambientColor[0];
    data[21] = aiLight.ambientColor[1];
    data[22] = aiLight.ambientColor[2];
    // sun_color
    data[24] = aiLight.sunColor[0];
    data[25] = aiLight.sunColor[1];
    data[26] = aiLight.sunColor[2];
    // camera_pos (unused by shader currently, but preserved)
    data[28] = 0; data[29] = 0; data[30] = 0;
    // time
    data[31] = performance.now() * 0.001;
    device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  /**
   * Update uniforms each frame.
   * @param {Float32Array} viewProjMatrix - 16 floats column-major
   * @param {vec3} cameraPos - camera world position
   * @param {number} time - elapsed time in seconds
   */
  setViewProj(viewProjMatrix, cameraPos = [0, 10, 20], time = 0) {
    const device = this.device;
    const data = new Float32Array(32);
    data.set(viewProjMatrix, 0);
    data[16] = -0.5; data[17] = 0.7; data[18] = -0.5; // sun_direction
    data[20] = 0.15; data[21] = 0.18; data[22] = 0.25; // ambient
    data[24] = 1.0; data[25] = 0.95; data[26] = 0.85;  // sun color
    data[28] = cameraPos[0]; data[29] = cameraPos[1]; data[30] = cameraPos[2]; // camera_pos
    data[31] = time;
    device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  /**
   * Render the terrain mesh. Call within a command encoder.
   * @param {GPUCommandEncoder} encoder
   * @param {GPUBuffer} vertexBuffer - from Phase5Extractor.getVertexBuffer()
   * @param {Phase5Draw} draw - the draw object (provides index buffer + indirect args)
   */
  render(encoder, vertexBuffer, draw) {
    const textureView = this.context.getCurrentTexture().createView();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.1, g: 0.15, b: 0.3, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.bindGroup);
    renderPass.setIndexBuffer(draw.indexBuffer, 'uint32');
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.drawIndexedIndirect(draw.drawArgsBuffer, 0);

    renderPass.end();
  }

  destroy() {
    if (this.uniformBuffer) this.uniformBuffer.destroy();
  }
}
