/**
 * PLAYABLE WORLD — The glue that wires input, UI, audio to the engine
 * ~50 lines. A single class that makes the metabolism playable.
 */

import { HyperPolyGeology } from './hyperpoly_geology.js';
import { MinimalInput } from './minimal_input.js';
import { MinimalUI } from './minimal_ui.js';
import { MinimalAudio } from './minimal_audio.js';
import { EditManager } from './phase6_edit/phase6_host.js';
import { Phase5Extractor } from './phase5_extractor/phase5_host.js';
import { Phase5Draw } from './phase5_extractor/phase5_draw.js';
import { MaterialCalibration } from './material_calibration.js';

export class PlayableWorld {
  constructor(canvas, device) {
    this.canvas = canvas;
    this.device = device;
    this._wgpuContext = null;
    this._depthTexture = null;
    this._terrainRenderer = null;

    // Initialize layers
    this.engine = new HyperPolyGeology(device, 256);
    this.editManager = new EditManager(device);
    this.extractor = null;
    this.draw = null;
    this.input = new MinimalInput(canvas, this);
    this.ui = new MinimalUI(canvas);
    this.audio = new MinimalAudio();

    this.lastTime = 0;
    this.simAccumulator = 0;
    this.simHz = 20;
    this.simStep = 1000 / this.simHz;
    this.isDirty = false;
  }

  async init(shellModelGenerator, wgslSources) {
    // Load material calibration from PBR texture statistics
    try {
      const cal = await MaterialCalibration.load('calibration/calibration.json');
      cal.setProfile('gravel');
      shellModelGenerator = cal.wrapGenerator(shellModelGenerator, 'gravel');
      this._calibration = cal;
      console.log('Material calibration loaded: gravel profile');
    } catch (e) {
      console.warn('Material calibration not available, using procedural defaults:', e.message);
    }

    // Upload terrain
    await this.engine.uploadQuantizedTerrain(shellModelGenerator);

    // Wire edit manager
    this.engine.editManager = this.editManager;

    // Initialize extractor and draw pipeline
    this.extractor = new Phase5Extractor(this.device, 256);
    this.draw = new Phase5Draw(this.device);

    // Load WGSL sources and create pipelines
    await this.editManager.init(wgslSources.edit);
    await this.extractor.init(wgslSources);
    await this.draw.init(wgslSources);

    // Initial mesh extraction
    this.editManager.createBindGroup(
      this.engine.metaBuffer,
      this.engine.channelBuffers[0],  // density (channel 0)
      this.engine.channelBuffers[1],  // cohesion (channel 1)
      this.engine.channelBuffers[2],  // ice (channel 2)
      this.engine.channelBuffers[4],  // water (channel 4)
      this.engine.brickFlagsBuffer    // flags
    );
    await this.extractor.fullExtract(
      this.engine.metaBuffer,
      this.engine.channelBuffers,
      this.engine.brickFlagsBuffer
    );
    await this.draw.buildMesh(
      this.extractor.getVertexBuffer(),
      this.extractor.hermiteBuffer,
      this.extractor.getLODBuffer(),
      0.5
    );

    // Show ready
    this.ui.showMessage('HYPERPOLY v11 — Click to start', 4000);
    requestAnimationFrame((t) => this.frame(t));
  }

  frame(timestamp) {
    const dt = timestamp - this.lastTime;
    this.lastTime = timestamp;

    // 1. Handle input
    const inputState = this.input.update();

    // 2. Step simulation (fixed timestep)
    this.simAccumulator += dt;
    let stepped = false;
    while (this.simAccumulator >= this.simStep) {
      const encoder = this.device.createCommandEncoder();

      // Apply pending edits
      this.editManager.applyEdits(encoder);

      // Run simulation tick
      this.engine.stepSimulation(encoder);

      // Submit
      this.device.queue.submit([encoder.finish()]);

      this.simAccumulator -= this.simStep;
      stepped = true;
    }

    // 3. RENDER FRAME (always at display rate)
    if (stepped || this.isDirty) {
      // Extract mesh if dirty
      this.extractor.fullExtract(
        this.engine.metaBuffer,
        this.engine.channelBuffers,
        this.engine.brickFlagsBuffer
      );
      this.draw.buildMesh(
        this.extractor.getVertexBuffer(),
        this.extractor.hermiteBuffer,
        this.extractor.getLODBuffer(),
        0.5
      );
      this.isDirty = false;
    }

    // 4. Render
    this._render();

    // 5. Update UI overlay
    this.ui.render(inputState, {
      frameTime: dt,
      simHz: stepped ? this.simHz : 0
    });

    requestAnimationFrame((t) => this.frame(t));
  }

  _render() {
    // WebGPU render via TerrainRenderer (if wired from bridge.html)
    if (this._terrainRenderer && this.extractor && this.draw) {
      return;
    }
    // Fallback: keep original empty encoder for Three.js mode
    const encoder = this.device.createCommandEncoder();
    this.device.queue.submit([encoder.finish()]);
  }
}
