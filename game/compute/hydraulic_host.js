// =============================================================================
// Phase 1: Two-Pass Dispatch Host — HYPERPOLY Hydraulic Solver
// =============================================================================
// Pass 1: Culling compute — reads brick_metadata, writes indirect dispatch buffer
// Pass 2: Solver compute — indirect dispatch from Pass 1 output
// =============================================================================

const WORLD_DIM   = 256;
const BRICK_DIM   = 16;
const BRICKS_X    = WORLD_DIM / BRICK_DIM;  // 16
const BRICKS_Y    = WORLD_DIM / BRICK_DIM;  // 16
const BRICKS_Z    = WORLD_DIM / BRICK_DIM;  // 16

const TOTAL_VOXELS       = WORLD_DIM * WORLD_DIM * WORLD_DIM;           // 16,777,216
const TOTAL_BRICK_SLICES = BRICKS_X * BRICKS_Y * BRICKS_Z;             // 4,096
const MAX_ACTIVE_BRICKS  = TOTAL_BRICK_SLICES;

const BUDGET_MAX_DEFAULT = 3000;

// =============================================================================
// GPU Resource Manager
// =============================================================================

export class HydraulicPipeline {
    constructor(device, budgetMax = BUDGET_MAX_DEFAULT) {
        this.device = device;
        this.budgetMax = budgetMax;

        // --- SoA Voxel Buffers (u16 per voxel = 2 bytes) ---
        this.voxelWater    = this._createStorage(TOTAL_VOXELS * 2);
        this.voxelWaterDst = this._createStorage(TOTAL_VOXELS * 2);
        this.voxelSediment = this._createStorage(TOTAL_VOXELS * 2);
        this.voxelPermX    = this._createStorage(TOTAL_VOXELS * 2);
        this.voxelPermY    = this._createStorage(TOTAL_VOXELS * 2);
        this.voxelPermZ    = this._createStorage(TOTAL_VOXELS * 2);
        this.voxelCohesion = this._createStorage(TOTAL_VOXELS * 2);

        // --- Brick Metadata (6 channels × 16 bytes per brick) ---
        this.brickMetadata = this._createStorage(TOTAL_BRICK_SLICES * 6 * 16);

        // --- Brick State for culling tracking (u32 per brick) ---
        this.brickState = this._createStorage(TOTAL_BRICK_SLICES * 4);

        // --- Dispatch Chain Buffers ---
        // Indirect dispatch: 3 × u32 = 12 bytes
        this.indirectBuffer = device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });

        // Active list: ActiveBrick (budgeted queue index list)
        this.activeList = this._createStorage(MAX_ACTIVE_BRICKS * 4);

        // Budget counter — reset per frame by GPU-side reset kernel
        this.budgetCounter = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Uniform buffer: total_bricks + dimensions + budget_max
        this._uniformData = new Uint32Array([
            TOTAL_BRICK_SLICES,  // total_bricks
            BRICKS_X,
            BRICKS_Y,
            WORLD_DIM,
            budgetMax,
        ]);
        this.uniformBuffer = device.createBuffer({
            size: 5 * 4,  // 5 × u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.uniformBuffer, 0, this._uniformData);

        // Pre-allocated zero buffer for resetting indirect dispatch state
        this._dispatchInit = new Uint32Array([0, 1, 1]);  // x=0, y=1, z=1

        // Meta-dispatcher budget
        this._budgetMax = budgetMax;
        this._metaParams = new Float32Array([budgetMax, 0.85, 0.3, 0.0]);
        this._budgetedQueue = this._createStorage(MAX_ACTIVE_BRICKS * 4);  // u32
        this._brickPriority = this._createStorage(TOTAL_BRICK_SLICES * 4); // f32

        // Culling schedule parameters (moistureThreshold, stabilityThreshold, emaAlpha, deadband)
        this.schedParamsBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.schedParamsBuffer, 0, new Float32Array([0.1, 0.5, 0.3, 0.05]));

        // Camera position uniform for LOD culling (vec3<f32> = 12 bytes)
        this._cameraPos = new Float32Array([128.0, 64.0, 128.0]);
        this.cameraBuffer = device.createBuffer({
            size: 16,  // vec3 padded to 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.cameraBuffer, 0, this._cameraPos);

        // Reset queue pipeline (GPU-side, eliminates host writeBuffer)
        this._resetPipeline = null;
        this._resetBindGroup = null;
    }

    // ==========================================================================
    // Initialization — must be called after WGSL sources are fetched
    // ==========================================================================

    async init(pass1WGSL, pass2WGSL, metaDispatchWGSL, editMinBuffer, editMaxBuffer) {
        const device = this.device;

        const pass1Module = device.createShaderModule({ code: pass1WGSL });
        const pass2Module = device.createShaderModule({ code: pass2WGSL });

        this.pass1Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: pass1Module, entryPoint: 'culling_pass' },
        });

        // --- Reset Queue Pipeline ---
        const resetModule = device.createShaderModule({
            code: await fetch('compute/reset_queue.wgsl').then(r => r.text()),
        });
        this._resetPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: resetModule, entryPoint: 'reset_pass' },
        });
        this._resetBindGroup = device.createBindGroup({
            layout: this._resetPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.budgetCounter }},
            ],
        });

        // --- Meta-Dispatcher Pipeline ---
        if (metaDispatchWGSL) {
            const metaModule = device.createShaderModule({ code: metaDispatchWGSL });
            this._metaPipeline = device.createComputePipeline({
                layout: 'auto',
                compute: { module: metaModule, entryPoint: 'meta_dispatcher' },
            });
            this._metaBindGroup = device.createBindGroup({
                layout: this._metaPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.activeList     }},      // compacted_queue
                    { binding: 1, resource: { buffer: this.budgetCounter  }},       // queue_count
                    { binding: 2, resource: { buffer: this._brickPriority }},       // brick_priority
                    { binding: 3, resource: { buffer: this._budgetedQueue }},       // budgeted_queue
                    { binding: 4, resource: { buffer: this.indirectBuffer }},       // dispatch_args (reuse mem)
                    { binding: 5, resource: { buffer: this._metaUniformBuffer }},   // meta_params
                ],
            });
            // Meta params uniform
            this._metaUniformBuffer = device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(this._metaUniformBuffer, 0, this._metaParams);
        }

        this.pass2Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: pass2Module, entryPoint: 'advection_pass' },
        });

        // --- Pass 1 Bind Group ---
        this.pass1BindGroup = device.createBindGroup({
            layout: this.pass1Pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.brickMetadata }},
                { binding: 1, resource: { buffer: this.brickState    }},
                { binding: 2, resource: { buffer: this.activeList     }},
                { binding: 3, resource: { buffer: this.budgetCounter  }},
                { binding: 4, resource: { buffer: this.schedParamsBuffer }},
                { binding: 5, resource: { buffer: this.cameraBuffer   }},
            ],
        });

        // --- Pass 2 Bind Groups ---
        this.pass2BindGroup0 = device.createBindGroup({
            layout: this.pass2Pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.brickMetadata }},
                { binding: 1, resource: { buffer: this.voxelWater    }},
                { binding: 2, resource: { buffer: this.voxelWaterDst }},
                { binding: 3, resource: { buffer: this.voxelPermX    }},
                { binding: 4, resource: { buffer: this.voxelPermY    }},
                { binding: 5, resource: { buffer: this.voxelPermZ    }},
                { binding: 10, resource: { buffer: editMinBuffer     }},
                { binding: 11, resource: { buffer: editMaxBuffer     }},
            ],
        });

        this.pass2BindGroup1 = device.createBindGroup({
            layout: this.pass2Pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: this._budgetedQueue }}, // compacted_queue
            ],
        });
    }

    // ==========================================================================
    // Upload initial world state from material tensor
    // ==========================================================================

    uploadInitialState({ water, sediment, permX, permY, permZ, cohesion, metadata }) {
        const q = this.device.queue;
        q.writeBuffer(this.voxelWater,    0, water);
        q.writeBuffer(this.voxelWaterDst, 0, water);
        q.writeBuffer(this.voxelSediment, 0, sediment);
        q.writeBuffer(this.voxelPermX,    0, permX);
        q.writeBuffer(this.voxelPermY,    0, permY);
        q.writeBuffer(this.voxelPermZ,    0, permZ);
        q.writeBuffer(this.voxelCohesion, 0, cohesion);
        q.writeBuffer(this.brickMetadata, 0, metadata);
    }

    // ==========================================================================
    // One simulation step (call per frame)
    // ==========================================================================

    step(cameraX, cameraY, cameraZ) {
        const device = this.device;
        const queue  = device.queue;

        const cmd = device.createCommandEncoder();

        // Update camera position for LOD culling
        this._cameraPos[0] = cameraX ?? this._cameraPos[0];
        this._cameraPos[1] = cameraY ?? this._cameraPos[1];
        this._cameraPos[2] = cameraZ ?? this._cameraPos[2];
        queue.writeBuffer(this.cameraBuffer, 0, this._cameraPos);

        // ========================================================================
        // Pass 0: GPU-side queue reset
        // ========================================================================
        {
            const pass = cmd.beginComputePass();
            pass.setPipeline(this._resetPipeline);
            pass.setBindGroup(0, this._resetBindGroup);
            pass.dispatchWorkgroups(1, 1, 1);
            pass.end();
        }

        // ========================================================================
        // Pass 1: Culling — determine active bricks
        // ========================================================================
        {
            const pass = cmd.beginComputePass();
            pass.setPipeline(this.pass1Pipeline);
            pass.setBindGroup(0, this.pass1BindGroup);

            const workgroups = Math.ceil(TOTAL_BRICK_SLICES / 64);
            pass.dispatchWorkgroups(workgroups, 1, 1);
            pass.end();
        }

        // ========================================================================
        // Pass 1b: Meta-dispatcher — budget-constrain the active list
        // ========================================================================
        if (this._metaPipeline) {
            // Update meta params uniform
            queue.writeBuffer(this._metaUniformBuffer, 0, this._metaParams);

            const pass = cmd.beginComputePass();
            pass.setPipeline(this._metaPipeline);
            pass.setBindGroup(0, this._metaBindGroup);
            pass.dispatchWorkgroups(16, 1, 1); // 16 workgroups * 256 threads covers 4096 bricks
            pass.end();
        }

        // ========================================================================
        // Pass 2: Hydraulic solver — only active bricks run
        // ========================================================================
        {
            const pass = cmd.beginComputePass();
            pass.setPipeline(this.pass2Pipeline);
            pass.setBindGroup(0, this.pass2BindGroup0);
            pass.setBindGroup(1, this.pass2BindGroup1);

            pass.dispatchWorkgroupsIndirect(this.indirectBuffer, 0);
            pass.end();
        }

        // ========================================================================
        // Pass 3: Water Ping-Pong Buffer Swap
        // ========================================================================
        cmd.copyBufferToBuffer(this.voxelWaterDst, 0, this.voxelWater, 0, TOTAL_VOXELS * 2);

        queue.submit([cmd.finish()]);
    }

    // ==========================================================================
    // Readback dispatch count for Phase 1 validation
    // ==========================================================================

    async readDispatchCount() {
        const staging = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const cmd = this.device.createCommandEncoder();
        cmd.copyBufferToBuffer(this.indirectBuffer, 0, staging, 0, 4);
        this.device.queue.submit([cmd.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const count = new Uint32Array(staging.getMappedRegion())[0];
        staging.destroy();
        return count;
    }

    // ==========================================================================
    // Update budget uniform
    // ==========================================================================

    setBudgetMax(newBudget) {
        this.budgetMax = newBudget;
        this._uniformData[4] = newBudget;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this._uniformData);
    }

    // ==========================================================================
    // Internal
    // ==========================================================================

    _createStorage(size) {
        return this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }
}

// =============================================================================
// Bootstrap helper
// =============================================================================

export async function createHydraulicPipeline(device, budgetMax, editMinBuffer, editMaxBuffer) {
    const [pass1WGSL, pass2WGSL, metaDispatchWGSL] = await Promise.all([
        fetch('compute/pass1_culling.wgsl').then(r => r.text()),
        fetch('compute/pass2_solver.wgsl').then(r => r.text()),
        fetch('compute/meta_dispatch.wgsl').then(r => r.text()),
    ]);

    const pipeline = new HydraulicPipeline(device, budgetMax);
    await pipeline.init(pass1WGSL, pass2WGSL, metaDispatchWGSL, editMinBuffer, editMaxBuffer);
    return pipeline;
}
