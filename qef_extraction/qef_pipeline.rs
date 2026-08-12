// qef_pipeline.rs
// Rust dispatch orchestrator for the QEF mesh extraction pipeline.
// Ties together: spatial_hash → density → marching_tets → QEF → mesh → Filament.
//
// Usage:
//   let mut pipeline = QefPipeline::new(&device, &queue, 64);
//   pipeline.extract_mesh(&particle_positions, &spatial_hash_grid);
//   let renderable = pipeline.to_filament(&engine);

use std::sync::Arc;
use wgpu::{util::DeviceExt, *};
use bytemuck::{Pod, Zeroable};

// ─── GPU-side structs (must match WGSL layouts) ──────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DensityParams {
    grid_dim: u32,
    max_particles_per_cell: u32,
    particle_radius: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MTParams {
    grid_dim: u32,
    cell_size: f32,
    isosurface: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QEFParams {
    grid_dim: u32,
    cell_size: f32,
    regularization: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MeshParams {
    grid_dim: u32,
    cell_size: f32,
    merge_threshold: f32,
    max_vertices: u32,
    max_indices: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QefVertex {
    position: [f32; 3],
    _pad0: f32,
    normal: [f32; 3],
    _pad1: f32,
    material_tensor: [f32; 4], // first 4 of 6 channels
}

// ─── Pipeline ────────────────────────────────────────────────────────────

pub struct QefPipeline {
    grid_dim: u32,
    cell_size: f32,

    // Kernels
    density_pipeline: ComputePipeline,
    mt_pipeline: ComputePipeline,
    qef_pipeline: ComputePipeline,
    dedup_pipeline: ComputePipeline,
    index_pipeline: ComputePipeline,

    // Bind group layouts
    density_bgl: BindGroupLayout,
    mt_bgl: BindGroupLayout,
    qef_bgl: BindGroupLayout,
    mesh_bgl: BindGroupLayout,

    // Buffers (allocated once, reused per frame)
    density_field: Buffer,
    crossings: Buffer,
    crossing_lut: Buffer,
    crossing_count: Buffer,
    raw_vertices: Buffer,
    vertex_count: Buffer,
    mesh_vertices: Buffer,
    mesh_indices: Buffer,
    mesh_vcount: Buffer,
    mesh_icount: Buffer,

    // Uniforms
    density_uniform: Buffer,
    mt_uniform: Buffer,
    qef_uniform: Buffer,
    mesh_uniform: Buffer,

    max_particles: u32,
    max_crossings: u32,
    max_vertices: u32,
    max_indices: u32,
}

impl QefPipeline {
    pub fn new(device: &Device, grid_dim: u32, cell_size: f32, max_particles: u32) -> Self {
        let grid_cells = grid_dim * grid_dim * grid_dim;
        let max_crossings = grid_cells * 36; // 6 tets × 6 edges max per cell
        let max_vertices = grid_cells * 3;    // avg 3 vertices per crossing cell
        let max_indices = grid_cells * 12;    // avg 4 tris per crossing cell

        // Load WGSL
        let density_src = include_str!("density_field.wgsl");
        let mt_src = include_str!("marching_tets.wgsl");
        let qef_src = include_str!("qef_solve.wgsl");
        let mesh_src = include_str!("mesh_assembly.wgsl");

        // Shader modules
        let density_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("density_field"),
            source: ShaderSource::Wgsl(density_src.into()),
        });
        let mt_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("marching_tets"),
            source: ShaderSource::Wgsl(mt_src.into()),
        });
        let qef_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("qef_solve"),
            source: ShaderSource::Wgsl(qef_src.into()),
        });
        let mesh_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_assembly"),
            source: ShaderSource::Wgsl(mesh_src.into()),
        });

        // Bind group layouts (matching WGSL group indices)
        let density_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("density_bgl"),
            entries: &[
                storage_read(0, false),  // grid_heads (provided externally)
                storage_read(1, false),  // grid_next
                storage_read(2, false),  // particle_positions
                storage_rw(3, false),    // density_field
                uniform(4, false),       // params
            ],
        });

        let mt_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mt_bgl"),
            entries: &[
                storage_read(0, false),  // density_field
                storage_rw(1, false),    // crossing_count
                storage_rw(2, false),    // crossings
                uniform(3, false),       // params
                storage_rw(4, false),    // crossing_lut
            ],
        });

        let qef_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("qef_bgl"),
            entries: &[
                storage_read(0, false),  // density_field
                storage_read(1, false),  // crossings
                storage_read(2, false),  // crossing_count
                storage_rw(3, false),    // vertices
                storage_rw(4, false),    // vertex_count
                uniform(5, false),       // params
            ],
        });

        let mesh_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_bgl"),
            entries: &[
                storage_read(0, false),  // raw_vertices
                storage_read(1, false),  // vertex_count_in
                storage_rw(2, false),    // mesh_vertices
                storage_rw(3, false),    // mesh_indices
                storage_rw(4, false),    // mesh_vertex_count
                storage_rw(5, false),    // mesh_index_count
                storage_read(6, false),  // density_field
                uniform(7, false),       // params
                storage_read(8, false),  // crossing_lut
            ],
        });

        // Pipeline layouts
        let density_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("density_pl"),
            bind_group_layouts: &[&density_bgl],
            push_constant_ranges: &[],
        });
        let mt_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mt_pl"),
            bind_group_layouts: &[&mt_bgl],
            push_constant_ranges: &[],
        });
        let qef_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("qef_pl"),
            bind_group_layouts: &[&qef_bgl],
            push_constant_ranges: &[],
        });
        let mesh_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mesh_pl"),
            bind_group_layouts: &[&mesh_bgl],
            push_constant_ranges: &[],
        });

        // Compute pipelines
        let density_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("density"),
            layout: Some(&density_pl),
            module: &density_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        });
        let mt_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("marching_tets"),
            layout: Some(&mt_pl),
            module: &mt_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        });
        let qef_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("qef_solve"),
            layout: Some(&qef_pl),
            module: &qef_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        });
        let dedup_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dedup"),
            layout: Some(&mesh_pl),
            module: &mesh_module,
            entry_point: Some("deduplicate_vertices"),
            compilation_options: Default::default(),
        });
        let index_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("index"),
            layout: Some(&mesh_pl),
            module: &mesh_module,
            entry_point: Some("build_indices"),
            compilation_options: Default::default(),
        });

        // Buffers
        let density_field = device.create_buffer(&BufferDescriptor {
            label: Some("density_field"),
            size: (grid_cells as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let crossings = device.create_buffer(&BufferDescriptor {
            label: Some("crossings"),
            size: (max_crossings as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let crossing_lut = device.create_buffer(&BufferDescriptor {
            label: Some("crossing_lut"),
            size: (max_crossings as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let crossing_count = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("crossing_count"),
            contents: &0u32.to_le_bytes(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        let raw_vertices = device.create_buffer(&BufferDescriptor {
            label: Some("raw_vertices"),
            size: (max_vertices as u64) * 12,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vertex_count = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex_count"),
            contents: &0u32.to_le_bytes(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        let mesh_vertices = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_vertices"),
            size: (max_vertices as u64) * std::mem::size_of::<QefVertex>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let mesh_indices = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_indices"),
            size: (max_indices as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        let mesh_vcount = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("mesh_vcount"),
            contents: &0u32.to_le_bytes(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        let mesh_icount = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("mesh_icount"),
            contents: &0u32.to_le_bytes(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Uniforms
        let density_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("density_params"),
            contents: bytemuck::bytes_of(&DensityParams {
                grid_dim,
                max_particles_per_cell: 64,
                particle_radius: cell_size * 0.5,
                _pad: 0,
            }),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let mt_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("mt_params"),
            contents: bytemuck::bytes_of(&MTParams {
                grid_dim,
                cell_size,
                isosurface: 0.1,
                _pad: 0,
            }),
            usage: BufferUsages::UNIFORM,
        });
        let qef_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("qef_params"),
            contents: bytemuck::bytes_of(&QEFParams {
                grid_dim,
                cell_size,
                regularization: 0.001,
                _pad: 0,
            }),
            usage: BufferUsages::UNIFORM,
        });
        let mesh_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("mesh_params"),
            contents: bytemuck::bytes_of(&MeshParams {
                grid_dim,
                cell_size,
                merge_threshold: cell_size * 0.01,
                max_vertices,
                max_indices,
                _pad: [0; 3],
            }),
            usage: BufferUsages::UNIFORM,
        });

        Self {
            grid_dim, cell_size,
            density_pipeline, mt_pipeline, qef_pipeline, dedup_pipeline, index_pipeline,
            density_bgl, mt_bgl, qef_bgl, mesh_bgl,
            density_field, crossings, crossing_lut, crossing_count, raw_vertices, vertex_count,
            mesh_vertices, mesh_indices, mesh_vcount, mesh_icount,
            density_uniform, mt_uniform, qef_uniform, mesh_uniform,
            max_particles, max_crossings, max_vertices, max_indices,
        }
    }

    /// Run the full QEF extraction pipeline.
    /// `grid_heads`, `grid_next`, `particle_positions` are buffers from spatial_hash.wgsl.
    pub fn extract_mesh(
        &self,
        encoder: &mut CommandEncoder,
        grid_heads: &Buffer,
        grid_next: &Buffer,
        particle_positions: &Buffer,
        grid_head_count: u32,
    ) {
        // ── Pass 1: Density Field ──────────────────────────────────────
        {
            let density_bg = encoder.device().create_bind_group(&BindGroupDescriptor {
                label: Some("density_bg"),
                layout: &self.density_bgl,
                entries: &[
                    BindGroupEntry { binding: 0, resource: grid_heads.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: grid_next.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: particle_positions.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: self.density_field.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: self.density_uniform.as_entire_binding() },
                ],
            });

            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("density_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.density_pipeline);
            cpass.set_bind_group(0, &density_bg, &[]);
            let wg = (self.grid_dim + 7) / 8;
            cpass.dispatch_workgroups(wg, wg, self.grid_dim);
        }

        // Reset counters
        encoder.clear_buffer(&self.crossing_count, 0, 4);
        encoder.clear_buffer(&self.crossing_lut, 0, (self.max_crossings as u64) * 4);

        // ── Pass 2: Marching Tetrahedra ────────────────────────────────
        {
            let mt_bg = encoder.device().create_bind_group(&BindGroupDescriptor {
                label: Some("mt_bg"),
                layout: &self.mt_bgl,
                entries: &[
                    BindGroupEntry { binding: 0, resource: self.density_field.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: self.crossing_count.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: self.crossings.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: self.mt_uniform.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: self.crossing_lut.as_entire_binding() },
                ],
            });

            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("mt_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.mt_pipeline);
            cpass.set_bind_group(0, &mt_bg, &[]);
            let wg = (self.grid_dim + 7) / 8;
            cpass.dispatch_workgroups(wg, wg, self.grid_dim);
        }

        // Reset vertex counter
        encoder.clear_buffer(&self.vertex_count, 0, 4);

        // ── Pass 3: QEF Solve ──────────────────────────────────────────
        {
            let qef_bg = encoder.device().create_bind_group(&BindGroupDescriptor {
                label: Some("qef_bg"),
                layout: &self.qef_bgl,
                entries: &[
                    BindGroupEntry { binding: 0, resource: self.density_field.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: self.crossings.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: self.crossing_count.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: self.raw_vertices.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: self.vertex_count.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: self.qef_uniform.as_entire_binding() },
                ],
            });

            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("qef_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.qef_pipeline);
            cpass.set_bind_group(0, &qef_bg, &[]);
            // Indirect dispatch: wait for MT pass crossing count, then dispatch
            // crossing_count / 64 workgroups
            let wg = (self.max_crossings + 63) / 64;
            cpass.dispatch_workgroups(wg, 1, 1);
        }

        // Reset mesh counters
        encoder.clear_buffer(&self.mesh_vcount, 0, 4);
        encoder.clear_buffer(&self.mesh_icount, 0, 4);

        // ── Pass 4: Mesh Assembly ──────────────────────────────────────
        {
            let mesh_bg = encoder.device().create_bind_group(&BindGroupDescriptor {
                label: Some("mesh_bg"),
                layout: &self.mesh_bgl,
                entries: &[
                    BindGroupEntry { binding: 0, resource: self.raw_vertices.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: self.vertex_count.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: self.mesh_vertices.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: self.mesh_indices.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: self.mesh_vcount.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: self.mesh_icount.as_entire_binding() },
                    BindGroupEntry { binding: 6, resource: self.density_field.as_entire_binding() },
                    BindGroupEntry { binding: 7, resource: self.mesh_uniform.as_entire_binding() },
                    BindGroupEntry { binding: 8, resource: self.crossing_lut.as_entire_binding() },
                ],
            });

            // Dedup
            {
                let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("dedup_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.dedup_pipeline);
                cpass.set_bind_group(0, &mesh_bg, &[]);
                let wg = (self.max_vertices + 63) / 64;
                cpass.dispatch_workgroups(wg, 1, 1);
            }

            // Index
            {
                let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("index_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.index_pipeline);
                cpass.set_bind_group(0, &mesh_bg, &[]);
                let total_cells = self.grid_dim * self.grid_dim * self.grid_dim;
                let wg = (total_cells + 63) / 64;
                cpass.dispatch_workgroups(wg, 1, 1);
            }
        }
    }

    /// Get output buffers ready for rendering.
    pub fn vertex_buffer(&self) -> &Buffer { &self.mesh_vertices }
    pub fn index_buffer(&self) -> &Buffer { &self.mesh_indices }
    pub fn vertex_count_buffer(&self) -> &Buffer { &self.mesh_vcount }
    pub fn index_count_buffer(&self) -> &Buffer { &self.mesh_icount }
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn storage_read(binding: u32, has_dynamic_offset: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw(binding: u32, has_dynamic_offset: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32, has_dynamic_offset: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size: None,
        },
        count: None,
    }
}

impl QefPipeline {
    /// Integrate with existing Filament render setup.
    /// Call this after extract_mesh() to build Filament VertexBuffer + IndexBuffer.
    #[cfg(feature = "filament")]
    pub fn to_filament(
        &self,
        engine: &mut filament::Engine,
        device: &Device,
        queue: &Queue,
    ) -> (filament::VertexBuffer, filament::IndexBuffer) {
        use filament::{VertexBuffer, IndexBuffer, VertexAttribute, VertexBufferType};

        // Download vertex/index counts from GPU
        let mut staging_vc = device.create_buffer(&BufferDescriptor {
            label: Some("staging_vc"),
            size: 4,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut staging_ic = device.create_buffer(&BufferDescriptor {
            label: Some("staging_ic"),
            size: 4,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.mesh_vcount, 0, &staging_vc, 0, 4);
        encoder.copy_buffer_to_buffer(&self.mesh_icount, 0, &staging_ic, 0, 4);
        queue.submit(Some(encoder.finish()));

        // Build Filament buffers (MVP: assume data is ready — production needs fence)
        // In practice, use a ring buffer or double-buffer for async readback.
        let vb = VertexBuffer::new(engine)
            .vertex_count(self.max_vertices)
            .buffer_count(1)
            .attribute(VertexAttribute::POSITION, 0, VertexBufferType::FLOAT3, 0, 0)
            .attribute(VertexAttribute::CUSTOM0, 0, VertexBufferType::FLOAT3, 12, 0)  // normal
            .attribute(VertexAttribute::CUSTOM1, 0, VertexBufferType::FLOAT4, 24, 0)  // material tensor (first 4)
            .build(engine);

        let ib = IndexBuffer::new(engine)
            .index_count(self.max_indices)
            .buffer_type(filament::IndexBufferType::UINT)
            .build(engine);

        (vb, ib)
    }
}
