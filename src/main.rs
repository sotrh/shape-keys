use std::{f32::consts::PI, num::NonZeroU64};

use anyhow::{bail, Context};
use bytemuck::{bytes_of, cast_slice};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: glam::Vec3,
    normal: glam::Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Morphs {
    d0_position: glam::Vec3,
    d0_normal: glam::Vec3,
    d1_position: glam::Vec3,
    d1_normal: glam::Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    weights: glam::Vec4,
    normal_mat: glam::Mat4,
    mvp: glam::Mat4,
}

struct UniformHandler {
    aspect_ratio: f32,
    fovy: f32,
    angle: f32,
    eye: glam::Vec3,
    look_at: glam::Vec3,
    rotation_speed: f32,
    data: Uniforms,
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    view: glam::Mat4,
    proj: glam::Mat4,
}

impl UniformHandler {
    pub fn new(
        device: &wgpu::Device,
        rotation_speed: f32,
        fovy: f32,
        aspect_ratio: f32,
        eye: glam::Vec3,
        look_at: glam::Vec3,
    ) -> Self {
        let proj = glam::Mat4::perspective_lh(fovy, aspect_ratio, 0.1, 100.0);
        let view = glam::Mat4::look_at_lh(eye, look_at, glam::Vec3::Y);
        let mvp = proj * view;

        let data = Uniforms {
            weights: glam::vec4(0.0, 0.0, 0.0, 0.0),
            normal_mat: view,
            mvp,
        };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytes_of(&data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as _),
                },
                visibility: wgpu::ShaderStages::VERTEX,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniforms"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            aspect_ratio,
            fovy,
            angle: 0.0,
            rotation_speed,
            eye,
            look_at,
            view,
            proj,
            data,
            buffer,
            layout,
            bind_group,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
        self.proj = glam::Mat4::perspective_lh(self.fovy, self.aspect_ratio, 0.1, 100.0);
    }

    fn set_weights(&mut self, w0: f32, w1: f32) {
        self.data.weights.x = w0;
        self.data.weights.y = w1;
    }

    fn update(&mut self, queue: &wgpu::Queue, dt: instant::Duration) {
        self.angle += self.rotation_speed * dt.as_secs_f32();
        let c = (self.angle * 2.0).cos() * 0.5 + 0.5;
        self.set_weights(c, 1.0 - c);
        self.data.normal_mat = self.view * glam::Mat4::from_rotation_y(self.angle);
        self.data.mvp = self.proj * self.data.normal_mat;
        queue.write_buffer(&self.buffer, 0, bytes_of(&self.data));
    }
}

struct Renderer {
    pipeline: wgpu::RenderPipeline,
    depth_format: wgpu::TextureFormat,
    depth_buffer: wgpu::TextureView,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        surf_config: &wgpu::SurfaceConfiguration,
        uniforms: &UniformHandler,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Buffer"),
            size: wgpu::Extent3d {
                width: surf_config.width,
                height: surf_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });
        let depth_buffer = depth_texture.create_view(&Default::default());

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Renderer"),
            bind_group_layouts: &[&uniforms.layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Renderer"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as _,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,
                            1 => Float32x3,
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Morphs>() as _,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            2 => Float32x3,
                            3 => Float32x3,
                            4 => Float32x3,
                            5 => Float32x3,
                        ],
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                entry_point: "fs_main",
                module: &shader,
                targets: &[Some(wgpu::ColorTargetState {
                    format: surf_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self {
            pipeline,
            depth_format,
            depth_buffer,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Buffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });
        self.depth_buffer = depth_texture.create_view(&Default::default());
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        uniforms: &UniformHandler,
        model: &Model,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_buffer,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &uniforms.bind_group, &[]);
        pass.set_index_buffer(model.index_buffer.slice(..), model.index_format);
        pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, model.morph_buffer.slice(..));
        pass.draw_indexed(0..model.num_indices, 0, 0..1);
    }
}

struct Model {
    vertex_buffer: wgpu::Buffer,
    morph_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    num_indices: u32,
}

impl Model {
    fn from_gltf(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        document: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
    ) -> anyhow::Result<Self> {
        // For this example we'll assume the file only has one mesh,
        // which has one primitive.
        let mesh = document
            .meshes()
            .next()
            .with_context(|| "Model should have 1 mesh")?;
        let prim = mesh
            .primitives()
            .next()
            .with_context(|| "Mesh should have 1 primitive")?;

        // We need to index format to render properly.
        let indices = prim.indices().unwrap();
        let index_format = match indices.data_type() {
            gltf::accessor::DataType::U16 => wgpu::IndexFormat::Uint16,
            gltf::accessor::DataType::U32 => wgpu::IndexFormat::Uint32,
            dt => bail!("Unsupported index type {:?}", dt),
        };

        // The index buffer usually doesn't have a stride,  so we can
        // upload the data to the gpu directly.
        let index_data = Self::get_data_for_accessor(&indices, buffers).unwrap();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: index_data,
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = indices.count() as u32;

        // Map each attribute to the ones we care about.
        let mut positions = None;
        let mut normals = None;
        prim.attributes().for_each(|(s, a)| match s {
            gltf::Semantic::Positions => positions = Some(a),
            gltf::Semantic::Normals => normals = Some(a),
            _ => (), // Ignore other attributes
        });

        let positions = positions.unwrap();
        let normals = normals.unwrap();

        // This shape-keys.glb model has vertex components separated
        // we'll combine them so the GPU doesn't have to jump around
        // when preparing for the vertex shader.
        let pos_data: &[glam::Vec3] =
            cast_slice(Self::get_data_for_accessor(&positions, buffers).unwrap());
        let norm_data: &[glam::Vec3] =
            cast_slice(Self::get_data_for_accessor(&normals, buffers).unwrap());
        let vertices = (0..pos_data.len().min(norm_data.len()))
            .map(|i| Vertex {
                position: pos_data[i],
                normal: norm_data[i],
            })
            .collect::<Vec<_>>();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // We need to do a similar thing to the morph data that we did
        // with the vertex data.
        let mut morphs = prim.morph_targets();
        let m0 = morphs.next().unwrap();
        let m1 = morphs.next().unwrap();

        let mp0 = m0.positions().unwrap();
        let mn0 = m0.normals().unwrap();

        let mp1 = m1.positions().unwrap();
        let mn1 = m1.normals().unwrap();

        let mp0_data: &[glam::Vec3] =
            cast_slice(Self::get_data_for_accessor(&mp0, buffers).unwrap());
        let mn0_data: &[glam::Vec3] =
            cast_slice(Self::get_data_for_accessor(&mn0, buffers).unwrap());
        let mp1_data: &[glam::Vec3] =
            cast_slice(Self::get_data_for_accessor(&mp1, buffers).unwrap());
        let mn1_data: &[glam::Vec3] =
            cast_slice(Self::get_data_for_accessor(&mn1, buffers).unwrap());
        let len = mp0_data
            .len()
            .min(mp1_data.len())
            .min(mn0_data.len())
            .min(mn1_data.len());
        let morphs = (0..len)
            .map(|i| Morphs {
                d0_position: mp0_data[i],
                d0_normal: mn0_data[i],
                d1_position: mp1_data[i],
                d1_normal: mn1_data[i],
            })
            .collect::<Vec<_>>();
        let morph_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Morphs"),
            contents: cast_slice(&morphs),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            vertex_buffer,
            morph_buffer,
            index_format,
            index_buffer,
            num_indices,
        })
    }

    /// Gets slice of the buffer for this accessor ignoring stride
    fn get_data_for_accessor<'a>(
        a: &gltf::Accessor<'a>,
        buffers: &'a [gltf::buffer::Data],
    ) -> Option<&'a [u8]> {
        let view = a.view()?;
        Some(&buffers[view.buffer().index()].0[view.offset()..view.offset() + view.length()])
    }
}

async fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Shape keys (morph targets)")
        .with_visible(false)
        .build(&event_loop)?;

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .filter(|a| a.is_surface_supported(&surface))
        .next()
        .with_context(|| "Unable to find valid WebGPU adapter")?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::default(),
                limits: wgpu::Limits::downlevel_webgl2_defaults(),
            },
            None,
        )
        .await
        .with_context(|| "Unable to find compatible WebGPU device")?;

    let mut surf_config = wgpu::SurfaceConfiguration {
        width: window.inner_size().width,
        height: window.inner_size().height,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_supported_formats(&adapter)[0],
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
    };
    surface.configure(&device, &surf_config);

    let mut uniforms = UniformHandler::new(
        &device,
        PI * 0.1,
        PI * 0.25,
        1.0,
        glam::vec3(0.0, 3.0, -5.0),
        glam::vec3(0.0, 0.0, 0.0),
    );
    let mut renderer = Renderer::new(&device, &surf_config, &uniforms);

    let (document, buffers, images) = gltf::import_slice(include_bytes!("shape-keys.glb"))?;
    let model = Model::from_gltf(&device, &queue, &document, &buffers, &images)?;

    window.set_visible(true);
    let mut last_time = instant::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state,
                            ..
                        },
                    ..
                } => match (key, state == ElementState::Pressed) {
                    (VirtualKeyCode::Escape, true) => control_flow.set_exit(),
                    _ => (),
                },
                WindowEvent::Resized(size) => {
                    renderer.resize(&device, size.width, size.height);
                    uniforms.resize(size.width, size.height);
                    surf_config.width = size.width;
                    surf_config.height = size.height;
                    surface.configure(&device, &surf_config);
                }
                _ => (),
            },
            Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                match surface.get_current_texture() {
                    Ok(tex) => {
                        let now = instant::Instant::now();
                        let dt = now - last_time;
                        last_time = now;

                        let view = tex.texture.create_view(&Default::default());
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });

                        uniforms.update(&queue, dt);
                        renderer.render(&mut encoder, &view, &uniforms, &model);

                        queue.submit([encoder.finish()]);
                        tex.present();
                    }
                    Err(wgpu::SurfaceError::Outdated) => {
                        // Called before resize occurs
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                        control_flow.set_exit_with_code(1);
                    }
                }
            }
            _ => (),
        }
    });
}

fn main() {
    pollster::block_on(run()).unwrap();
}
