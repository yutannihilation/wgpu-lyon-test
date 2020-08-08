use std::io::{self, Read, Write};

use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation;
use lyon::tessellation::geometry_builder::*;
use lyon::tessellation::{StrokeOptions, StrokeTessellator};

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use futures::executor::block_on;

use std::borrow::Cow::Borrowed;
use wgpu::util::DeviceExt;

const SAMPLE_COUNT: u32 = 4;

const IMAGE_DIR: &str = "img";

const BLUR_COUNT: usize = 30;

const EXPOSURE: f32 = 2.0;

// The vertex type that we will use to represent a point on our triangle.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlurVertex {
    position: [f32; 2],
    // tex_coords: [f32; 2],
}

unsafe impl bytemuck::Pod for BlurVertex {}
unsafe impl bytemuck::Zeroable for BlurVertex {}

#[rustfmt::skip]
const VERTICES: &[BlurVertex] = &[
    BlurVertex { position: [ 1.0,  1.0], },
    BlurVertex { position: [-1.0, -1.0], },
    BlurVertex { position: [ 1.0, -1.0], },

    BlurVertex { position: [ 1.0,  1.0], },
    BlurVertex { position: [-1.0, -1.0], },
    BlurVertex { position: [-1.0,  1.0], },
];

#[repr(C)] // We need this for Rust to store our data correctly for the shaders
#[derive(Debug, Copy, Clone)] // This is so we can store this in a buffer
struct BlurUniforms {
    horizontal: bool,
}

impl BlurUniforms {
    fn new() -> Self {
        Self { horizontal: true }
    }

    fn flip(&mut self) {
        self.horizontal = !self.horizontal;
    }
}

unsafe impl bytemuck::Pod for BlurUniforms {}
unsafe impl bytemuck::Zeroable for BlurUniforms {}

#[repr(C)] // We need this for Rust to store our data correctly for the shaders
#[derive(Debug, Copy, Clone)] // This is so we can store this in a buffer
struct BlendUniforms {
    exposure: f32,
}

impl BlendUniforms {
    fn new(exposure: f32) -> Self {
        Self { exposure }
    }
}

unsafe impl bytemuck::Pod for BlendUniforms {}
unsafe impl bytemuck::Zeroable for BlendUniforms {}

// State is derived from sotrh/learn-wgpu
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,

    // A render pipeline to draw both the usual texture and bright-part-only texture.
    extract_render_pipeline: wgpu::RenderPipeline,
    staging_texture: wgpu::Texture,

    // A render pipeline to apply gaussian blur. To use gaussian blur, we need two
    // textures so that one can be rendered to another and vice versa.
    blur_bind_group_layout: wgpu::BindGroupLayout,
    blur_uniform_bind_group_layout: wgpu::BindGroupLayout,
    blur_uniform: BlurUniforms,
    blur_render_pipeline: wgpu::RenderPipeline,
    blur_textures: [wgpu::Texture; 2],

    // A render pipeline for blending the results
    blend_bind_group_layout: wgpu::BindGroupLayout,
    blend_uniform_bind_group_layout: wgpu::BindGroupLayout,
    blend_render_pipeline: wgpu::RenderPipeline,

    multisample_texture: wgpu::Texture,
    geometry: VertexBuffers<Vertex, u16>,

    size: winit::dpi::PhysicalSize<u32>,

    output_dir: std::path::PathBuf,

    frame: u32,
}

impl State {
    async fn new(window: &Window) -> Self {
        // create an instance
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        // create an surface
        let (size, surface) = unsafe {
            let size = window.inner_size();
            let surface = instance.create_surface(window);
            (size, surface)
        };

        // create an adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        // create a device and a queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::default(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                // trace_path can be used for API call tracing
                None,
            )
            .await
            .unwrap();

        // create a swap chain
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        // Build a Path.
        let mut builder = Path::builder();
        builder.begin(point(-0.8, -0.3));
        builder.quadratic_bezier_to(point(1.5, 2.3), point(0.2, -0.9));
        builder.end(false);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex, u16> = VertexBuffers::new();

        let tolerance = 0.0001;

        let mut stroke_tess = StrokeTessellator::new();
        stroke_tess
            .tessellate_path(
                &path,
                &StrokeOptions::tolerance(tolerance).with_line_width(0.13),
                &mut BuffersBuilder::new(&mut geometry, |vertex: tessellation::StrokeVertex| {
                    Vertex {
                        position: vertex.position().to_array(),
                    }
                }),
            )
            .unwrap();

        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&[
                    wgpu::BindGroupLayoutEntry::new(
                        0,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                        },
                    ),
                    wgpu::BindGroupLayoutEntry::new(
                        1,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::Sampler { comparison: false },
                    ),
                ]),
                label: None,
            });

        let blur_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&[wgpu::BindGroupLayoutEntry::new(
                    0,
                    wgpu::ShaderStage::FRAGMENT,
                    wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<bool>() as _),
                    },
                )]),
                label: None,
            });

        // For staging buffer, we don't use bindgroups
        let extract_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: Borrowed(&[]),
                push_constant_ranges: Borrowed(&[]),
            });

        // For blur, we do use bind_group_layout
        let blur_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: Borrowed(&[
                    &blur_bind_group_layout,
                    &blur_uniform_bind_group_layout,
                ]),
                push_constant_ranges: Borrowed(&[]),
            });

        let blur_uniform = BlurUniforms::new();

        let blur_textures = [
            create_framebuffer(&device, &sc_desc, 1),
            create_framebuffer(&device, &sc_desc, 1),
        ];

        let blend_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&[
                    wgpu::BindGroupLayoutEntry::new(
                        0,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                        },
                    ),
                    wgpu::BindGroupLayoutEntry::new(
                        1,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                        },
                    ),
                    wgpu::BindGroupLayoutEntry::new(
                        2,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::Sampler { comparison: false },
                    ),
                ]),
                label: None,
            });

        let blend_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&[wgpu::BindGroupLayoutEntry::new(
                    0,
                    wgpu::ShaderStage::FRAGMENT,
                    wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<f32>() as _),
                    },
                )]),
                label: None,
            });

        // For blur, we do use bind_group_layout
        let blend_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: Borrowed(&[
                    &blend_bind_group_layout,
                    &blend_uniform_bind_group_layout,
                ]),
                push_constant_ranges: Borrowed(&[]),
            });

        let staging_texture = create_framebuffer(&device, &sc_desc, 1);

        let multisample_texture = create_framebuffer(&device, &sc_desc, SAMPLE_COUNT);

        let extract_render_pipeline = create_render_pipeline(
            &device,
            &extract_render_pipeline_layout,
            &sc_desc,
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.frag.spv")),
            &wgpu::vertex_attr_array![0 => Float2],
            1,
            2,
        );

        let blur_render_pipeline = create_render_pipeline(
            &device,
            &blur_render_pipeline_layout,
            &sc_desc,
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.frag.spv")),
            &wgpu::vertex_attr_array![0 => Float2, 1 => Float2],
            1,
            1,
        );

        let blend_render_pipeline = create_render_pipeline(
            &device,
            &blend_render_pipeline_layout,
            &sc_desc,
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/blend.frag.spv")),
            &wgpu::vertex_attr_array![0 => Float2],
            SAMPLE_COUNT,
            1,
        );

        let mut output_dir = std::path::PathBuf::new();
        output_dir.push(IMAGE_DIR);
        if !output_dir.is_dir() {
            std::fs::create_dir(output_dir.clone()).unwrap();
        }

        State {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,

            extract_render_pipeline,
            staging_texture,

            blur_bind_group_layout,
            blur_uniform_bind_group_layout,
            blur_uniform,
            blur_render_pipeline,
            blur_textures,

            blend_bind_group_layout,
            blend_uniform_bind_group_layout,
            blend_render_pipeline,

            multisample_texture,
            geometry,
            size,

            output_dir,

            frame: 0,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        println!("Resized to {:?}", new_size);
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);

        self.blur_textures = [
            create_framebuffer(&self.device, &self.sc_desc, 1),
            create_framebuffer(&self.device, &self.sc_desc, 1),
        ];
        self.staging_texture = create_framebuffer(&self.device, &self.sc_desc, 1);
        self.multisample_texture = create_framebuffer(&self.device, &self.sc_desc, SAMPLE_COUNT);
    }

    fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        self.frame += 1;
    }

    fn render(&mut self) {
        let frame = match self.swap_chain.get_current_frame() {
            Ok(frame) => frame,
            Err(_) => {
                self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
                self.swap_chain
                    .get_current_frame()
                    .expect("Failed to acquire next swap chain texture!")
            }
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.geometry.vertices),
                usage: wgpu::BufferUsage::VERTEX,
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.geometry.indices),
                usage: wgpu::BufferUsage::INDEX,
            });

        let blur_texture_views = [
            self.blur_textures[0].create_default_view(),
            self.blur_textures[1].create_default_view(),
        ];

        let staging_texture_view = self.staging_texture.create_default_view();

        // draw into staging buffer
        {
            let mut extract_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: Borrowed(&[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &staging_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &blur_texture_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    },
                ]),
                depth_stencil_attachment: None,
            });

            extract_render_pass.set_pipeline(&self.extract_render_pipeline);
            extract_render_pass.set_index_buffer(index_buffer.slice(..));
            extract_render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

            extract_render_pass.draw_indexed(0..(self.geometry.indices.len() as u32), 0, 0..1);
        }

        // Apply blur multiple times
        let vertex_square = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&VERTICES),
                usage: wgpu::BufferUsage::VERTEX,
            });

        let blur_count =
            (BLUR_COUNT as f32 * (1.0 + 0.3 * (self.frame as f32 / 83.0).sin())) as usize;
        for i in 0..blur_count {
            let bind_group = create_bind_group(
                &self.device,
                &self.blur_bind_group_layout,
                &self.blur_textures[i % 2],
            );

            let blur_uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&[self.blur_uniform]),
                        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
                    });

            let blur_uniform_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.blur_uniform_bind_group_layout,
                    entries: Borrowed(&[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(blur_uniform_buffer.slice(..)),
                    }]),
                    label: None,
                });
            self.blur_uniform.flip();

            let resolve_target = &self.blur_textures[(i + 1) % 2].create_default_view();
            {
                let mut blur_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: Borrowed(&[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: resolve_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    }]),
                    depth_stencil_attachment: None,
                });

                blur_render_pass.set_pipeline(&self.blur_render_pipeline);
                blur_render_pass.set_bind_group(0, &bind_group, &[]);
                blur_render_pass.set_bind_group(1, &blur_uniform_bind_group, &[]);
                blur_render_pass.set_vertex_buffer(0, vertex_square.slice(..));

                blur_render_pass.draw(0..VERTICES.len() as u32, 0..1);
            }
        }

        let sampler = &self
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let blend_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.blend_bind_group_layout,
            entries: Borrowed(&[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&staging_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &self.blur_textures[blur_count % 2].create_default_view(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ]),
            label: None,
        });

        let blend_uniform = BlendUniforms::new(EXPOSURE * (self.frame as f32 / 100.0).sin());

        let blend_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[blend_uniform]),
                    usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
                });

        let blend_uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.blend_uniform_bind_group_layout,
            entries: Borrowed(&[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(blend_uniform_buffer.slice(..)),
            }]),
            label: None,
        });

        let multisample_texture_view = &self.multisample_texture.create_default_view();

        {
            let mut blend_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: Borrowed(&[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: multisample_texture_view,
                    resolve_target: Some(&frame.output.view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }]),
                depth_stencil_attachment: None,
            });

            blend_render_pass.set_pipeline(&self.blend_render_pipeline);
            blend_render_pass.set_bind_group(0, &blend_bind_group, &[]);
            blend_render_pass.set_bind_group(1, &blend_uniform_bind_group, &[]);
            blend_render_pass.set_vertex_buffer(0, vertex_square.slice(..));

            blend_render_pass.draw(0..VERTICES.len() as u32, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));

        // TOOD:
        // if self.frame < 1000 {
        //     let file = self.output_dir.clone();
        //     create_png(file.join(format!("{:03}.png", self.frame)), &self.device, &);
        // }
    }
}

fn create_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    sample_count: u32,
) -> wgpu::Texture {
    let texture_extent = wgpu::Extent3d {
        width: sc_desc.width,
        height: sc_desc.height,
        depth: 1,
    };
    let frame_descriptor = &wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: sc_desc.format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        label: None,
    };

    device.create_texture(frame_descriptor)
}

fn create_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    staging_texture: &wgpu::Texture,
) -> wgpu::BindGroup {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    let staging_texture_view = staging_texture.create_default_view();

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: bind_group_layout,
        entries: Borrowed(&[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&staging_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ]),
        label: None,
    })
}

fn create_render_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    sc_desc: &wgpu::SwapChainDescriptor,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: &wgpu::ShaderModule,
    attr_array: &[wgpu::VertexAttributeDescriptor],
    sample_count: u32,
    n_color_states: u32,
) -> wgpu::RenderPipeline {
    let v: Vec<_> = (0..n_color_states)
        .map(|_| wgpu::ColorStateDescriptor {
            format: sc_desc.format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        })
        .collect();

    // Load shader modules.
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_mod,
            entry_point: Borrowed("main"),
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_mod,
            entry_point: Borrowed("main"),
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            ..Default::default()
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: Borrowed(&v.as_slice()),
        depth_stencil_state: None,
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: Borrowed(&[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: Borrowed(attr_array),
            }]),
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
}

// The original code is https://github.com/gfx-rs/wgpu-rs/blob/8e4d0015862507027f3a6bd68056c64568d11366/examples/capture/main.rs#L122-L194
async fn create_png(
    png_output_path: &str,
    device: wgpu::Device,
    output_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
) {
    let bytes_per_pixel = std::mem::size_of::<u32>();
    let unpadded_bytes_per_row = width as usize * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
    let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

    // Note that we're not calling `.await` here.
    let buffer_slice = output_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let padded_buffer = buffer_slice.get_mapped_range();

        let mut png_encoder = png::Encoder::new(
            std::fs::File::create(png_output_path).unwrap(),
            width,
            height,
        );
        png_encoder.set_depth(png::BitDepth::Eight);
        png_encoder.set_color(png::ColorType::RGBA);
        let mut png_writer = png_encoder
            .write_header()
            .unwrap()
            .into_stream_writer_with_size(unpadded_bytes_per_row);

        // from the padded_buffer we write just the unpadded bytes into the image
        for chunk in padded_buffer.chunks(padded_bytes_per_row) {
            png_writer.write(&chunk[..unpadded_bytes_per_row]).unwrap();
        }
        png_writer.finish().unwrap();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(padded_buffer);

        output_buffer.unmap();
    }
}

// main() is derived from sotrh/learn-wgpu
fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("test")
        .build(&event_loop)
        .unwrap();

    // Since main can't be async, we're going to need to block
    let mut state = block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &mut so w have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                state.render();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }
    });
}
