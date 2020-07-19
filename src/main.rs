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

const SAMPLE_COUNT: u32 = 1;

// The vertex type that we will use to represent a point on our triangle.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

// State is derived from sotrh/learn-wgpu
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,

    // vertex_buffer: wgpu::Buffer,
    // index_buffer: wgpu::Buffer,
    // index_count: usize,
    // bind_group: wgpu::BindGroup,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    staging_render_pipeline: wgpu::RenderPipeline,
    blur_render_pipeline: wgpu::RenderPipeline,
    staging_texture: wgpu::Texture,
    geometry: VertexBuffers<Vertex, u16>,
    stroke_range: std::ops::Range<u32>,

    size: winit::dpi::PhysicalSize<u32>,
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
            // TODO: Allow srgb unconditionally
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
                entries: &[
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
                ],
                label: Some("staging"),
            });

        // For staging buffer, we don't use bindgroups
        let staging_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        // For blur, we do use bind_group_layout
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            // bind_group_layouts: &[&blur_bind_group_layout],
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let staging_texture = create_framebuffer(&device, &sc_desc);
        // let bind_group = create_bind_group(&device, &blur_bind_group_layout, &staging_texture);

        let staging_render_pipeline = create_render_pipeline(
            &device,
            &staging_pipeline_layout,
            &sc_desc,
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.frag.spv")),
        );

        let blur_render_pipeline = create_render_pipeline(
            &device,
            &blur_pipeline_layout,
            &sc_desc,
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.frag.spv")),
        );

        // extend the index data to the alined size
        let stroke_range = 0..(geometry.indices.len() as u32);
        let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize / std::mem::size_of::<u16>();
        let fraction = geometry.indices.len() % alignment;
        if fraction > 0 {
            geometry
                .indices
                .extend(std::iter::repeat(0).take(alignment - fraction));
        }

        State {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,

            // vertex_buffer,
            // index_buffer,
            // index_count,
            // bind_group,
            blur_bind_group_layout,
            staging_render_pipeline,
            staging_texture,
            blur_render_pipeline,
            geometry,
            stroke_range,
            size,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        println!("Resized to {:?}", new_size);
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);

        self.staging_texture = create_framebuffer(&self.device, &self.sc_desc);
    }

    fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        let frame = match self.swap_chain.get_next_frame() {
            Ok(frame) => frame,
            Err(_) => {
                self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
                self.swap_chain
                    .get_next_frame()
                    .expect("Failed to acquire next swap chain texture!")
            }
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let vertex_buffer = self.device.create_buffer_with_data(
            bytemuck::cast_slice(&self.geometry.vertices),
            wgpu::BufferUsage::VERTEX,
        );

        let index_buffer = self.device.create_buffer_with_data(
            bytemuck::cast_slice(&self.geometry.indices),
            wgpu::BufferUsage::INDEX,
        );

        let staging_texture_view = self.staging_texture.create_default_view();
        // draw staging buffer
        {
            let mut staging_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &staging_texture_view,
                    // attachment: &frame.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.01,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            staging_render_pass.set_pipeline(&self.staging_render_pipeline);
            staging_render_pass.set_index_buffer(index_buffer.slice(..));
            staging_render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

            staging_render_pass.draw_indexed(self.stroke_range.clone(), 0, 0..1);
        }

        let bind_group = create_bind_group(
            &self.device,
            &self.blur_bind_group_layout,
            &self.staging_texture,
        );

        {
            let mut blur_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.01,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            blur_render_pass.set_pipeline(&self.blur_render_pipeline);
            blur_render_pass.set_bind_group(0, &bind_group, &[]);
            blur_render_pass.set_index_buffer(index_buffer.slice(..));
            blur_render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

            blur_render_pass.draw_indexed(self.stroke_range.clone(), 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

fn create_framebuffer(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor) -> wgpu::Texture {
    let texture_extent = wgpu::Extent3d {
        width: sc_desc.width,
        height: sc_desc.height,
        depth: 1,
    };
    let frame_descriptor = &wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: SAMPLE_COUNT,
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

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &staging_texture.create_default_view(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
        label: Some("staging"),
    })
}

fn create_render_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    sc_desc: &wgpu::SwapChainDescriptor,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    // Load shader modules.
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_mod,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_mod,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: sc_desc.format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float2],
            }],
        },
        sample_count: SAMPLE_COUNT,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
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
