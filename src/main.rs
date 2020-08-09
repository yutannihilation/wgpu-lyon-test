use std::io::Write;

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

// sample count for MSAA
const SAMPLE_COUNT: u32 = 4;

const IMAGE_DIR: &str = "img";

// how many times to repeat gaussian blur
const BLUR_COUNT: usize = 10;

// exposure level used in blend.frag
const EXPOSURE: f32 = 2.0;
// gamma correction used in blend.frag
const GAMMA: f32 = 1.0;

// Vertex for lines drawn by lyon
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    // color: [f32; 4],    // Use this when I want more colors
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

// Simple vertex to draw the texture identically as the original
#[repr(C)]
#[derive(Clone, Copy)]
struct BlurVertex {
    position: [f32; 2],
}

unsafe impl bytemuck::Pod for BlurVertex {}
unsafe impl bytemuck::Zeroable for BlurVertex {}

//   4-1
//   |/|
//   2-3
//
#[rustfmt::skip]
const VERTICES: &[BlurVertex] = &[
    BlurVertex { position: [ 1.0,  1.0], },
    BlurVertex { position: [-1.0, -1.0], },
    BlurVertex { position: [ 1.0, -1.0], },

    BlurVertex { position: [ 1.0,  1.0], },
    BlurVertex { position: [-1.0, -1.0], },
    BlurVertex { position: [-1.0,  1.0], },
];

// Parameters for gaussian blur;
// As gaussian blur is done horizontally and vertically repeadedly,
// we need a flag to flip the orientation.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
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

// Parameters for blending the original texture and the blurred texture
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct BlendUniforms {
    exposure: f32,
    gamma: f32,
}

impl BlendUniforms {
    fn new(exposure: f32, gamma: f32) -> Self {
        Self { exposure, gamma }
    }
}

unsafe impl bytemuck::Pod for BlendUniforms {}
unsafe impl bytemuck::Zeroable for BlendUniforms {}

// Dimension for writing out as PNG images. The original code is at https://github.com/gfx-rs/wgpu-rs/blob/master/examples/capture/main.rs
struct BufferDimensions {
    width: usize,
    height: usize,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl BufferDimensions {
    fn new(width: usize, height: usize) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>();
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

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
    square_vertex: wgpu::Buffer,

    // A render pipeline for blending the results
    blend_bind_group_layout: wgpu::BindGroupLayout,
    blend_uniform_bind_group_layout: wgpu::BindGroupLayout,
    blend_render_pipeline: wgpu::RenderPipeline,

    // Texture for MASS
    multisample_texture: wgpu::Texture,
    multisample_png_texture: wgpu::Texture, // a texture for PNG has a different TextureFormat, so we need another multisampled texture than others

    // Texture for writing out as PNG
    png_texture: wgpu::Texture,
    png_buffer: wgpu::Buffer,
    png_dimensions: BufferDimensions,

    geometry: VertexBuffers<Vertex, u16>,

    size: winit::dpi::PhysicalSize<u32>,

    output_dir: std::path::PathBuf,

    frame: u32,
    record: bool,
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
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        // Build a besier line VertexBuffer using lyon
        let geometry = create_lyon_geometry();

        // Extract ------------------------------------------------------------------------------------------------------------------
        //
        // Extract render pipeline just draw vertex buffer to two textures, so no bind groups are needed
        let extract_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: Borrowed(&[]),
                push_constant_ranges: Borrowed(&[]),
            });

        let extract_render_pipeline = create_render_pipeline(
            &device,
            &extract_render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/shader.frag.spv")),
            &wgpu::vertex_attr_array![0 => Float2],
            SAMPLE_COUNT,
            vec![sc_desc.format, sc_desc.format],
        );

        // Texture to draw the unmodified version
        let staging_texture = create_framebuffer(&device, &sc_desc, sc_desc.format);

        // Blur ----------------------------------------------------------------------------------------------------------------------
        //
        // Blur render pipeline needs two bind groups:
        //   - A bind group for texture containing the bright part of the drawing
        //   - A bind group for parameters of gaussian blur
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&[
                    wgpu::BindGroupLayoutEntry::new(
                        0,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            multisampled: true,
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

        let blur_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: Borrowed(&[
                    &blur_bind_group_layout,
                    &blur_uniform_bind_group_layout,
                ]),
                push_constant_ranges: Borrowed(&[]),
            });

        let blur_render_pipeline = create_render_pipeline(
            &device,
            &blur_render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.frag.spv")),
            &wgpu::vertex_attr_array![0 => Float2, 1 => Float2],
            SAMPLE_COUNT,
            vec![sc_desc.format],
        );

        // Parameters for blur
        let blur_uniform = BlurUniforms::new();

        // Textures to process gaussian blur in a ping-pong manner
        let blur_textures = [
            create_framebuffer(&device, &sc_desc, sc_desc.format),
            create_framebuffer(&device, &sc_desc, sc_desc.format),
        ];

        // Vetex to draw the texture identically
        let square_vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&VERTICES),
            usage: wgpu::BufferUsage::VERTEX,
        });

        // Blend ------------------------------------------------------------------------------------------------------------------
        //
        // Blend render pipeline needs two bind groups
        //   - A bind group for texture to blend
        //   - A bind group for parameters of blending
        let blend_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&[
                    wgpu::BindGroupLayoutEntry::new(
                        0,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            multisampled: true,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                        },
                    ),
                    wgpu::BindGroupLayoutEntry::new(
                        1,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            multisampled: true,
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

        // Parameters for blend
        let blend_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: Borrowed(&[wgpu::BindGroupLayoutEntry::new(
                    0,
                    wgpu::ShaderStage::FRAGMENT,
                    wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (std::mem::size_of::<f32>() * 2) as _,
                        ),
                    },
                )]),
                label: None,
            });

        let blend_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: Borrowed(&[
                    &blend_bind_group_layout,
                    &blend_uniform_bind_group_layout,
                ]),
                push_constant_ranges: Borrowed(&[]),
            });

        let blend_render_pipeline = create_render_pipeline(
            &device,
            &blend_render_pipeline_layout,
            &device.create_shader_module(wgpu::include_spirv!("shaders/blur.vert.spv")),
            &device.create_shader_module(wgpu::include_spirv!("shaders/blend.frag.spv")),
            &wgpu::vertex_attr_array![0 => Float2],
            SAMPLE_COUNT,
            vec![sc_desc.format, wgpu::TextureFormat::Rgba8UnormSrgb], // Texture to write out as PNG needs to be in RGBA format
        );

        // MSAA --------------------------------------------------------------------------------------------------------
        let multisample_texture =
            create_multisampled_framebuffer(&device, &sc_desc, sc_desc.format);

        let multisample_png_texture =
            create_multisampled_framebuffer(&device, &sc_desc, wgpu::TextureFormat::Rgba8UnormSrgb);

        // PNG output ----------------------------------------------------------------------------------------------------

        // Output dir
        let mut output_dir = std::path::PathBuf::new();
        output_dir.push(IMAGE_DIR);
        if !output_dir.is_dir() {
            std::fs::create_dir(output_dir.clone()).unwrap();
        }

        // PNG size, buffer, and texture
        let (png_dimensions, png_buffer, png_texture) =
            create_png_texture_and_buffer(&device, sc_desc.width as usize, sc_desc.height as usize);

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
            blur_render_pipeline,
            blur_uniform,
            blur_textures,
            square_vertex,

            blend_bind_group_layout,
            blend_uniform_bind_group_layout,
            blend_render_pipeline,

            multisample_texture,
            multisample_png_texture,

            png_texture,
            png_buffer,
            png_dimensions,

            geometry,
            size,

            output_dir,

            frame: 0,
            record: false,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        println!("Resized to {:?}", new_size);
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);

        self.blur_textures = [
            create_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format),
            create_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format),
        ];
        self.staging_texture = create_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format);
        self.multisample_texture =
            create_multisampled_framebuffer(&self.device, &self.sc_desc, self.sc_desc.format);
        self.multisample_png_texture = create_multisampled_framebuffer(
            &self.device,
            &self.sc_desc,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );

        let (png_dimensions, png_buffer, png_texture) = create_png_texture_and_buffer(
            &self.device,
            self.sc_desc.width as usize,
            self.sc_desc.height as usize,
        );

        self.png_dimensions = png_dimensions;
        self.png_buffer = png_buffer;
        self.png_texture = png_texture;
    }

    fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        self.frame += 1;
        if self.frame > 1000 {
            println!("End recording");
            self.frame = 0;
            self.record = false;
        }
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

        // Texture views
        let staging_texture_view = self.staging_texture.create_default_view();
        let blur_texture_views = [
            self.blur_textures[0].create_default_view(),
            self.blur_textures[1].create_default_view(),
        ];
        let multisample_texture_view = self.multisample_texture.create_default_view();

        // A sampler for textures
        let sampler = &self
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        // Extract ------------------------------------------------------------------------------------------------------------------

        {
            let mut extract_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: Borrowed(&[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &multisample_texture_view,
                        resolve_target: Some(&staging_texture_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &multisample_texture_view,
                        resolve_target: Some(&blur_texture_views[0]),
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

        // Blur ------------------------------------------------------------------------------------------------------------------

        // Apply blur multiple times
        let blur_count =
            (BLUR_COUNT as f32 * (1.0 + 0.5 * (self.frame as f32 / 300.0).sin())) as usize;
        for i in 0..blur_count {
            let src_texture = &self.blur_textures[i % 2];
            let dst_texture = &self.blur_textures[(i + 1) % 2];

            let bind_group = create_bind_group(
                &self.device,
                &self.blur_bind_group_layout,
                src_texture,
                &sampler,
            );

            let blur_uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&[self.blur_uniform]),
                        usage: wgpu::BufferUsage::UNIFORM,
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

            // Flip the orientation between horizontally and vertically
            self.blur_uniform.flip();

            let resolve_target = dst_texture.create_default_view();
            {
                let mut blur_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: Borrowed(&[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &multisample_texture_view,
                        resolve_target: Some(&resolve_target),
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
                blur_render_pass.set_vertex_buffer(0, self.square_vertex.slice(..));

                blur_render_pass.draw(0..VERTICES.len() as u32, 0..1);
            }
        }

        // Blend ------------------------------------------------------------------------------------------------------------------

        let blend_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.blend_bind_group_layout,
            entries: Borrowed(&[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // a texture that contains the unmodified version
                    resource: wgpu::BindingResource::TextureView(&staging_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    // a texture that contains the last result of gaussian blur
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

        let blend_uniform = BlendUniforms::new(
            EXPOSURE * (1.0 + 0.3 * self.frame as f32 / 30.0).sin(),
            GAMMA,
        );

        let blend_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[blend_uniform]),
                    usage: wgpu::BufferUsage::UNIFORM,
                });

        let blend_uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.blend_uniform_bind_group_layout,
            entries: Borrowed(&[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(blend_uniform_buffer.slice(..)),
            }]),
            label: None,
        });

        let png_texture_view = &self.png_texture.create_default_view();
        let multisample_png_texture_view = self.multisample_png_texture.create_default_view();

        {
            let mut blend_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: Borrowed(&[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &multisample_texture_view,
                        resolve_target: Some(&frame.output.view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &multisample_png_texture_view,
                        resolve_target: Some(&png_texture_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    },
                ]),
                depth_stencil_attachment: None,
            });

            blend_render_pass.set_pipeline(&self.blend_render_pipeline);
            blend_render_pass.set_bind_group(0, &blend_bind_group, &[]);
            blend_render_pass.set_bind_group(1, &blend_uniform_bind_group, &[]);
            blend_render_pass.set_vertex_buffer(0, self.square_vertex.slice(..));

            blend_render_pass.draw(0..VERTICES.len() as u32, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &self.png_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &self.png_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: self.png_dimensions.padded_bytes_per_row as u32,
                    rows_per_image: 0,
                },
            },
            wgpu::Extent3d {
                width: self.sc_desc.width,
                height: self.sc_desc.height,
                depth: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        if self.record {
            let file = self.output_dir.clone();
            block_on(create_png(
                &file
                    .join(format!("{:03}.png", self.frame))
                    .to_str()
                    .unwrap(),
                &self.device,
                &self.png_buffer,
                &self.png_dimensions,
            ))
        }
    }
}

fn create_lyon_geometry() -> VertexBuffers<Vertex, u16> {
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
            &mut BuffersBuilder::new(&mut geometry, |vertex: tessellation::StrokeVertex| Vertex {
                position: vertex.position().to_array(),
            }),
        )
        .unwrap();

    geometry
}

fn create_texture(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    sample_count: u32,
    usage: wgpu::TextureUsage,
    format: wgpu::TextureFormat,
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
        format: format,
        usage,
        label: None,
    };

    device.create_texture(frame_descriptor)
}

fn create_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    create_texture(
        device,
        sc_desc,
        1,
        wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        format,
    )
}

fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    create_texture(
        device,
        sc_desc,
        SAMPLE_COUNT,
        wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format,
    )
}

fn create_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    staging_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
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
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ]),
        label: None,
    })
}

fn create_render_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: &wgpu::ShaderModule,
    attr_array: &[wgpu::VertexAttributeDescriptor],
    sample_count: u32,
    formats: Vec<wgpu::TextureFormat>,
) -> wgpu::RenderPipeline {
    let v: Vec<_> = formats
        .iter()
        .map(|format| wgpu::ColorStateDescriptor {
            format: *format,
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

fn create_png_texture_and_buffer(
    device: &wgpu::Device,
    width: usize,
    height: usize,
) -> (BufferDimensions, wgpu::Buffer, wgpu::Texture) {
    let png_dimensions = BufferDimensions::new(width, height);
    // The output buffer lets us retrieve the data as an array
    let png_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (png_dimensions.padded_bytes_per_row * png_dimensions.height) as u64,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    // The render pipeline renders data into this texture
    let png_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: png_dimensions.width as u32,
            height: png_dimensions.height as u32,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: None,
    });

    (png_dimensions, png_buffer, png_texture)
}

// The original code is https://github.com/gfx-rs/wgpu-rs/blob/8e4d0015862507027f3a6bd68056c64568d11366/examples/capture/main.rs#L122-L194
async fn create_png(
    png_output_path: &str,
    device: &wgpu::Device,
    output_buffer: &wgpu::Buffer,
    buffer_dimensions: &BufferDimensions,
) {
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
            buffer_dimensions.width as u32,
            buffer_dimensions.height as u32,
        );
        png_encoder.set_depth(png::BitDepth::Eight);
        png_encoder.set_color(png::ColorType::RGBA);
        png_encoder.set_compression(png::Compression::Fast);
        let mut png_writer = png_encoder
            .write_header()
            .unwrap()
            .into_stream_writer_with_size(buffer_dimensions.unpadded_bytes_per_row);

        // from the padded_buffer we write just the unpadded bytes into the image
        for chunk in padded_buffer.chunks(buffer_dimensions.padded_bytes_per_row) {
            png_writer
                .write(&chunk[..buffer_dimensions.unpadded_bytes_per_row])
                .unwrap();
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
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::R),
                                ..
                            } => {
                                println!("Start recording");
                                state.record = true
                            }
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
