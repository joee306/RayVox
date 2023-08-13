mod render;

use render::Render;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    image::{view::ImageView, ImageAccess, SwapchainImage},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
    window::Window,
};

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct RayVVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SINT)]
    pub resolution: [u32; 2],
}

fn main() {
    let mut render = Render::new();
    let memory_allocator = StandardMemoryAllocator::new_default(render.device.clone());

    // We now create a buffer that will store the shape of our triangle. We use `#[repr(C)]` here
    // to force rustc to use a defined layout for our data, as the default representation has *no
    // guarantees*.

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "./assets/shader/vert.glsl"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "./assets/shader/frag.glsl"
        }
    }

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "./assets/shader/compute.glsl"
        }
    }

    let vs = vs::load(render.device.clone()).unwrap();
    let fs = fs::load(render.device.clone()).unwrap();
    let cs = cs::load(render.device.clone()).unwrap();

    let push_constants = cs::PushConstantData {
        camera_dir: [0.0, 0.0, 0.0].into(),
        camera_plane_u: [1.0, 0.0, 0.0].into(),
        camera_plane_v: [0.0, 1.0, 0.0].into(),
        position: [0.0, 0.0, -10.0].into(),
    };

    let data_buffer = {
        let data_iter = 0..16;
        Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            data_iter,
        )
    }
    .unwrap();

    let mut vertices = render::get_vertices_for_entire_screen();
    render.add_pipelines(
        GraphicsPipeline::start()
            .render_pass(Subpass::from(render.render_pass.clone(), 0).unwrap())
            .vertex_input_state(RayVVertex::per_vertex())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .build(render.device.clone())
            .unwrap(),
        ComputePipeline::new(
            render.device.clone(),
            cs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap(),
    );

    let mut framebuffers = window_size_dependent_setup(
        &render.images,
        render.render_pass.clone(),
        &mut render.viewport,
    );

    let mut recreate_swapchain = false;

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(render.device.clone(), Default::default());

    let mut previous_frame_end = Some(sync::now(render.device.clone()).boxed());

    render.event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                let window = render
                    .surface
                    .object()
                    .unwrap()
                    .downcast_ref::<Window>()
                    .unwrap();
                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }
                if recreate_swapchain {
                    for v in vertices.iter_mut() {
                        v.resolution = [dimensions.width, dimensions.height];
                    }
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match render.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..render.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("failed to recreate swapchain: {e}"),
                        };

                    render.swapchain = new_swapchain;

                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render.render_pass.clone(),
                        &mut render.viewport,
                    );

                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(render.swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let vertex_buffer = Buffer::from_iter(
                    &memory_allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        usage: MemoryUsage::Upload,
                        ..Default::default()
                    },
                    vertices.clone(),
                )
                .unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    render.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let descriptor_set_allocator =
                    StandardDescriptorSetAllocator::new(render.device.clone());
                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    render
                        .compute_pipeline
                        .clone()
                        .unwrap()
                        .layout()
                        .set_layouts()
                        .get(0)
                        .unwrap()
                        .clone(),
                    [
                        WriteDescriptorSet::image_view(0, render.image.clone()),
                        WriteDescriptorSet::buffer(1, data_buffer.clone()),
                    ],
                )
                .unwrap();

                builder
                    .bind_pipeline_compute(render.compute_pipeline.clone().unwrap().clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        render.compute_pipeline.clone().unwrap().layout().clone(),
                        0,
                        set,
                    )
                    .push_constants(
                        render.compute_pipeline.clone().unwrap().layout().clone(),
                        0,
                        push_constants,
                    )
                    .dispatch([dimensions.width / 16, dimensions.height / 8, 1])
                    .unwrap()
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .set_viewport(0, [render.viewport.clone()])
                    .bind_pipeline_graphics(render.graphics_pipeline.clone().unwrap().clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(render.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        render.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            render.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(render.device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
            _ => (),
        }
    });
}

/// This function is called once during initialization, then again whenever the window is resized.
pub fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
