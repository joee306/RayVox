use crate::app::FractalApp;
use vulkano::{image::ImageUsage, swapchain::PresentMode, sync::GpuFuture};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::{VulkanoWindowRenderer, DEFAULT_IMAGE_FORMAT},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

mod app;
mod fractal_compute_pipeline;
mod pixels_draw_pipeline;
mod place_over_frame;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 1 {
        println!("no render distance");
        return;
    }
    let render_distance = match args[1].parse::<u32>() {
        Ok(v) => v,
        Err(err) => panic!("{}", err),
    };
    let mut event_loop = EventLoop::new();
    let context = VulkanoContext::new(VulkanoConfig::default());
    let mut windows = VulkanoWindows::default();
    let _id = windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor {
            title: "RayVox".to_string(),
            present_mode: PresentMode::Fifo,
            ..Default::default()
        },
        |_| {},
    );

    let render_target_id = 0;
    let primary_window_renderer = windows.get_primary_renderer_mut().unwrap();

    primary_window_renderer.add_additional_image_view(
        render_target_id,
        DEFAULT_IMAGE_FORMAT,
        ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
    );

    let gfx_queue = context.graphics_queue();

    let mut app = FractalApp::new(
        gfx_queue.clone(),
        primary_window_renderer.swapchain_format(),
        render_distance,
    );
    loop {
        if !handle_events(&mut event_loop, primary_window_renderer, &mut app) {
            break;
        }

        match primary_window_renderer.window_size() {
            [w, h] => {
                if w == 0.0 || h == 0.0 {
                    continue;
                }
            }
        }

        app.update_state_after_inputs(primary_window_renderer);
        compute_then_render(primary_window_renderer, &mut app, render_target_id);
        app.reset_input_state();
        app.update_time();
        primary_window_renderer.window().set_title(&format!(
            "RayVox [fps: {:.2} dt: {:.2}]",
            app.avg_fps(),
            app.dt(),
        ));
    }
}

fn handle_events(
    event_loop: &mut EventLoop<()>,
    renderer: &mut VulkanoWindowRenderer,
    app: &mut FractalApp,
) -> bool {
    let mut is_running = true;

    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match &event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => is_running = false,
                WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                    renderer.resize()
                }
                _ => (),
            },
            Event::MainEventsCleared => *control_flow = ControlFlow::Exit,
            _ => (),
        }

        app.handle_input(renderer.window_size(), &event);
    });

    is_running && app.is_running()
}

fn compute_then_render(
    renderer: &mut VulkanoWindowRenderer,
    app: &mut FractalApp,
    target_image_id: usize,
) {
    let before_pipeline_future = match renderer.acquire() {
        Err(e) => {
            println!("{e}");
            return;
        }
        Ok(future) => future,
    };

    let image = renderer.get_additional_image_view(target_image_id);

    let after_compute = app.compute(image.clone()).join(before_pipeline_future);

    let after_renderpass_future =
        app.place_over_frame
            .render(after_compute, image, renderer.swapchain_image_view());

    renderer.present(after_renderpass_future, true);
}
