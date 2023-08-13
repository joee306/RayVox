use crate::{fractal_compute_pipeline::Controller, place_over_frame::RenderPassPlaceOverFrame};
use cgmath::Vector2;
use std::{sync::Arc, time::Instant};
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator, device::Queue,
    memory::allocator::StandardMemoryAllocator, sync::GpuFuture,
};
use vulkano_util::{
    renderer::{DeviceImageView, VulkanoWindowRenderer},
    window::WindowDescriptor,
};
use winit::{
    dpi::PhysicalPosition,
    event::{
        ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
    window::Fullscreen,
};

pub struct FractalApp {
    controller_pipeline: Controller,
    pub place_over_frame: RenderPassPlaceOverFrame,
    time: Instant,
    dt: f32,
    dt_sum: f32,
    frame_count: f32,
    avg_fps: f32,
    input_state: InputState,
}

impl FractalApp {
    pub fn new(
        gfx_queue: Arc<Queue>,
        image_format: vulkano::format::Format,
        render_distance: u32,
    ) -> FractalApp {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            gfx_queue.device().clone(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            gfx_queue.device().clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            gfx_queue.device().clone(),
        ));

        FractalApp {
            controller_pipeline: Controller::new(
                gfx_queue.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                render_distance,
            ),
            place_over_frame: RenderPassPlaceOverFrame::new(
                gfx_queue,
                &memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
                image_format,
            ),
            time: Instant::now(),
            dt: 0.0,
            dt_sum: 0.0,
            frame_count: 0.0,
            avg_fps: 0.0,
            input_state: InputState::new(),
        }
    }

    /// Runs our compute pipeline and return a future of when the compute is finished.
    pub fn compute(&self, image_target: DeviceImageView) -> Box<dyn GpuFuture> {
        self.controller_pipeline.compute(image_target)
    }

    /// Returns whether the app should quit. (Happens on when pressing ESC.)
    pub fn is_running(&self) -> bool {
        !self.input_state.should_quit
    }

    /// Returns the average FPS.
    pub fn avg_fps(&self) -> f32 {
        self.avg_fps
    }

    /// Returns the delta time in milliseconds.
    pub fn dt(&self) -> f32 {
        self.dt * 1000.0
    }

    /// Updates times and dt at the end of each frame.
    pub fn update_time(&mut self) {
        // Each second, update average fps & reset frame count & dt sum.
        if self.dt_sum > 1.0 {
            self.avg_fps = self.frame_count / self.dt_sum;
            self.frame_count = 0.0;
            self.dt_sum = 0.0;
        }
        self.dt = self.time.elapsed().as_secs_f32();
        self.dt_sum += self.dt;
        self.frame_count += 1.0;
        self.time = Instant::now();
    }

    pub fn handle_input(&mut self, window_size: [f32; 2], event: &Event<()>) {
        self.input_state.handle_input(window_size, event);
    }

    /// Reset input state at the end of the frame.
    pub fn reset_input_state(&mut self) {
        self.input_state.reset()
    }
    pub fn update_state_after_inputs(&mut self, renderer: &mut VulkanoWindowRenderer) {
        if self.input_state.forward {
            self.controller_pipeline.position[2] += 5.0 * self.dt * self.input_state.move_speed;
        }
        if self.input_state.backward {
            self.controller_pipeline.position[2] -= 5.0 * self.dt * self.input_state.move_speed;
        }
        if self.input_state.left {
            self.controller_pipeline.position[0] -= 5.0 * self.dt * self.input_state.move_speed;
        }
        if self.input_state.right {
            self.controller_pipeline.position[0] += 5.0 * self.dt * self.input_state.move_speed;
        }
        if self.input_state.up {
            self.controller_pipeline.position[1] += 5.0 * self.dt * self.input_state.move_speed;
        }
        if self.input_state.down {
            self.controller_pipeline.position[1] -= 5.0 * self.dt * self.input_state.move_speed;
        }
        if self.input_state.mouse_pos.x == 0.1 {
            self.controller_pipeline.rotation[0] += 0.05;
            self.input_state.mouse_pos.x = 0.0;
        }
        if self.input_state.mouse_pos.x == -0.1 {
            self.controller_pipeline.rotation[0] -= 0.05;
            self.input_state.mouse_pos.x = 0.0;
        }
        if self.input_state.mouse_pos.y == 0.1 {
            self.controller_pipeline.rotation[2] += 0.05;
            self.input_state.mouse_pos.y = 0.0;
        }
        if self.input_state.mouse_pos.y == -0.1 {
            self.controller_pipeline.rotation[2] -= 0.05;
            self.input_state.mouse_pos.y = 0.0;
        }
        if self.input_state.toggle_full_screen {
            let is_full_screen = renderer.window().fullscreen().is_some();
            renderer.window().set_fullscreen(if !is_full_screen {
                Some(Fullscreen::Borderless(renderer.window().current_monitor()))
            } else {
                None
            });
        }
    }
}

fn state_is_pressed(state: ElementState) -> bool {
    match state {
        ElementState::Pressed => true,
        ElementState::Released => false,
    }
}

struct InputState {
    pub window_size: [f32; 2],
    pub forward: bool,
    pub backward: bool,
    pub right: bool,
    pub left: bool,
    pub up: bool,
    pub down: bool,
    pub toggle_full_screen: bool,
    pub should_quit: bool,
    pub move_speed: f32,
    pub mouse_pos: Vector2<f32>,
}

impl InputState {
    fn new() -> InputState {
        InputState {
            window_size: [
                WindowDescriptor::default().width,
                WindowDescriptor::default().height,
            ],
            forward: false,
            backward: false,
            right: false,
            left: false,
            up: false,
            down: false,
            toggle_full_screen: false,
            should_quit: false,
            move_speed: 1.0,
            mouse_pos: Vector2::new(0.0, 0.0),
        }
    }

    /*fn normalized_mouse_pos(&self) -> Vector2<f32> {
        Vector2::new(
            (self.mouse_pos.x / self.window_size[0]).clamp(0.0, 1.0),
            (self.mouse_pos.y / self.window_size[1]).clamp(0.0, 1.0),
        )
    }*/

    fn reset(&mut self) {
        *self = InputState {
            toggle_full_screen: false,
            ..*self
        }
    }

    fn handle_input(&mut self, window_size: [f32; 2], event: &Event<()>) {
        self.window_size = window_size;
        if let winit::event::Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::KeyboardInput { input, .. } => self.on_keyboard_event(input),
                WindowEvent::MouseInput { state, button, .. } => {
                    self.on_mouse_click_event(*state, *button)
                }
                WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved_event(position),
                WindowEvent::MouseWheel { delta, .. } => self.on_mouse_wheel_event(delta),
                _ => {}
            }
        }
    }

    fn on_keyboard_event(&mut self, input: &KeyboardInput) {
        if let Some(key_code) = input.virtual_keycode {
            match key_code {
                VirtualKeyCode::Escape => self.should_quit = state_is_pressed(input.state),
                VirtualKeyCode::W => self.forward = state_is_pressed(input.state),
                VirtualKeyCode::A => self.left = state_is_pressed(input.state),
                VirtualKeyCode::S => self.backward = state_is_pressed(input.state),
                VirtualKeyCode::D => self.right = state_is_pressed(input.state),
                VirtualKeyCode::Space => self.up = state_is_pressed(input.state),
                VirtualKeyCode::LControl => self.down = state_is_pressed(input.state),
                VirtualKeyCode::RShift => self.toggle_full_screen = state_is_pressed(input.state),
                VirtualKeyCode::Up => self.mouse_pos.y += 0.1,
                VirtualKeyCode::Down => self.mouse_pos.y -= 0.1,
                VirtualKeyCode::Left => self.mouse_pos.x += 0.1,
                VirtualKeyCode::Right => self.mouse_pos.x -= 0.1,
                _ => (),
            }
        }
    }
    fn on_mouse_wheel_event(&mut self, delta: &MouseScrollDelta) {
        let change = match delta {
            MouseScrollDelta::LineDelta(_x, y) => *y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        self.move_speed += change;
    }
    fn on_cursor_moved_event(&mut self, pos: &PhysicalPosition<f64>) {
        self.mouse_pos = Vector2::new(pos.x as f32, pos.y as f32);
    }
    fn on_mouse_click_event(&mut self, state: ElementState, mouse_btn: winit::event::MouseButton) {
        if mouse_btn == MouseButton::Right {}
    }
}
