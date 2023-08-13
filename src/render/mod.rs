use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::AutoCommandBufferBuilder,
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{ImageUsage, StorageImage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{graphics::viewport::Viewport, ComputePipeline, GraphicsPipeline},
    render_pass::{Framebuffer, RenderPass},
    swapchain::{AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError},
    sync, VulkanLibrary,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::DeviceImageView,
};
use vulkano_win::VkSurfaceBuild;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::RayVVertex as Vertex;

pub struct Render {
    pub instance: Arc<Instance>,
    pub event_loop: EventLoop<()>,
    pub library: Arc<VulkanLibrary>,
    pub graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    pub compute_pipeline: Option<Arc<ComputePipeline>>,
    pub surface: Arc<Surface>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue_family_index: u32,
    pub queue: Arc<Queue>,
    pub swapchain: Arc<Swapchain>,
    pub render_pass: Arc<RenderPass>,
    pub viewport: Viewport,
    pub images: Vec<Arc<SwapchainImage>>,
    pub image: DeviceImageView,
    pub context: VulkanoContext,
}

impl Render {
    pub fn new() -> Self {
        let library = VulkanLibrary::new().unwrap();

        let required_extensions = vulkano_win::required_extensions(&library);

        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                // Enable enumerating devices that use non-conformant Vulkan implementations. (e.g.
                // MoltenVK)
                enumerate_portability: true,
                ..Default::default()
            },
        )
        .unwrap();

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        // Choose device extensions that we're going to use. In order to present images to a surface,
        // we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),

                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.1..1.0,
        };

        let context = VulkanoContext::new(VulkanoConfig::default());

        let image = StorageImage::general_purpose_image_view(
            context.memory_allocator(),
            queue.clone(),
            [1920, 1080],
            Format::R8G8B8A8_UNORM,
            ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
        )
        .unwrap();

        Self {
            instance,
            event_loop,
            library,
            graphics_pipeline: None,
            compute_pipeline: None,
            surface,
            physical_device,
            device,
            queue_family_index,
            queue,
            swapchain,
            render_pass,
            viewport,
            images,
            image,
            context,
        }
    }
    pub fn add_pipelines(
        &mut self,
        graphics_pipeline: Arc<GraphicsPipeline>,
        compute_pipeline: Arc<ComputePipeline>,
    ) {
        self.graphics_pipeline = Some(graphics_pipeline);
        self.compute_pipeline = Some(compute_pipeline);
    }
}

pub fn get_vertices_for_entire_screen() -> Vec<Vertex> {
    vec![
        Vertex {
            // a
            position: [-1.0, -1.0],
            resolution: [1046, 1025],
        },
        Vertex {
            // b
            position: [1.0, -1.0],
            resolution: [1046, 1025],
        },
        Vertex {
            // c
            position: [1.0, 1.0],
            resolution: [1046, 1025],
        },
        Vertex {
            // a
            position: [-1.0, -1.0],
            resolution: [1046, 1025],
        },
        Vertex {
            //c
            position: [1.0, 1.0],
            resolution: [1046, 1025],
        },
        Vertex {
            //d
            position: [-1.0, 1.0],
            resolution: [1046, 1025],
        },
    ]
}
