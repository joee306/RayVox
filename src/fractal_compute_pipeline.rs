use rand::Rng;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::ImageAccess,
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};
use vulkano_util::renderer::DeviceImageView;

pub struct Controller {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    //memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    world_buffer: Subbuffer<[[[u32; 256]; 256]]>,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub render_distance: u32,
}

impl Controller {
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        render_distance: u32,
    ) -> Self {
        let mut world = vec![[[0; 256]; 256]; 256];
        for x in 0..250 {
            for y in 0..250 {
                for z in 0..250 {
                    if rand::thread_rng().gen_range(1..20) == 1 {
                        world[x][y][z] = rand::thread_rng().gen_range(1..10);
                    }
                }
            }
        }
        let world_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            world,
        )
        .unwrap();
        let pipeline = {
            let shader = cs::load(queue.device().clone()).unwrap();
            ComputePipeline::new(
                queue.device().clone(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };

        Self {
            queue,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
            world_buffer,
            position: [0.0, 0.0, -10.0],
            rotation: [0.0, 0.0, 0.0],
            render_distance,
        }
    }

    pub fn compute(&self, image: DeviceImageView) -> Box<dyn GpuFuture> {
        let img_dims = image.image().dimensions().width_height();
        let pipeline_layout = self.pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, image),
                WriteDescriptorSet::buffer(1, self.world_buffer.clone()),
            ],
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let push_constants = cs::PushConstants {
            resolution: img_dims.into(),
            camera_dir: [0.0, 0.0, 0.8].into(),
            rotation: self.rotation.into(),
            position: self.position.into(),
            render_distance: self.render_distance,
        };
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .push_constants(pipeline_layout.clone(), 0, push_constants)
            .dispatch([img_dims[0] / 16, img_dims[1] / 16, 1])
            .unwrap();
        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }
}

mod cs {
    vulkano_shaders::shader! {
         ty: "compute",
         path: "./assets/shader/compute.glsl"
    }
}
