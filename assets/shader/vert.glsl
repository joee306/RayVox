#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in ivec2 resolution;
                                                                                                       
layout(location = 1) out ivec2 out_resolution;

void main() {
    out_resolution = resolution;
    gl_Position = vec4(position, 0.0, 1.0);
}
