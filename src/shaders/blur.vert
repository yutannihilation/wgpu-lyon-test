
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(clamp(position, -1.0, 1.0), 0.0, 1.0);
}