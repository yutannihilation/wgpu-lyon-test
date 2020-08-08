#version 450

layout(location = 0) in vec2 a_position;

layout(location = 0) out vec2 v_tex_coords;

void main() {
    v_tex_coords = 0.5 + vec2(0.5, -0.5) * a_position;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
