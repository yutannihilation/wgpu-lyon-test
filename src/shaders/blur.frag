#version 450

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_staging;
layout(set = 0, binding = 1) uniform sampler s_staging;

vec2 off1 = vec2(1.3333333333333333);
vec2 resolution = vec2(1000);

void main() {
    // f_color = texture(sampler2D(t_staging, s_staging), v_tex_coords);
    f_color = vec4(0.0);
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords) * 0.29411764705882354;
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords + (off1 / resolution)) * 0.35294117647058826;
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords - (off1 / resolution)) * 0.35294117647058826;
}
