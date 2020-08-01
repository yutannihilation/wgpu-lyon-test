#version 450

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_staging;
layout(set = 0, binding = 1) uniform sampler s_staging;

vec2 off1 = vec2(1.0);
vec2 off2 = vec2(2.0);

void main() {
    vec2 resolution = textureSize(sampler2D(t_staging, s_staging), 0).xy;
    // f_color = texture(sampler2D(t_staging, s_staging), v_tex_coords);

    // Based on the implementation of https://github.com/Jam3/glsl-fast-gaussian-blur/blob/master/9.glsl
    f_color = vec4(0.0);
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords) * 0.2270270270;
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords + (off1 / resolution)) * 0.3162162162;
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords - (off1 / resolution)) * 0.3162162162;
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords + (off2 / resolution)) * 0.0702702703;
    f_color += texture(sampler2D(t_staging, s_staging), v_tex_coords - (off2 / resolution)) * 0.0702702703;
}
