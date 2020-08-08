#version 450

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;     // frame
layout(location = 1) out vec4 png_color;   // buffer to save as png

layout(set = 0, binding = 0) uniform texture2D t_base;
layout(set = 0, binding = 1) uniform texture2D t_bloom;
layout(set = 0, binding = 2) uniform sampler s_base;

const float gamma = 2.2;

layout(set = 1, binding = 0)
uniform Uniforms {
    float exposure;
};

void main() {
    vec3 hdr_color = texture(sampler2D(t_base, s_base), v_tex_coords).rgb;      
    vec3 bloom_color = texture(sampler2D(t_bloom, s_base), v_tex_coords).rgb;

    // additive blending
    hdr_color += bloom_color;

    // tone mapping
    vec3 result = vec3(1.0) - exp(-hdr_color * exposure);

    // also gamma correct while we're at it       
    result = pow(result, vec3(1.0 / gamma));
    f_color = vec4(result, 1.0);

    // just copy the colors
    png_color = f_color;
}
