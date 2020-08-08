#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 b_color;

void main() {
    f_color = vec4(0.8, 0.4, 0.4, 1.0);
    float brightness = dot(f_color.rgb, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > 0.1)
        b_color = vec4(f_color.rgb, 1.0);
    else
        b_color = vec4(0.0, 0.0, 0.0, 1.0);
}
