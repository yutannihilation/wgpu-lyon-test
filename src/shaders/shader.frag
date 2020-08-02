#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 b_color;

void main() {
    f_color = vec4(1.0, 0.8, 0.8, 1.0);
    b_color = f_color;
    // float brightness = dot(f_color.rgb, vec3(0.2126, 0.7152, 0.0722));
    // if (brightness > 1.0)
    //     b_color = vec4(f_color.rgb, 1.0);
    // else
    //     b_color = vec4(0.0, 0.0, 0.0, 1.0);
}
