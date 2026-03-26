#version 330

uniform vec2 resolution;

in vec2 position;
in vec3 color;

out vec3 fragColor;

void main()
{
    vec2 screen_pos = position / resolution * 2.0 - 1.0;
    gl_Position = vec4(screen_pos, 0.0, 1.0);
    fragColor = color;
}
