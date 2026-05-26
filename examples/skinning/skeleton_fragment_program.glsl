#version 330

uniform vec3 line_color;

out vec4 out_color;

void main()
{
    out_color = vec4(line_color, 1.0);
}
