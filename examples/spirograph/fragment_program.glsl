#version 330

in vec3 fragment_color;
out vec4 out_color;

void main()
{
    out_color = vec4(fragment_color, 1.0);
}
