#version 330

uniform vec3 emission_color;

out vec3 out_color;

void main()
{
    out_color = emission_color;
}
