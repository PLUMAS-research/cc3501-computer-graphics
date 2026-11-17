#version 330

in vec3 normal_mundo;
out vec4 outColor;

uniform vec3 color;

const vec3 direccion_luz = normalize(vec3(0.4, 1.0, 0.6));

void main()
{
    float difuso = max(dot(normalize(normal_mundo), direccion_luz), 0.0);
    outColor = vec4(color * (0.35 + 0.65 * difuso), 1.0);
}
