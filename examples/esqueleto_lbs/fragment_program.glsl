#version 330

// Sombreado difuso de dos caras (abs en el producto punto) sobre el color
// de mezcla de pesos que llega del vertex shader. La luz es direccional y
// fija: lo unico que cambia en el ejemplo es la pose del esqueleto.

in vec3 world_normal;
in vec3 weight_color;

uniform vec3 light_direction;
uniform float ambient_strength;

out vec4 out_color;

void main()
{
    vec3 n = normalize(world_normal);
    float diffuse = abs(dot(n, normalize(light_direction)));
    float lighting = ambient_strength + (1.0 - ambient_strength) * diffuse;
    out_color = vec4(weight_color * lighting, 1.0);
}
