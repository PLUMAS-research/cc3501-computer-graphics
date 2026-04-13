#version 330

in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform vec3 color_cool;
uniform vec3 color_warm;

out vec3 fragColor;

void main()
{
    // Gradiente frío-a-cálido según la componente Y de la normal en espacio del modelo.
    // Cada conejo recibe su propio par de colores como uniform.
    float blend = 0.5 + 0.5 * normal.y;
    fragColor = mix(color_cool, color_warm, blend);
    gl_Position = transform * vec4(position, 1.0f);
}
