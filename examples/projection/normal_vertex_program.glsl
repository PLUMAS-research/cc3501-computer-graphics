#version 330

in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main()
{
    // Gradiente frío-cálido según la componente y de la normal en espacio del modelo.
    // y ≈ +1 (superficies mirando hacia arriba) → color cálido
    // y ≈ -1 (superficies mirando hacia abajo)  → color frío
    float blend = 0.5 + 0.5 * normal.y;
    vec3 color_cool = vec3(0.08, 0.18, 0.72);
    vec3 color_warm = vec3(0.95, 0.45, 0.05);
    fragColor = mix(color_cool, color_warm, blend);

    gl_Position = projection * view * transform * vec4(position, 1.0f);
}
