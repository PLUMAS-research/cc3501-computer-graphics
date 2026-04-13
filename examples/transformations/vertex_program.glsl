#version 330

in vec3 position;
in vec3 normal;

uniform mat4 view_transform;
uniform float time;
uniform float homogeneous_strength;

out vec3 fragColor;

void main()
{
    // Gradiente frío-a-cálido según la dirección Y de la normal en espacio del modelo
    float blend = 0.5 + 0.5 * normal.y;
    vec3 color_cool = vec3(
        0.08 + 0.06 * sin(time * 0.19),
        0.18 + 0.10 * sin(time * 0.13),
        0.72 + 0.12 * sin(time * 0.17)
    );
    vec3 color_warm = vec3(
        0.95 + 0.04 * sin(time * 0.11),
        0.42 + 0.12 * cos(time * 0.23),
        0.04 + 0.04 * sin(time * 0.15)
    );
    fragColor = mix(color_cool, color_warm, blend);

    // Posición en clip space antes de la división homogénea
    vec4 clip_position = view_transform * vec4(position, 1.0f);

    // Normalmente w = 1 y la GPU simplemente hace xyz / 1 = xyz.
    // Cuando homogeneous_strength > 0, w depende de la altura en clip space:
    //   y > 0  →  w > 1  →  el vértice se ve más pequeño (parece más lejano)
    //   y < 0  →  w < 1  →  el vértice se ve más grande  (parece más cercano)
    // Este es el mismo mecanismo que usa la proyección perspectiva, que veremos más adelante.
    float w = 1.0 + homogeneous_strength * clip_position.y;
    gl_Position = vec4(clip_position.xyz, w);
}
