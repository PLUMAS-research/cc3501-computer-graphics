#version 330
in vec3 position;
in vec3 normal;

uniform vec3 color;
uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main()
{
    // Iluminación Lambert simple con luz direccional fija.
    // Con esto las instancias se distinguen unas de otras aunque compartan color.
    vec3 light_direction = normalize(vec3(0.4, 0.9, 0.5));
    vec3 world_normal = normalize(mat3(transform) * normal);
    float diffuse_term = max(dot(world_normal, light_direction), 0.0);
    float ambient_term = 0.35;
    fragColor = color * (ambient_term + diffuse_term * 0.7);

    gl_Position = projection * view * transform * vec4(position, 1.0);
}
