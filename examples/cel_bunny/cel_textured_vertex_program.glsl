#version 330

// Igual al de cel_vertex_program, con las coordenadas de textura agregadas:
// en un modelo con mapa difuso el color base no es un uniform, sino lo que
// diga la textura en cada punto.

in vec3 position;
in vec3 normal;
in vec2 uv;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 world_position;
out vec3 world_normal;
out vec2 fragment_texcoord;

void main()
{
    vec4 position_world = transform * vec4(position, 1.0);
    world_position = position_world.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(transform)));
    world_normal = normalize(normal_matrix * normal);

    fragment_texcoord = uv;

    gl_Position = projection * view * position_world;
}
