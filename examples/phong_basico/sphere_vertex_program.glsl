#version 330
in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 world_position;
out vec3 world_normal;

void main()
{
    vec4 position_world = transform * vec4(position, 1.0);
    world_position = position_world.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(transform)));
    world_normal = normalize(normal_matrix * normal);

    gl_Position = projection * view * position_world;
}
