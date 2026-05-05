#version 330
in vec3 position;
in vec3 normal;
in vec2 uv;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 world_position;
out vec3 world_normal;
out vec2 frag_uv;

void main()
{
    vec4 position_world = transform * vec4(position, 1.0);
    world_position = position_world.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(transform)));
    world_normal = normalize(normal_matrix * normal);

    frag_uv = uv;
    gl_Position = projection * view * position_world;
}
