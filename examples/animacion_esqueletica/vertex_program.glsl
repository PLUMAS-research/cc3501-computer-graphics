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
    vec4 world = transform * vec4(position, 1.0);
    world_position = world.xyz;

    // transpuesta de la inversa para que las normales sigan siendo
    // perpendiculares a la superficie aunque la jerarquia incluya escalas
    // no uniformes (algunos sub-objetos del mecha si las tienen)
    mat3 normal_matrix = transpose(inverse(mat3(transform)));
    world_normal = normalize(normal_matrix * normal);

    frag_uv = uv;
    gl_Position = projection * view * world;
}
