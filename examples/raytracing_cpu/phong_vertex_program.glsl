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
    vec4 posicion_mundo = transform * vec4(position, 1.0f);
    world_position = posicion_mundo.xyz;
    // normal a espacio de mundo con la inversa traspuesta (admite escalas no uniformes)
    world_normal = mat3(transpose(inverse(transform))) * normal;
    gl_Position = projection * view * posicion_mundo;
}
