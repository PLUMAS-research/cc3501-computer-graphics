#version 330
in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 normal_mundo;

void main()
{
    mat3 matriz_normal = transpose(inverse(mat3(transform)));
    normal_mundo = normalize(matriz_normal * normal);

    gl_Position = projection * view * transform * vec4(position, 1.0);
}
