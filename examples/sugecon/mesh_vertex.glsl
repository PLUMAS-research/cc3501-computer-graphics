#version 330

in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragNormal;

void main()
{
    fragNormal = normalize(mat3(view * transform) * normal);
    gl_Position = projection * view * transform * vec4(position, 1.0);
}