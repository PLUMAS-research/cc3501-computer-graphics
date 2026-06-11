#version 330

in vec3 position;
in vec3 normal;
in float value;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragNormal;
out float fragValue;

void main()
{
    fragNormal = normalize(mat3(view * transform) * normal);
    fragValue = value;
    gl_Position = projection * view * transform * vec4(position, 1.0);
}
