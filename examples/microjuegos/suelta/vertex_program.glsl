#version 330

in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 normal_mundo;

void main()
{
    // las cajas se escalan sin uniformidad, así que la normal usa la inversa transpuesta
    normal_mundo = mat3(transpose(inverse(transform))) * normal;
    gl_Position = projection * view * transform * vec4(position, 1.0);
}
