#version 330

in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 view_normal;
out vec3 view_position;

void main() {
    mat4 modelview = view * transform;
    vec4 posicion_vista = modelview * vec4(position, 1.0);
    view_position = posicion_vista.xyz;
    view_normal = mat3(modelview) * normal;
    gl_Position = projection * posicion_vista;
}
