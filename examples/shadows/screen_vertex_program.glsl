#version 330

in vec2 position;
in vec2 uv;

out vec2 fragment_texcoord;
uniform mat4 transform;
uniform mat4 projection;
uniform mat4 view;

void main() {
    fragment_texcoord = uv;
    gl_Position = projection * view * transform * vec4(position, 0.0, 1.0);
}
