#version 330

in vec2 position;
in vec2 uv;

out vec2 frag_texcoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    frag_texcoord = uv;
}
