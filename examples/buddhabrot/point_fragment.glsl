#version 330

out vec4 out_color;
uniform vec4 channel_color;

void main() {
    out_color = channel_color;
}
