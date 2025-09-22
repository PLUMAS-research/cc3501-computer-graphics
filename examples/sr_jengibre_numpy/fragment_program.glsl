#version 330

in vec2 frag_texcoord;
out vec4 out_color;
uniform sampler2D sampler_tex;

void main() {
    out_color = 1.5 * texture(sampler_tex, frag_texcoord);
}