#version 330

in vec2 frag_texcoord;
out vec4 out_color;

uniform sampler2D sampler_tex;
uniform float background_dim;

void main() {
    vec4 texel = texture(sampler_tex, frag_texcoord);
    // oscurecemos el fondo para que las aristas UV resalten encima
    out_color = vec4(texel.rgb * background_dim, 1.0);
}
