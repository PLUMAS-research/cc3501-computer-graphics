#version 330

in vec2 frag_texcoord;
out vec4 out_color;
uniform sampler2D sampler_tex;

void main() {
    // La paleta ya viene aplicada desde la CPU,
    // así que aquí solo suavizamos con un pequeño boost de contraste
    vec3 color = texture(sampler_tex, frag_texcoord).rgb;
    out_color = vec4(color, 1.0);
}
