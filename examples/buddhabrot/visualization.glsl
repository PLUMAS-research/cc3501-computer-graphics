#version 330

in vec2 frag_texcoord;
out vec4 out_color;
uniform sampler2D accum_tex;
uniform float exposure;

void main() {
    vec3 accum = texture(accum_tex, frag_texcoord).rgb;
    // Raíz cuadrada para comprimir el rango dinámico,
    // controlado por la exposición
    vec3 color = sqrt(accum * exposure);
    out_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
