#version 330

in vec2 frag_texcoord;
out vec4 out_color;
uniform sampler2D accum_tex;
uniform float exposure;

void main() {
    float accum = texture(accum_tex, frag_texcoord).r;

    // Tone mapping: raíz cuadrada + exposición
    float t = clamp(sqrt(accum * exposure), 0.0, 1.0);

    // Paleta "hot metal": negro -> rojo -> naranja -> amarillo -> blanco
    float r = clamp(3.0 * t, 0.0, 1.0);
    float g = clamp(3.0 * t - 1.0, 0.0, 1.0);
    float b = clamp(3.0 * t - 2.0, 0.0, 1.0);

    out_color = vec4(r, g, b, 1.0);
}
