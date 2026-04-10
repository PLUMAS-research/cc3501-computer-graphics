#version 330

in vec2 frag_texcoord;
out vec4 out_color;
uniform sampler2D accum_tex;
uniform float exposure;

void main() {
    vec3 accumulated_value = texture(accum_tex, frag_texcoord).rgb;
    // Raíz cuadrada para comprimir el rango dinámico,
    // controlado por la exposición
    vec3 tone_mapped_color = sqrt(accumulated_value * exposure);
    out_color = vec4(clamp(tone_mapped_color, 0.0, 1.0), 1.0);
}
