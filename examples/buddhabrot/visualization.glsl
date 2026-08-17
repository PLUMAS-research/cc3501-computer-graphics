#version 330

in vec2 frag_texcoord;
out vec4 out_color;
uniform sampler2D accum_tex;
uniform float exposure;

// Los tres canales acumulan a escalas muy distintas: el de pocas iteraciones
// recibe muchas más visitas que el de muchas. Sin esta ganancia por canal la
// imagen sale de un solo color.
const vec3 GANANCIA = vec3(1.0, 0.35, 0.22);

void main() {
    vec3 accumulated_value = texture(accum_tex, frag_texcoord).rgb;
    // Raíz cuadrada para comprimir el rango dinámico,
    // controlado por la exposición
    vec3 tone_mapped_color = sqrt(accumulated_value * exposure * GANANCIA);
    out_color = vec4(clamp(tone_mapped_color, 0.0, 1.0), 1.0);
}
