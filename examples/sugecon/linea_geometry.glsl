#version 330

// Cada segmento se dibuja como un rectángulo de ancho fijo en píxeles,
// porque el ancho de GL_LINES depende del driver.
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in float opacidad_vertice[];

uniform mat4 view;
uniform mat4 projection;
uniform vec2 tamano_ventana;
uniform float ancho_px;

// distancia a la línea central, en píxeles
out float distancia;
out float opacidad;

void main() {
    vec4 c0 = projection * view * gl_in[0].gl_Position;
    vec4 c1 = projection * view * gl_in[1].gl_Position;
    if (c0.w <= 0.0 || c1.w <= 0.0) {
        return;
    }
    vec2 s0 = c0.xy / c0.w * 0.5 * tamano_ventana;
    vec2 s1 = c1.xy / c1.w * 0.5 * tamano_ventana;
    vec2 a_lo_largo = s1 - s0;
    if (dot(a_lo_largo, a_lo_largo) < 1e-8) {
        return;
    }
    a_lo_largo = normalize(a_lo_largo);
    vec2 perpendicular = vec2(-a_lo_largo.y, a_lo_largo.x);

    // un píxel extra a cada lado para el antialiasing; los extremos se alargan
    // medio ancho para que los segmentos consecutivos se unan sin huecos
    float mitad = 0.5 * ancho_px + 1.0;
    vec2 extremo = a_lo_largo * 0.5 * ancho_px;
    vec2 desde_pantalla = 2.0 / tamano_ventana;

    vec2 esquinas[4] = vec2[4](
        -extremo + perpendicular * mitad, -extremo - perpendicular * mitad,
        extremo + perpendicular * mitad, extremo - perpendicular * mitad
    );
    for (int i = 0; i < 4; i++) {
        vec4 base = i < 2 ? c0 : c1;
        gl_Position = base + vec4(esquinas[i] * desde_pantalla * base.w, 0.0, 0.0);
        distancia = (i % 2 == 0) ? mitad : -mitad;
        opacidad = opacidad_vertice[i < 2 ? 0 : 1];
        EmitVertex();
    }
    EndPrimitive();
}
