#version 330

// cada primitiva es una arista con sus dos vértices opuestos: (a, p0, p1, b).
// La cara A = (p0, p1, a) y la cara B = (p1, p0, b) comparten la arista p0-p1.
// En una arista de borde no hay cara B y b se repite igual a a: el borde de
// una malla abierta se dibuja siempre, como el contorno de una hoja.
layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 posicion_camara;
uniform vec2 tamano_ventana;
uniform float ancho_px;

// distancia a la línea central, en píxeles
out float distancia;

vec4 a_clip(vec3 punto) {
    // la línea se adelanta un poco hacia la cámara para no quedar tapada por
    // las caras que la rodean, que están a casi la misma profundidad
    vec3 hacia_camara = normalize(posicion_camara - punto);
    return projection * view * vec4(punto + 0.004 * hacia_camara, 1.0);
}

void main() {
    vec3 a = gl_in[0].gl_Position.xyz;
    vec3 p0 = gl_in[1].gl_Position.xyz;
    vec3 p1 = gl_in[2].gl_Position.xyz;
    vec3 b = gl_in[3].gl_Position.xyz;

    vec3 hacia_camara = posicion_camara - p0;
    bool frente_a = dot(cross(p1 - p0, a - p0), hacia_camara) > 0.0;
    bool frente_b = dot(cross(p0 - p1, b - p1), hacia_camara) > 0.0;
    bool es_borde = a == b;
    // silueta: una cara mira a la cámara y la otra no
    if (!es_borde && frente_a == frente_b) {
        return;
    }

    vec4 c0 = a_clip(p0);
    vec4 c1 = a_clip(p1);
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
        EmitVertex();
    }
    EndPrimitive();
}
