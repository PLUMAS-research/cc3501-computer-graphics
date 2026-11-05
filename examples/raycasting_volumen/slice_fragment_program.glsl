#version 330

// Corte 2D (MPR) de la textura 3D en un plano ortogonal. Cada panel fija uno de
// los tres ejes en una posicion (slice_pos) y muestra el plano perpendicular en
// escala de grises con ventana radiologica (window/level). La region que queda
// fuera de la caja de recorte se atenua, para que se vea en el corte que esa
// zona no entra a la vista 3D.
//
//   axis 0  fija z (anterior-posterior) -> plano (x = uv.x, y = uv.y).
//   axis 1  fija y (superior-inferior)  -> plano (x = uv.x, z = uv.y).
//   axis 2  fija x (izquierda-derecha)  -> plano (z = uv.x, y = uv.y).
//
// En axis 2 se cruza uv para que el eje superior-inferior (y) quede vertical, y
// el corte sagital salga derecho en vez de acostado.

in vec2 frag_uv;
out vec4 out_color;

uniform sampler3D volume;
uniform int axis;
uniform float slice_pos;       // posicion del corte en el eje fijo, en [0,1].
uniform vec2 window_level;     // (nivel, ancho) de la ventana en HU.
uniform vec3 clip_min;
uniform vec3 clip_max;
uniform sampler3D mascara;     // segmentacion: la etiqueta de cada voxel (0 = fondo), en R8.
uniform int usar_mascara;      // 1 si se cargo una segmentacion.

// La misma paleta de fragment_program.glsl, un color por etiqueta.
const vec3 PALETA[8] = vec3[8](
    vec3(1.00, 0.80, 0.10), vec3(0.20, 0.85, 1.00), vec3(0.55, 1.00, 0.35),
    vec3(1.00, 0.45, 0.85), vec3(1.00, 0.55, 0.15), vec3(0.65, 0.55, 1.00),
    vec3(0.95, 1.00, 0.55), vec3(0.35, 1.00, 0.80));

void main() {
    vec3 coordenada;
    if (axis == 0)      coordenada = vec3(frag_uv.x, frag_uv.y, slice_pos);
    else if (axis == 1) coordenada = vec3(frag_uv.x, slice_pos, frag_uv.y);
    else                coordenada = vec3(slice_pos, frag_uv.y, frag_uv.x);

    float hu = texture(volume, coordenada).r;

    // ventana radiologica: mapea [nivel - ancho/2, nivel + ancho/2] a [0,1].
    float lo = window_level.x - 0.5 * window_level.y;
    float hi = window_level.x + 0.5 * window_level.y;
    float gris = clamp((hu - lo) / (hi - lo), 0.0, 1.0);

    bool dentro = all(greaterThanEqual(coordenada, clip_min))
               && all(lessThanEqual(coordenada, clip_max));

    vec3 color = vec3(gris);
    float etiqueta = usar_mascara == 1 ? round(texture(mascara, coordenada).r * 255.0) : 0.0;
    if (etiqueta > 0.0) {
        color = mix(color, PALETA[int(mod(etiqueta - 1.0, 8.0))], 0.6);
    }
    if (!dentro) {
        color *= 0.35;  // atenua lo que el recorte deja fuera.
    }
    out_color = vec4(color, 1.0);
}
