#version 330

uniform sampler2D tam_claros;
uniform sampler2D tam_oscuros;
// esquina inferior izquierda del panel, en píxeles de la ventana
uniform vec2 origen;

out vec4 color_salida;

const float SEPARACION = 8.0;
const float ANCHO_COLUMNA = 128.0;

// El panel muestra el TAM como la figura 2 del paper: una columna por tono,
// de claro a oscuro, y una fila por nivel de mipmap, con el de 32x32 arriba y
// el de 128x128 abajo. Cada texel ocupa un píxel.
void main() {
    vec2 punto = gl_FragCoord.xy - origen;
    int columna = int(floor((punto.x - SEPARACION) / (ANCHO_COLUMNA + SEPARACION)));
    float x_columna = SEPARACION + float(columna) * (ANCHO_COLUMNA + SEPARACION);

    float papel = 1.0;
    float y_fila = SEPARACION;
    for (int nivel = 1; nivel <= 3; nivel++) {
        float lado = float(256 >> nivel);
        vec2 dentro = punto - vec2(x_columna + 0.5 * (ANCHO_COLUMNA - lado), y_fila);
        bool en_marco = all(greaterThanEqual(dentro, vec2(-1.0))) && all(lessThan(dentro, vec2(lado + 1.0)));
        bool en_imagen = all(greaterThanEqual(dentro, vec2(0.0))) && all(lessThan(dentro, vec2(lado)));
        if (en_imagen && columna >= 0 && columna < 6) {
            // la fila 0 de la imagen va arriba, como al dibujarla en papel
            ivec2 texel = ivec2(int(dentro.x), int(lado) - 1 - int(dentro.y));
            vec3 claros = texelFetch(tam_claros, texel, nivel).rgb;
            vec3 oscuros = texelFetch(tam_oscuros, texel, nivel).rgb;
            papel = 1.0 - (columna < 3 ? claros[columna] : oscuros[columna - 3]);
        } else if (en_marco && columna >= 0 && columna < 6) {
            papel = 0.4;
        }
        y_fila += lado + SEPARACION;
    }
    color_salida = vec4(vec3(papel), 1.0);
}
