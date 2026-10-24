#version 330

in vec2 uv_tam;
in vec2 uv_mascara;
in float alfa_margen;
in vec2 uv_textura;
flat in float capa_textura;
in vec3 pesos_claros;
in vec3 pesos_oscuros;
flat in vec3 pesos_claros_cara;
flat in vec3 pesos_oscuros_cara;

// los seis tonos del TAM en los canales RGB de dos texturas, con sus mipmaps
uniform sampler2D tam_claros;
uniform sampler2D tam_oscuros;
// alfa del contorno del parche (la mancha)
uniform sampler2D contorno;
// texturas y colores del modelo, una capa por material
uniform sampler2DArray textura_modelo;
uniform int usar_textura;
uniform int esquema_mezcla;
// umbral de tono: aproxima un escalón para que los trazos nuevos aparezcan
// como líneas negras que se alargan en vez de líneas grises que se oscurecen
uniform int umbral;

out vec4 color_salida;

void main() {
    vec3 claros = esquema_mezcla >= 2 ? pesos_claros_cara : pesos_claros;
    vec3 oscuros = esquema_mezcla >= 2 ? pesos_oscuros_cara : pesos_oscuros;

    // los pesos interpolados entre los tres vértices mezclan hasta seis
    // imágenes del TAM; el mipmap elige el nivel según el tamaño en pantalla
    float tinta = dot(claros, texture(tam_claros, uv_tam).rgb)
                + dot(oscuros, texture(tam_oscuros, uv_tam).rgb);
    float papel = 1.0 - tinta;
    if (umbral == 1) {
        papel = clamp(8.0 * papel - 3.5, 0.0, 1.0);
    }
    // usar_textura 0: tinta negra sobre papel blanco. 1: la tinta negra
    // oscurece el color del modelo, como el achurado a pluma sobre una aguada.
    // 2: los trazos tienen el color del modelo y el papel queda blanco, como
    // el achurado con lápices de colores
    vec3 color_modelo = texture(textura_modelo, vec3(uv_textura, capa_textura)).rgb;
    vec3 color_papel = usar_textura == 1 ? color_modelo : vec3(1.0);
    vec3 color_tinta = usar_textura == 2 ? color_modelo : vec3(0.0);
    color_salida = vec4(mix(color_tinta, color_papel, papel), texture(contorno, uv_mascara).r * alfa_margen);
}
