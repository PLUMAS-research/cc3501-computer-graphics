#version 330

in vec3 normal_mundo;
in vec2 uv_textura;
out vec4 outColor;

uniform sampler2D textura;
uniform vec3 direccion_luz;

void main()
{
    vec4 color = texture(textura, uv_textura);
    // las partes transparentes de las plumas no se dibujan
    if (color.a < 0.05) {
        discard;
    }
    // las alas son planos que se ven por ambos lados, así que la iluminación
    // usa el valor absoluto del coseno
    float difuso = abs(dot(normalize(normal_mundo), direccion_luz));
    outColor = vec4(color.rgb * (0.45 + 0.55 * difuso), color.a);
}
