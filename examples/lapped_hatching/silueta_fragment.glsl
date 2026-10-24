#version 330

in float distancia;

uniform float ancho_px;

out vec4 color_salida;

void main() {
    // tinta negra con el borde suavizado en el último píxel
    float alfa = clamp(0.5 * ancho_px + 0.5 - abs(distancia), 0.0, 1.0);
    color_salida = vec4(0.0, 0.0, 0.0, alfa);
}
