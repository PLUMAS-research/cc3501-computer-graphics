#version 330

in float distancia;
in float opacidad;

uniform float ancho_px;
uniform vec3 color;

out vec4 color_salida;

void main() {
    // el borde se suaviza en el último píxel
    float cobertura = clamp(0.5 * ancho_px + 0.5 - abs(distancia), 0.0, 1.0);
    color_salida = vec4(color, opacidad * cobertura);
}
