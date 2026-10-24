#version 330

uniform vec3 color;

out vec4 color_salida;

void main() {
    color_salida = vec4(color, 1.0);
}
