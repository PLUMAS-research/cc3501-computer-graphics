#version 330

// posicion en pixeles de la ventana, con el origen abajo a la izquierda.
in vec2 position;
in vec3 color;

uniform vec2 resolution;

out vec3 fragColor;

void main() {
    fragColor = color;
    gl_Position = vec4(2.0f * position / resolution - 1.0f, 0.0f, 1.0f);
}
