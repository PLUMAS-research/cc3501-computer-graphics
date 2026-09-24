#version 330

// Las pasadas de pantalla completa dibujan un rectángulo cuyas posiciones ya
// están en NDC, así que no hay cámara ni proyección que aplicar.

in vec2 position;
in vec2 uv;

out vec2 fragment_texcoord;

void main() {
    fragment_texcoord = uv;
    gl_Position = vec4(position, 0.0f, 1.0f);
}
