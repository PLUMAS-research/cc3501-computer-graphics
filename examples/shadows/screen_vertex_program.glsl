#version 330

// El quad ya está en NDC (rectangle_2d entrega posiciones en [-1, 1]^2),
// así que no aplicamos view/projection ni una matriz de modelo: el vertex
// shader solo pasa la posición tal cual al pipeline.
in vec2 position;
in vec2 uv;

out vec2 fragment_texcoord;

void main() {
    fragment_texcoord = uv;
    gl_Position = vec4(position, 0.0, 1.0);
}
