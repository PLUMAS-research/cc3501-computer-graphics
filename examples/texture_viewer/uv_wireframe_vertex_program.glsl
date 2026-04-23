#version 330

in vec2 uv;

void main() {
    // cada vértice usa directamente su coordenada UV como posición 2D,
    // reescalando de [0,1] a clip space [-1,1]
    gl_Position = vec4(2.0 * uv - 1.0, 0.0, 1.0);
}
