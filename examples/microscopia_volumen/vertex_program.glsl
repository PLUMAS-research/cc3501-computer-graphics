#version 330

// El ray casting reconstruye el rayo por pixel con gl_FragCoord, asi que el quad
// que cubre la pantalla solo necesita aportar su posicion en NDC.
in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0f, 1.0f);
}
