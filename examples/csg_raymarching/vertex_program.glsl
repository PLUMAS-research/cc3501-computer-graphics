#version 330

// el raymarching usa gl_FragCoord para reconstruir el rayo por pixel, asi que
// el quad solo necesita aportar su posicion en NDC.
in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0f, 1.0f);
}
