#version 330

// el raymarching reconstruye el rayo por pixel con gl_FragCoord, asi que el
// quad solo aporta su posicion en NDC (sin uv, que el driver descartaria).
in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0f, 1.0f);
}
