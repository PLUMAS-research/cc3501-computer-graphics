#version 330

// Quad de un panel MPR. El quad se escala con quad_scale para respetar las
// proporciones fisicas del corte dentro del viewport (un corte anisotropico no
// debe salir estirado). El uv viaja al fragment program para ubicar el corte.
in vec2 position;
in vec2 uv;

out vec2 frag_uv;

uniform vec2 quad_scale;

void main() {
    gl_Position = vec4(position * quad_scale, 0.0f, 1.0f);
    frag_uv = uv;
}
