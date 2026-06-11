#version 330
in vec2 position;

out vec2 ndc;

void main()
{
    // el quad cubre toda la pantalla en coordenadas normalizadas [-1, 1]
    ndc = position;
    gl_Position = vec4(position, 0.0f, 1.0f);
}
