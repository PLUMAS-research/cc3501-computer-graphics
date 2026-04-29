#version 330

in vec2 position;

uniform vec2 resolution;

void main()
{
    // Convierte coordenadas en pixeles a NDC
    vec2 ndc = (position / resolution) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
