#version 330

in vec2 position;

uniform float escala;  // 1 cubre la ventana completa
uniform float angulo;
uniform float razon;   // ancho / alto de la ventana

out vec2 uv;

void main()
{
    uv = (position + 1.0) / 2.0;
    // se gira en un sistema con la misma escala en x e y, para no deformar el cuadro
    vec2 punto = vec2(position.x * razon, position.y) * escala;
    punto = mat2(cos(angulo), sin(angulo), -sin(angulo), cos(angulo)) * punto;
    gl_Position = vec4(punto.x / razon, punto.y, 0.0, 1.0);
}
