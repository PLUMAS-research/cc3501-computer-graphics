#version 330
/*
este vertex program deforma la geometría para simular
la distorsión horizontal de una señal de TV análoga.
a diferencia del VP del ejemplo color_wheel (que es una identidad),
aquí el VP realiza cálculos que modifican la posición de los vértices.
*/

in vec2 position;

uniform float time;

// esta variable se interpola entre vértices y llega al fragment program
out vec2 frag_uv;

void main() {
    // coordenadas UV a partir de la posición original (mapeadas de [-1,1] a [0,1])
    // se calculan ANTES de deformar, para que el FP conozca
    // la posición "real" de cada vértice
    frag_uv = (position + 1.0) / 2.0;

    // distorsión horizontal: dos ondas sinusoidales
    // simulan la pérdida de sincronización horizontal
    float wave = sin(position.y * 10.0 + time * 4.0) * 0.06;
    wave += sin(position.y * 3.5 - time * 1.5) * 0.04;

    // la intensidad de la distorsión varía con el tiempo:
    // a veces la TV "casi funciona", a veces se distorsiona mucho
    wave *= (0.5 + 0.5 * sin(time * 0.8));

    // aplicamos la deformación solo en el eje X
    gl_Position = vec4(position.x + wave, position.y, 0.0, 1.0);
}
