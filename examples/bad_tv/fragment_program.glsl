#version 330
/*
este fragment program dibuja barras de color SMPTE,
la clásica "carta de ajuste" de televisión análoga.
la variable frag_uv llega interpolada desde el vertex program
y nos indica la posición original (sin deformar) del píxel.
*/

in vec2 frag_uv;

out vec3 outColor;

void main() {
    // seleccionamos una de 7 barras según la posición horizontal
    int bar = int(frag_uv.x * 7.0);

    vec3 color;

    if (bar == 0) color = vec3(1.0, 1.0, 1.0);      // blanco
    else if (bar == 1) color = vec3(1.0, 1.0, 0.0);  // amarillo
    else if (bar == 2) color = vec3(0.0, 1.0, 1.0);  // cian
    else if (bar == 3) color = vec3(0.0, 1.0, 0.0);  // verde
    else if (bar == 4) color = vec3(1.0, 0.0, 1.0);  // magenta
    else if (bar == 5) color = vec3(1.0, 0.0, 0.0);  // rojo
    else color = vec3(0.0, 0.0, 1.0);                // azul

    // efecto scanline: oscurecemos levemente líneas alternas
    // para simular un monitor CRT
    float scanline = 0.85 + 0.15 * sin(gl_FragCoord.y * 2.0);

    outColor = color * scanline;
}
