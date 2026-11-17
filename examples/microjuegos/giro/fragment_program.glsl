#version 330

in vec2 uv;
out vec4 outColor;

uniform sampler2D cuadro;
uniform bool fondo;      // true dibuja las franjas, false el último cuadro
uniform float progreso;
uniform float razon;

void main()
{
    if (fondo) {
        // doce franjas que giran en torno al centro, con los colores del curso
        vec2 punto = (uv - 0.5) * vec2(razon, 1.0);
        float vueltas = atan(punto.y, punto.x) / 6.2832;
        float franja = step(0.5, fract(12.0 * vueltas + 2.0 * progreso));
        outColor = vec4(mix(vec3(0.04, 0.05, 0.31), vec3(0.81, 0.22, 0.54), franja), 1.0);
    } else {
        outColor = vec4(texture(cuadro, uv).rgb, 1.0);
    }
}
