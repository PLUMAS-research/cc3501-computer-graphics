#version 330

in vec3 view_normal;
in vec3 view_position;

uniform vec3 color_base;
uniform float opacidad;  // 1.0 para lineas y piso; < 1.0 para la gelatina.

out vec4 out_color;

void main() {
    vec3 normal_unitaria = normalize(view_normal);
    // luz tipo linterna en la camara: el cubo queda iluminado desde cualquier
    // angulo de rotacion del arcball.
    vec3 direccion_luz = normalize(-view_position);
    float difusa = abs(dot(normal_unitaria, direccion_luz));
    float ambiente = 0.3;
    vec3 color = color_base * (ambiente + (1.0 - ambiente) * difusa);
    out_color = vec4(color, opacidad);
}
