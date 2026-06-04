#version 330

in vec3 view_normal;
in vec3 view_position;

uniform vec3 color_base;

out vec4 out_color;

void main() {
    vec3 normal_unitaria = normalize(view_normal);
    // luz tipo linterna: ubicada en la camara (origen del espacio de vista),
    // asi el modelo queda iluminado desde cualquier angulo de rotacion.
    vec3 direccion_luz = normalize(-view_position);
    // las caras de los hoyos del modelo miran al otro lado; abs evita que se
    // vean completamente negras y mantiene visible la silueta poligonal.
    float difusa = abs(dot(normal_unitaria, direccion_luz));
    float ambiente = 0.25;
    vec3 color = color_base * (ambiente + (1.0 - ambiente) * difusa);
    out_color = vec4(color, 1.0);
}
