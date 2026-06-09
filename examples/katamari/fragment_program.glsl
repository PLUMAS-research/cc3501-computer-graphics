#version 330

in vec3 view_normal;
in vec3 view_position;

uniform vec3 instance_color;

out vec4 out_color;

void main() {
    vec3 normal_unitaria = normalize(view_normal);
    // luz tipo linterna ubicada en la camara (origen del espacio de vista):
    // la escena queda iluminada desde cualquier angulo de la camara que sigue
    // a la bola.
    vec3 direccion_luz = normalize(-view_position);
    float difusa = max(dot(normal_unitaria, direccion_luz), 0.0);
    float ambiente = 0.35;
    vec3 color = instance_color * (ambiente + (1.0 - ambiente) * difusa);
    out_color = vec4(color, 1.0);
}
