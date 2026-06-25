#version 330

// Lineas de overlay de un panel MPR (rectangulo de recorte y cruz de cortes).
// Usa el mismo quad_scale que el corte para que las lineas calcen con la imagen.
in vec2 position;

uniform vec2 quad_scale;

void main() {
    gl_Position = vec4(position * quad_scale, 0.0f, 1.0f);
}
