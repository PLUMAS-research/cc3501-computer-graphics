#version 330

in vec2 ndc;

// base de la camara: reconstruimos la direccion del rayo por pixel con la
// misma formula que usa el trazador (Camara.rayo), asi el fondo rasterizado
// queda identico al fondo del ray tracing (mismo gradiente segun la direccion)
uniform vec3 forward;
uniform vec3 right;
uniform vec3 up_vector;
uniform float half_width;
uniform float half_height;

uniform vec3 color_cielo_arriba;
uniform vec3 color_cielo_abajo;

out vec4 outColor;

void main()
{
    vec3 direccion = normalize(
        forward + ndc.x * half_width * right + ndc.y * half_height * up_vector
    );
    float mezcla = 0.5f * (direccion.y + 1.0f);
    outColor = vec4(mix(color_cielo_abajo, color_cielo_arriba, mezcla), 1.0f);
}
