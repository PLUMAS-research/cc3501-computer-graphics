#version 330
in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 posicion_mundo;
out vec3 normal_mundo;

void main()
{
    vec4 posicion = transform * vec4(position, 1.0);
    posicion_mundo = posicion.xyz;

    // transpuesta de la inversa: las piezas del carrusel usan escalados no
    // uniformes (una pata es un cilindro largo y delgado) y la normal dejaria
    // de ser perpendicular si se transformara con la misma matriz
    mat3 matriz_normal = transpose(inverse(mat3(transform)));
    normal_mundo = normalize(matriz_normal * normal);

    gl_Position = projection * view * posicion;
}
