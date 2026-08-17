#version 330

// El arbol llega a la GPU como una sola malla ya construida: la tortuga
// aplico la pila de matrices en la CPU y horneo cada rama en coordenadas del
// arbol. Por eso aca no hay matriz de modelo, solo vista y proyeccion.
//
// El color viene por vertice y no por uniform porque un mismo draw call dibuja
// el tronco, las ramas y las hojas.

in vec3 position;
in vec3 normal;
in vec3 color;

uniform mat4 view;
uniform mat4 projection;

out vec3 world_normal;
out vec3 vertex_color;

void main()
{
    world_normal = normal;
    vertex_color = color;
    gl_Position = projection * view * vec4(position, 1.0);
}
