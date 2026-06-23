#version 330

// Dibuja huesos (lineas) y articulaciones (puntos) del esqueleto. Las
// posiciones se reescriben por cuadro desde las poses globales de cada hueso.

in vec3 position;
in vec3 color;

uniform mat4 view;
uniform mat4 projection;

out vec3 vertex_color;

void main()
{
    vertex_color = color;
    gl_Position = projection * view * vec4(position, 1.0);
}
