#version 330

in vec3 position;

uniform mat4 view_transform;
uniform vec3 axis_color;

out vec3 fragColor;

void main()
{
    fragColor = axis_color;
    // Los ejes siempre tienen w = 1: no se ven afectados por homogeneous_strength.
    gl_Position = view_transform * vec4(position, 1.0);
}
