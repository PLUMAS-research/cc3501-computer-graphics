#version 330

in vec3 position;

uniform mat4 transform;
uniform vec3 color;

out vec3 fragment_color;

void main()
{
    fragment_color = color;
    gl_Position = transform * vec4(position, 1.0);
}
