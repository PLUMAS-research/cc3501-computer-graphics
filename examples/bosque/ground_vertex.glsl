#version 330

in vec3 position;
in vec3 color;

uniform mat4 view;
uniform mat4 projection;

out vec3 frag_color;

void main()
{
    gl_Position = projection * view * vec4(position, 1.0);
    frag_color = color;
}
