#version 330

in vec3 position;
in float alpha;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 color;

out vec4 vertColor;

void main()
{
    vertColor = vec4(color, alpha);
    gl_Position = projection * view * transform * vec4(position, 1.0);
}