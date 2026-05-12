#version 330
in vec3 position;
uniform mat4 projection;
uniform mat4 transform;
uniform mat4 view;
uniform vec3 instance_color;

out vec3 fragColor;

void main()
{
    fragColor = instance_color;
    gl_Position = projection * view * transform * vec4(position, 1.0f);
}
