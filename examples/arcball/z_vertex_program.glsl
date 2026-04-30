#version 330
in vec3 position;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out float view_z;

void main()
{
    vec4 view_pos = view * transform * vec4(position, 1.0f);
    gl_Position = projection * view_pos;

    // Profundidad lineal en espacio cámara (positiva hacia adelante).
    view_z = -view_pos.z;
}
