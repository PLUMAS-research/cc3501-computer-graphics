#version 330
in vec3 position;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

uniform float near_plane;
uniform float far_plane;

out float frag_depth;

void main()
{
    vec4 view_pos = view * transform * vec4(position, 1.0f);
    vec4 clip_pos = projection * view_pos;
    gl_Position = clip_pos;

    frag_depth = (-view_pos.z - near_plane) / (far_plane - near_plane);
}