#version 330

in vec3 position;
in vec2 patch_uv;
in vec3 patch_color;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec2 fragment_patch_uv;
out vec3 fragment_patch_color;

void main() {
    fragment_patch_uv = patch_uv;
    fragment_patch_color = patch_color;
    gl_Position = projection * view * transform * vec4(position, 1.0);
}
