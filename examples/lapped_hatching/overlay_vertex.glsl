#version 330

in vec2 ndc_position;
in vec2 patch_uv;

out vec2 fragment_patch_uv;

void main() {
    fragment_patch_uv = patch_uv;
    gl_Position = vec4(ndc_position, 0.0, 1.0);
}
