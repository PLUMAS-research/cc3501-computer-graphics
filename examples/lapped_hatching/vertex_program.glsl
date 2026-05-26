#version 330

in vec3 position;
in vec3 normal;
in vec2 patch_uv;
in vec3 patch_color;
in float boundary_flag;
in vec3 barycentric;
in float alpha;
in float alignment_offset;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragment_normal;
out vec2 fragment_patch_uv;
out vec3 fragment_patch_color;
out float fragment_boundary_flag;
out vec3 fragment_barycentric;
out float fragment_alpha;
out float fragment_alignment_offset;

void main() {
    // El modelo se renderiza con transform = identidad, asi que la normal
    // del vertice ya esta en espacio de mundo. Si se usa otra transform
    // habria que aplicar mat3(transpose(inverse(transform))).
    fragment_normal = normal;
    fragment_patch_uv = patch_uv;
    fragment_patch_color = patch_color;
    fragment_boundary_flag = boundary_flag;
    fragment_barycentric = barycentric;
    fragment_alpha = alpha;
    fragment_alignment_offset = alignment_offset;
    gl_Position = projection * view * transform * vec4(position, 1.0);
}
