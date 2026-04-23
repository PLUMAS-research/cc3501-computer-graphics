#version 330

in vec3 position;
in vec2 texcoords;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec2 frag_texcoords_correct;
noperspective out vec2 frag_texcoords_affine;

void main() {
    gl_Position = projection * view * transform * vec4(position, 1.0);
    frag_texcoords_correct = texcoords;
    frag_texcoords_affine = texcoords;
}
