#version 330

in vec2 frag_texcoords_correct;
noperspective in vec2 frag_texcoords_affine;

uniform sampler2D texture_sampler;
uniform bool perspective_correct;
uniform bool show_seam;

out vec4 out_color;

void main() {
    vec2 uv = perspective_correct ? frag_texcoords_correct : frag_texcoords_affine;
    vec3 sampled = texture(texture_sampler, uv).rgb;

    if (show_seam) {
        // resalta la diagonal donde se juntan los dos triángulos
        float seam_distance = abs(frag_texcoords_correct.x - frag_texcoords_correct.y);
        float seam_mask = 1.0 - smoothstep(0.0, 0.01, seam_distance);
        sampled = mix(sampled, vec3(1.0, 0.2, 0.2), seam_mask * 0.85);
    }

    out_color = vec4(sampled, 1.0);
}
