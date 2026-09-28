#version 330

in vec2 fragment_texcoord;
out vec4 out_color;
uniform sampler2D sampler_tex;
uniform float light_near_plane;
uniform float light_far_plane;

void main() {
    // El shadow map guarda la profundidad que sale de la proyección, que no
    // es proporcional a la distancia: casi todo el rango se gasta cerca de
    // la luz. Para verla como distancia se deshace la proyección, primero de
    // [0, 1] a NDC y después de NDC a la profundidad en la vista de la luz.
    float depth = texture(sampler_tex, fragment_texcoord).r;
    float depth_ndc = 2.0 * depth - 1.0;
    float view_depth = 2.0 * light_near_plane * light_far_plane
        / (light_far_plane + light_near_plane
           - depth_ndc * (light_far_plane - light_near_plane));

    // de [near, far] a [0, 1]: negro junto a la luz, blanco en el plano lejano
    float visualized = (view_depth - light_near_plane)
        / (light_far_plane - light_near_plane);
    out_color = vec4(vec3(visualized), 1.0);
}
