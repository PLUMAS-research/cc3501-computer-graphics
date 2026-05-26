#version 330

in vec3 world_position;
in vec3 world_normal;
in vec2 frag_uv;

uniform sampler2D diffuse_texture;
uniform vec3 base_color_factor;
uniform vec3 light_direction;
uniform float ambient_strength;
uniform float alpha_cutoff;

out vec4 out_color;

void main()
{
    vec4 sampled = texture(diffuse_texture, frag_uv);
    // descarte de pixeles practicamente invisibles para no escribir al
    // depth buffer puntos que no aportan. El cutoff es bajo (~0.01)
    // porque el resto de la transparencia se resuelve con blending
    if (sampled.a < alpha_cutoff) {
        discard;
    }
    vec3 albedo = sampled.rgb * base_color_factor;
    vec3 normal = normalize(world_normal);
    vec3 light = normalize(light_direction);
    float diffuse_intensity = max(dot(normal, light), 0.0);
    float shading = ambient_strength + (1.0 - ambient_strength) * diffuse_intensity;
    out_color = vec4(albedo * shading, sampled.a);
}
