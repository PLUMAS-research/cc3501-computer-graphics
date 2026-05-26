#version 330

in vec3 world_position;
in vec3 world_normal;
in vec2 frag_uv;

uniform sampler2D diffuse_texture;
uniform vec3 base_color_factor;
uniform vec3 light_direction;
uniform float ambient_strength;

out vec4 out_color;

void main()
{
    // siempre muestreamos la textura: primitivas sin baseColorTexture
    // reciben una textura blanca de 1x1, asi el shader es uniforme y
    // base_color_factor controla el tinte final
    vec4 sampled = texture(diffuse_texture, frag_uv);
    vec3 albedo = sampled.rgb * base_color_factor;

    vec3 normal = normalize(world_normal);
    vec3 light = normalize(light_direction);
    float diffuse_intensity = max(dot(normal, light), 0.0);

    float shading = ambient_strength + (1.0 - ambient_strength) * diffuse_intensity;
    out_color = vec4(albedo * shading, 1.0);
}
