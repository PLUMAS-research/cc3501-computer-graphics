#version 330

in vec3 world_position;
in vec3 world_normal;
in vec2 frag_uv;

uniform sampler2D diffuse_texture;

uniform vec3 light_position;
uniform vec3 light_color;
uniform vec3 ambient_light;

// ambient_factor permite que Tierra y Luna compartan la misma pipeline
// pero usen luz ambiente distinta. La Tierra recibe el ambient completo
// (factor 1.0) para que su lado nocturno sea legible, la Luna recibe
// muy poco (factor ~0.1) para que su lado oscuro se vea casi negro
// como en la realidad
uniform float ambient_factor;

out vec3 out_color;

void main()
{
    vec3 normal = normalize(world_normal);
    vec3 light_direction = normalize(light_position - world_position);

    vec3 albedo = texture(diffuse_texture, frag_uv).rgb;

    vec3 ambient = ambient_light * albedo * ambient_factor;

    float diffuse_intensity = max(dot(normal, light_direction), 0.0);
    vec3 diffuse = albedo * light_color * diffuse_intensity;

    out_color = min(ambient + diffuse, vec3(1.0));
}
