#version 330

in vec3 world_position;
in vec3 world_normal;

uniform vec3 view_position;

uniform vec3 light_1_position;
uniform vec3 light_2_position;
uniform vec3 light_1_color;
uniform vec3 light_2_color;
uniform vec3 ambient_light;

uniform vec3 material_ambient;
uniform vec3 material_diffuse;
uniform vec3 material_specular;
uniform float material_shininess;

out vec3 out_color;

vec3 phong_light(vec3 normal, vec3 view_direction,
                 vec3 light_position, vec3 light_color)
{
    vec3 light_direction = normalize(light_position - world_position);

    // componente difusa: k_d * L * max(0, N . L)
    float diffuse_intensity = max(dot(normal, light_direction), 0.0);
    vec3 diffuse = material_diffuse * light_color * diffuse_intensity;

    // componente especular: k_s * L * max(0, R . V)^alpha
    vec3 reflect_direction = reflect(-light_direction, normal);
    float specular_intensity = pow(
        max(dot(view_direction, reflect_direction), 0.0),
        material_shininess
    );
    vec3 specular = material_specular * light_color * specular_intensity;

    return diffuse + specular;
}

void main()
{
    vec3 normal = normalize(world_normal);
    vec3 view_direction = normalize(view_position - world_position);

    vec3 ambient = material_ambient * ambient_light;
    vec3 light_1 = phong_light(normal, view_direction, light_1_position, light_1_color);
    vec3 light_2 = phong_light(normal, view_direction, light_2_position, light_2_color);

    out_color = min(ambient + light_1 + light_2, vec3(1.0));
}
