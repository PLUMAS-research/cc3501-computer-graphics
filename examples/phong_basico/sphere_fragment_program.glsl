#version 330

in vec3 world_position;
in vec3 world_normal;

uniform vec3 view_position;

uniform vec3 light_position;
uniform vec3 light_color;
uniform vec3 ambient_light;

// k_a, k_d, k_s. Para "apagar" un termino desde el CPU
// basta con poner el coeficiente respectivo en cero.
uniform vec3 material_ambient;
uniform vec3 material_diffuse;
uniform vec3 material_specular;
uniform float material_shininess;

out vec3 out_color;

void main()
{
    vec3 normal = normalize(world_normal);
    vec3 view_direction = normalize(view_position - world_position);
    vec3 light_direction = normalize(light_position - world_position);

    // L_ambient = k_a * I_a
    vec3 ambient = material_ambient * ambient_light;

    // L_diffuse = k_d * I_light * max(0, N . L)
    float diffuse_intensity = max(dot(normal, light_direction), 0.0);
    vec3 diffuse = material_diffuse * light_color * diffuse_intensity;

    // L_specular = k_s * I_light * max(0, R . V)^alpha
    vec3 reflect_direction = reflect(-light_direction, normal);
    float specular_intensity = pow(
        max(dot(view_direction, reflect_direction), 0.0),
        material_shininess
    );
    vec3 specular = material_specular * light_color * specular_intensity;

    out_color = min(ambient + diffuse + specular, vec3(1.0));
}
