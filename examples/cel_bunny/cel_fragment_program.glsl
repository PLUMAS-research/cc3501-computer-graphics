#version 330

in vec3 world_position;
in vec3 world_normal;

uniform vec3 view_position;

uniform vec3 light_1_position;
uniform vec3 light_2_position;
uniform vec3 light_1_color;
uniform vec3 light_2_color;
uniform vec3 ambient_light;

uniform vec3 material_diffuse;
uniform vec3 material_specular;
uniform float material_shininess;

uniform int num_bands;
uniform float specular_threshold;
uniform int outline_enabled;
uniform float outline_threshold;

out vec3 out_color;

vec3 cel_light(vec3 normal, vec3 view_direction,
               vec3 light_position, vec3 light_color)
{
    vec3 light_direction = normalize(light_position - world_position);

    // componente difusa cuantizada en num_bands escalones
    float diffuse_raw = max(dot(normal, light_direction), 0.0);
    float bands = float(num_bands);
    float diffuse_quantized = ceil(diffuse_raw * bands) / bands;
    vec3 diffuse = material_diffuse * light_color * diffuse_quantized;

    // componente especular umbralizada (highlight binario)
    vec3 reflect_direction = reflect(-light_direction, normal);
    float specular_raw = pow(
        max(dot(view_direction, reflect_direction), 0.0),
        material_shininess
    );
    float specular_intensity = step(specular_threshold, specular_raw);
    vec3 specular = material_specular * light_color * specular_intensity;

    return diffuse + specular;
}

void main()
{
    vec3 normal = normalize(world_normal);
    vec3 view_direction = normalize(view_position - world_position);

    // outline por silueta: si la normal es casi perpendicular a la
    // dirección de la vista, el píxel está en el borde del objeto
    float silhouette_factor = max(dot(normal, view_direction), 0.0);
    if (outline_enabled == 1 && silhouette_factor < outline_threshold) {
        out_color = vec3(0.0);
        return;
    }

    vec3 ambient = ambient_light * material_diffuse;
    vec3 light_1 = cel_light(normal, view_direction, light_1_position, light_1_color);
    vec3 light_2 = cel_light(normal, view_direction, light_2_position, light_2_color);

    out_color = min(ambient + light_1 + light_2, vec3(1.0));
}
