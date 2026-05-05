#version 330

// Entradas desde el vertex shader
in vec4 fragment_color;
in vec3 fragment_normal;
in vec3 fragment_world_position;
in vec4 position_in_light_space;

out vec4 out_color;

uniform vec3 light_position;
uniform sampler2D shadow_map;

// Perillas controladas por el alumno desde el teclado.
uniform float shadow_bias_min;
uniform float shadow_bias_max;
uniform int pcf_kernel_radius;  // 0 = 1 muestra, 1 = 3x3, 2 = 5x5, 3 = 7x7

vec3 calc_diffuse(vec3 normal_world, vec3 light_dir, vec3 light_color, vec3 material_diffuse) {
    float diff = max(dot(normal_world, light_dir), 0.0);
    return light_color * (diff * material_diffuse);
}

// 1.0 si el fragmento está en sombra, 0.0 si está iluminado.
float calc_shadow(vec4 fragment_position_light_space) {
    vec3 projected_coords = fragment_position_light_space.xyz / fragment_position_light_space.w;
    projected_coords = projected_coords * 0.5 + 0.5;

    if (projected_coords.x < 0.0 || projected_coords.x > 1.0 ||
        projected_coords.y < 0.0 || projected_coords.y > 1.0 ||
        projected_coords.z < 0.0 || projected_coords.z > 1.0) {
        return 0.0;
    }

    float current_depth = projected_coords.z;

    // bias dependiente de la inclinación: superficies casi paralelas a los
    // rayos necesitan más holgura para no shadow-acne. min y max son perillas
    // del alumno.
    vec3 light_dir = normalize(light_position - fragment_world_position);
    float slope_factor = 1.0 - dot(normalize(fragment_normal), light_dir);
    float bias = max(shadow_bias_max * slope_factor, shadow_bias_min);

    // PCF: muestra una vecindad cuadrada de (2r+1)x(2r+1) texels y promedia.
    // pcf_kernel_radius = 0 colapsa a una sola muestra (sin suavizado).
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float shadow = 0.0;
    int sample_count = 0;
    for (int x = -pcf_kernel_radius; x <= pcf_kernel_radius; ++x) {
        for (int y = -pcf_kernel_radius; y <= pcf_kernel_radius; ++y) {
            float closest_depth = texture(
                shadow_map,
                projected_coords.xy + vec2(x, y) * texel_size
            ).r;
            shadow += current_depth - bias > closest_depth ? 1.0 : 0.0;
            sample_count += 1;
        }
    }
    shadow /= float(sample_count);

    if (projected_coords.z > 0.995) {
        shadow = 0.0;
    }

    return shadow;
}

void main() {
    vec3 material_ambient = vec3(0.2, 0.2, 0.2);
    vec3 material_diffuse = vec3(fragment_color.rgb);

    vec3 light_color = vec3(1.0, 1.0, 1.0);
    vec3 light_dir = normalize(light_position - fragment_world_position);
    vec3 diffuse = calc_diffuse(
        normalize(fragment_normal), light_dir, light_color, material_diffuse
    );

    vec3 ambient = material_ambient * vec3(fragment_color.rgb);

    float shadow = calc_shadow(position_in_light_space);
    float in_light = 1.0 - shadow;

    out_color = vec4(ambient + in_light * diffuse, fragment_color.a);
    out_color = min(out_color, vec4(1.0));
}
