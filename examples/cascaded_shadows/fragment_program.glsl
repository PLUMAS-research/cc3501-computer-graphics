#version 330

in vec3 fragment_color;
in vec3 fragment_normal;
in vec3 fragment_world_position;
in float fragment_view_depth;
in vec4 position_in_shadow_map_0;
in vec4 position_in_shadow_map_1;
in vec4 position_in_shadow_map_2;
in vec4 position_in_spot_map_0;
in vec4 position_in_spot_map_1;
in vec4 position_in_spot_map_2;

out vec4 out_color;

uniform float ambient;

// El sol: dirección unitaria desde la escena hacia él, e intensidad.
uniform vec3 sun_direction;
uniform float sun_intensity;

uniform sampler2D shadow_map_0;
uniform sampler2D shadow_map_1;
uniform sampler2D shadow_map_2;

// Distancia a la cámara donde termina cada cascada. Solo se usan las
// primeras cascade_count.
uniform vec3 cascade_ends;
uniform int cascade_count;

// El bias de cada cascada en sus unidades de profundidad. Python lo calcula
// a partir de un bias en téxeles, porque el téxel de cada cascada cubre un
// trozo distinto del mundo.
uniform vec3 cascade_biases;
uniform int show_cascades;

// Los focos: posición, eje del cono (unitario) y color de cada uno.
uniform vec3 spot_position_0;
uniform vec3 spot_position_1;
uniform vec3 spot_position_2;
uniform vec3 spot_direction_0;
uniform vec3 spot_direction_1;
uniform vec3 spot_direction_2;
uniform vec3 spot_color_0;
uniform vec3 spot_color_1;
uniform vec3 spot_color_2;
uniform sampler2D spot_shadow_map_0;
uniform sampler2D spot_shadow_map_1;
uniform sampler2D spot_shadow_map_2;

uniform float spot_intensity;
uniform float spot_near;
uniform float spot_far;
// bias de los focos por unidad de distancia al foco (el téxel crece con ella)
uniform float spot_bias_per_distance;

// El spotlight de la unidad de iluminación: I * max(0, cos(theta))^p, con la
// atenuación 1 / (a + b d + c d^2).
const float SPOT_EXPONENT = 12.0;
const vec3 ATTENUATION = vec3(1.0, 0.05, 0.01);

// Tangente del ángulo de incidencia, acotada: sobre una superficie inclinada
// la profundidad cambia dentro de un téxel en esa proporción.
float incidence_slope(float cosine) {
    float bounded = clamp(cosine, 0.05, 1.0);
    return min(sqrt(1.0 - bounded * bounded) / bounded, 4.0);
}

// Sombra del sol en una cascada. 1.0 si está en sombra, 0.0 si no. PCF 3x3.
float cascade_shadow(sampler2D shadow_map, vec4 shadow_map_position, float bias) {
    vec3 projected_coords = shadow_map_position.xyz / shadow_map_position.w;
    if (projected_coords.z > 1.0) {
        return 0.0;
    }

    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float closest_depth = texture(
                shadow_map, projected_coords.xy + vec2(x, y) * texel_size
            ).r;
            shadow += projected_coords.z - bias > closest_depth ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

// De la profundidad guardada, en [0, 1], a la profundidad en la vista del
// foco: deshace su proyección en perspectiva.
float spot_view_depth(float depth) {
    float depth_ndc = 2.0 * depth - 1.0;
    return 2.0 * spot_near * spot_far
        / (spot_far + spot_near - depth_ndc * (spot_far - spot_near));
}

// Sombra de un foco. Con perspectiva la profundidad guardada no es
// proporcional a la distancia, así que se comparan profundidades de la vista
// del foco, en unidades de la escena.
float spot_shadow(sampler2D shadow_map, vec4 shadow_map_position, float slope) {
    if (shadow_map_position.w <= 0.0) {
        return 0.0;
    }
    vec3 projected_coords = shadow_map_position.xyz / shadow_map_position.w;
    if (projected_coords.z > 1.0) {
        return 0.0;
    }

    float current_depth = spot_view_depth(projected_coords.z);
    float bias = spot_bias_per_distance * current_depth * (1.0 + slope);
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float closest_depth = spot_view_depth(texture(
                shadow_map, projected_coords.xy + vec2(x, y) * texel_size
            ).r);
            shadow += current_depth - bias > closest_depth ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

vec3 spot_light(vec3 normal, vec3 light_position, vec3 light_axis, vec3 light_color,
                sampler2D shadow_map, vec4 shadow_map_position) {
    vec3 to_light = light_position - fragment_world_position;
    float light_distance = length(to_light);
    vec3 light_dir = to_light / light_distance;

    float cone = pow(max(dot(-light_dir, light_axis), 0.0), SPOT_EXPONENT);
    float cosine = max(dot(normal, light_dir), 0.0);
    if (spot_intensity == 0.0 || cone * cosine == 0.0) {
        return vec3(0.0);
    }

    float attenuation = 1.0 / (ATTENUATION.x + ATTENUATION.y * light_distance
                               + ATTENUATION.z * light_distance * light_distance);
    float shadow = spot_shadow(shadow_map, shadow_map_position, incidence_slope(cosine));
    return light_color * spot_intensity * cone * attenuation * cosine * (1.0 - shadow);
}

void main() {
    vec3 normal = normalize(fragment_normal);
    float sun_cosine = max(dot(normal, sun_direction), 0.0);

    // La primera cascada cuyo tramo contiene al punto. GLSL 330 no permite
    // indexar un arreglo de samplers con una variable, por eso los if.
    int cascade = cascade_count - 1;
    for (int index = cascade_count - 2; index >= 0; --index) {
        if (fragment_view_depth < cascade_ends[index]) {
            cascade = index;
        }
    }

    float bias = cascade_biases[cascade] * (1.0 + incidence_slope(sun_cosine));
    float sun_shadow;
    if (cascade == 0) {
        sun_shadow = cascade_shadow(shadow_map_0, position_in_shadow_map_0, bias);
    } else if (cascade == 1) {
        sun_shadow = cascade_shadow(shadow_map_1, position_in_shadow_map_1, bias);
    } else {
        sun_shadow = cascade_shadow(shadow_map_2, position_in_shadow_map_2, bias);
    }

    vec3 light = vec3(ambient + sun_intensity * sun_cosine * (1.0 - sun_shadow));
    light += spot_light(normal, spot_position_0, spot_direction_0, spot_color_0,
                        spot_shadow_map_0, position_in_spot_map_0);
    light += spot_light(normal, spot_position_1, spot_direction_1, spot_color_1,
                        spot_shadow_map_1, position_in_spot_map_1);
    light += spot_light(normal, spot_position_2, spot_direction_2, spot_color_2,
                        spot_shadow_map_2, position_in_spot_map_2);
    vec3 color = fragment_color * light;

    // modo de depuración: tiñe cada píxel con el color de su cascada
    if (show_cascades == 1) {
        vec3 tints[3] = vec3[3](vec3(1.0, 0.35, 0.35), vec3(0.35, 1.0, 0.35), vec3(0.4, 0.5, 1.0));
        color = mix(color, tints[cascade], 0.35);
    }
    out_color = vec4(min(color, vec3(1.0)), 1.0);
}
