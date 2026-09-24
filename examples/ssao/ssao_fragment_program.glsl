#version 330

// Segunda pasada: el factor de oclusión, un valor por píxel.
//
// Para cada píxel se toman muestras dentro del hemisferio orientado según su
// normal, se proyectan a pantalla y se compara la profundidad de la muestra
// con la que el framebuffer de geometría ya registró en ese lugar. Una
// muestra que queda por detrás de la superficie registrada significa que hay
// geometría entre el punto y esa dirección, o sea que esa dirección está
// tapada.

#define MAX_SAMPLES 64

in vec2 fragment_texcoord;
out float out_occlusion;

uniform sampler2D position_texture;
uniform sampler2D normal_texture;
uniform sampler2D noise_texture;

uniform mat4 projection;
uniform vec3 kernel[MAX_SAMPLES];
uniform vec2 noise_scale;
uniform float sample_radius;
uniform float depth_bias;
uniform float occlusion_strength;
uniform int sample_count;

void main() {
    vec3 fragment_position = texture(position_texture, fragment_texcoord).xyz;
    vec3 normal = normalize(texture(normal_texture, fragment_texcoord).xyz);

    // El fondo no tiene geometría: su posición quedó en cero y no se ocluye.
    if (length(fragment_position) < 0.0001f) {
        out_occlusion = 1.0f;
        return;
    }

    // Base ortonormal del hemisferio. El vector de ruido gira la base de un
    // píxel al siguiente, de modo que cada uno muestrea direcciones distintas:
    // con la misma base en todos, las mismas pocas muestras producen bandas.
    vec3 random_vector = normalize(texture(noise_texture, fragment_texcoord * noise_scale).xyz);
    vec3 tangent = normalize(random_vector - normal * dot(random_vector, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tangent_to_view = mat3(tangent, bitangent, normal);

    float occluded_samples = 0.0f;

    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        vec3 sample_position =
            fragment_position + tangent_to_view * kernel[sample_index] * sample_radius;

        // De espacio de vista a coordenadas de textura: proyectar, dividir por
        // w para llegar a NDC y llevar el rango [-1, 1] al [0, 1].
        vec4 projected_sample = projection * vec4(sample_position, 1.0f);
        projected_sample.xyz /= projected_sample.w;
        vec2 sample_texcoord = projected_sample.xy * 0.5f + 0.5f;

        float registered_depth = texture(position_texture, sample_texcoord).z;

        // En espacio de vista la cámara mira hacia -z, así que una profundidad
        // mayor es una superficie más cercana. El sesgo evita que una
        // superficie plana se ocluya a sí misma por el error de precisión.
        float is_occluded = registered_depth >= sample_position.z + depth_bias ? 1.0f : 0.0f;

        // Corrección de rango: una superficie mucho más cercana que el radio
        // de muestreo está delante del objeto, no lo rodea, y no debe
        // oscurecerlo. Sin esto aparece un halo alrededor de cada silueta.
        float range_check = smoothstep(
            0.0f, 1.0f, sample_radius / abs(fragment_position.z - registered_depth));

        occluded_samples += is_occluded * range_check;
    }

    // Se invierte para que el resultado se pueda multiplicar directamente por
    // la luz ambiental: 1 es sin oclusión y 0 es completamente tapado. La
    // intensidad es un parámetro de ajuste, no una cantidad física: escala la
    // fracción tapada antes de invertirla.
    float occluded_fraction = occluded_samples / float(sample_count);
    out_occlusion = clamp(1.0f - occlusion_strength * occluded_fraction, 0.0f, 1.0f);
}
