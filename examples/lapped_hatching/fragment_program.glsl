#version 330

in vec2 fragment_patch_uv;
in vec3 fragment_patch_color;

uniform float stripe_period;
uniform float stripe_half_width;
uniform int show_uv_gradient;
uniform int show_patches;

out vec4 fragment_color;

const float CROSS_ANGLE = 1.0472; // 60 grados

// Trazo periódico en Y dentro de la UV del parche. La UV ya viene en
// unidades del mundo, así que el período es estable entre parches.
float stripe_mask_at(vec2 uv, float angle) {
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    vec2 rotated = vec2(cos_a * uv.x - sin_a * uv.y,
                        sin_a * uv.x + cos_a * uv.y);
    float stripe_distance = abs(fract(rotated.y / stripe_period + 0.5) - 0.5)
                            * stripe_period;
    return 1.0 - smoothstep(stripe_half_width * 0.7,
                            stripe_half_width,
                            stripe_distance);
}

void main() {
    if (show_uv_gradient != 0) {
        // Diagnóstico: la UV del parche, envuelta para que se lean los trazos.
        vec2 wrapped = fract(fragment_patch_uv / (stripe_period * 6.0) + 0.5);
        fragment_color = vec4(wrapped.x, wrapped.y, 0.55, 1.0);
        return;
    }

    if (show_patches != 0) {
        // Diagnóstico: color plano por parche con gradiente UV superpuesto.
        vec2 wrapped = fract(fragment_patch_uv / (stripe_period * 6.0) + 0.5);
        vec3 uv_tint = vec3(wrapped.x, wrapped.y, 0.6);
        vec3 color = mix(fragment_patch_color, uv_tint, 0.35);
        fragment_color = vec4(color, 1.0);
        return;
    }

    // Tres capas de achurado con pesos fijos (sin iluminación en esta
    // unidad). Los ángulos se toman dentro del marco del parche: como la
    // UV cambia de orientación entre parches, los trazos también.
    float primary = stripe_mask_at(fragment_patch_uv, 0.0);
    float secondary = stripe_mask_at(fragment_patch_uv, CROSS_ANGLE);
    float tertiary = stripe_mask_at(fragment_patch_uv, -CROSS_ANGLE);

    float ink_amount = primary;
    ink_amount += (1.0 - ink_amount) * secondary * 0.6;
    ink_amount += (1.0 - ink_amount) * tertiary * 0.35;

    vec3 paper_color = vec3(0.97, 0.94, 0.84);
    vec3 ink_color = vec3(0.18, 0.12, 0.08);
    fragment_color = vec4(mix(paper_color, ink_color, ink_amount), 1.0);
}
