#version 330

in vec3 fragment_normal;
in vec2 fragment_patch_uv;
in vec3 fragment_patch_color;
in float fragment_boundary_flag;
in vec3 fragment_barycentric;
in float fragment_alpha;
in float fragment_alignment_offset;

uniform float stripe_period;
uniform float stripe_half_width;
uniform int show_uv_gradient;
uniform int show_patches;
uniform int hatching_layers;
uniform int align_patches;

uniform sampler3D tam_volume;
uniform int tam_levels;
uniform vec3 light_direction;
uniform float tile_world_size;
uniform int use_tam;

out vec4 fragment_color;

// 60 grados: tres direcciones igualmente espaciadas (0, +60, -60). Es el
// reparto que tejedores y dibujantes usan para tartán y trama densa.
const float CROSS_ANGLE = 1.0472;

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

vec2 apply_alignment(vec2 uv) {
    if (align_patches == 0) {
        return uv;
    }
    float c = cos(fragment_alignment_offset);
    float s = sin(fragment_alignment_offset);
    return vec2(c * uv.x + s * uv.y, c * uv.y - s * uv.x);
}

float ink_from_tam(vec2 uv, float tone) {
    // Mapeo del tono [0, 1] al centro de cada slice del volumen 3D.
    // Con N niveles los centros estan en (k + 0.5) / N. La interpolacion
    // lineal entre slices la hace el sampler.
    float r = (tone * float(tam_levels - 1) + 0.5) / float(tam_levels);
    vec2 tex_uv = uv / tile_world_size;
    return texture(tam_volume, vec3(tex_uv, r)).r;
}

void main() {
    vec2 uv = apply_alignment(fragment_patch_uv);

    if (show_uv_gradient != 0) {
        vec2 wrapped = fract(uv / (stripe_period * 6.0) + 0.5);
        fragment_color = vec4(wrapped.x, wrapped.y, 0.55, 1.0);
        return;
    }

    if (show_patches != 0) {
        float edge_distance = min(min(fragment_barycentric.x,
                                      fragment_barycentric.y),
                                  fragment_barycentric.z);
        float edge_pixels = fwidth(edge_distance);
        float edge_mask = 1.0 - smoothstep(0.0, edge_pixels * 1.5,
                                           edge_distance);
        float seam = edge_mask * fragment_boundary_flag;
        vec3 seam_color = vec3(0.12, 0.08, 0.06);
        vec3 color = mix(fragment_patch_color, seam_color, seam);
        fragment_color = vec4(color, 1.0);
        return;
    }

    float ink_amount;
    if (use_tam != 0) {
        // Difuso lambertiano. La normal viene del vertex shader interpolada.
        // En areas iluminadas la "tinta" cae, en areas oscuras crece.
        vec3 n = normalize(fragment_normal);
        float diffuse = max(0.0, dot(n, light_direction));
        // Pequena luz ambiente para que las zonas en sombra no queden
        // completamente cubiertas de tinta.
        float lit = 0.12 + 0.88 * diffuse;
        float tone = 1.0 - lit;
        ink_amount = ink_from_tam(uv, tone);
    } else {
        // Achurado procedural sin iluminacion: una a tres capas de trazos
        // a 0, +60 y -60 grados con pesos fijos.
        float primary = stripe_mask_at(uv, 0.0);
        ink_amount = primary;
        if (hatching_layers >= 2) {
            float secondary = stripe_mask_at(uv, CROSS_ANGLE);
            ink_amount += (1.0 - ink_amount) * secondary * 0.6;
        }
        if (hatching_layers >= 3) {
            float tertiary = stripe_mask_at(uv, -CROSS_ANGLE);
            ink_amount += (1.0 - ink_amount) * tertiary * 0.35;
        }
    }

    vec3 paper_color = vec3(0.97, 0.94, 0.84);
    vec3 ink_color = vec3(0.18, 0.12, 0.08);
    fragment_color = vec4(mix(paper_color, ink_color, ink_amount),
                          fragment_alpha);
}
