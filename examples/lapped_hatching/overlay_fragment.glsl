#version 330

in vec2 fragment_patch_uv;

out vec4 fragment_color;

void main() {
    vec2 local = fragment_patch_uv - 0.5;

    vec3 paper_color = vec3(0.97, 0.94, 0.84);
    vec3 ink_color = vec3(0.18, 0.12, 0.08);
    vec3 frame_color = vec3(0.20, 0.15, 0.10);

    // Una "celda de referencia" del hatching: stripes periódicos en Y y
    // cross-hatch a 60 grados, sin máscara alfa. Replica el aspecto que el
    // fragment shader principal genera dentro de cada parche.
    float stripe_period = 0.18;
    float stripe_half_width = 0.022;

    float stripe_distance = abs(fract(local.y / stripe_period + 0.5) - 0.5)
                            * stripe_period;
    float primary = 1.0 - smoothstep(stripe_half_width * 0.7,
                                     stripe_half_width,
                                     stripe_distance);

    float cos_a = cos(1.0472);
    float sin_a = sin(1.0472);
    vec2 rotated = vec2(cos_a * local.x - sin_a * local.y,
                        sin_a * local.x + cos_a * local.y);
    float rotated_distance = abs(fract(rotated.y / stripe_period + 0.5) - 0.5)
                             * stripe_period;
    float secondary = 1.0 - smoothstep(stripe_half_width * 0.7,
                                       stripe_half_width,
                                       rotated_distance);

    float ink_amount = primary + (1.0 - primary) * secondary * 0.5;
    vec3 color = mix(paper_color, ink_color, ink_amount);

    // Marco del preview
    vec2 border_distance = min(fragment_patch_uv, 1.0 - fragment_patch_uv);
    float inside_mask = smoothstep(0.0, 0.02,
                                   min(border_distance.x, border_distance.y));
    color = mix(frame_color, color, inside_mask);

    fragment_color = vec4(color, 1.0);
}
