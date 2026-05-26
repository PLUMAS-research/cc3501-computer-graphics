#version 330

in vec2 frag_texcoord;
out vec4 out_color;

// la textura tiene un solo canal (R): el valor de temperatura de cada celda.
uniform sampler2D campo_temperatura;

// colormap tipo "inferno": frio oscuro -> caliente claro.
vec3 colormap_calor(float t) {
    t = clamp(t, 0.0f, 1.0f);
    vec3 c0 = vec3(0.0f, 0.0f, 0.05f);
    vec3 c1 = vec3(0.35f, 0.05f, 0.45f);
    vec3 c2 = vec3(0.85f, 0.25f, 0.20f);
    vec3 c3 = vec3(0.98f, 0.75f, 0.20f);
    vec3 c4 = vec3(1.0f, 1.0f, 0.90f);
    if (t < 0.25f) return mix(c0, c1, t / 0.25f);
    if (t < 0.50f) return mix(c1, c2, (t - 0.25f) / 0.25f);
    if (t < 0.75f) return mix(c2, c3, (t - 0.50f) / 0.25f);
    return mix(c3, c4, (t - 0.75f) / 0.25f);
}

void main() {
    float temperatura = texture(campo_temperatura, frag_texcoord).r;
    out_color = vec4(colormap_calor(temperatura), 1.0f);
}
