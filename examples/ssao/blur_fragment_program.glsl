#version 330

// Tercera pasada: el promedio del factor de oclusión en una vecindad del
// tamaño de la textura de ruido. El ruido que evita las bandas deja un patrón
// repetido de 4 x 4 píxeles, y promediar esa misma vecindad lo cancela.

in vec2 fragment_texcoord;
out float out_occlusion;

uniform sampler2D occlusion_texture;
uniform int blur_radius;

void main() {
    vec2 texel_size = 1.0f / vec2(textureSize(occlusion_texture, 0));
    float accumulated = 0.0f;
    float samples = 0.0f;

    for (int x = -blur_radius; x <= blur_radius; ++x) {
        for (int y = -blur_radius; y <= blur_radius; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            accumulated += texture(occlusion_texture, fragment_texcoord + offset).r;
            samples += 1.0f;
        }
    }

    out_occlusion = accumulated / samples;
}
