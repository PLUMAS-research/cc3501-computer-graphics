#version 330

// Muestra en pantalla completa una de las texturas intermedias. Es la forma
// de ver qué produce cada pasada por separado.

in vec2 fragment_texcoord;
out vec4 out_color;

uniform sampler2D debug_texture;
uniform int debug_mode;   // 0: un canal en escala de grises, 1: normales

void main() {
    if (debug_mode == 1) {
        // Las normales van de -1 a 1 y la pantalla muestra de 0 a 1.
        vec3 normal = texture(debug_texture, fragment_texcoord).xyz;
        out_color = vec4(normal * 0.5f + 0.5f, 1.0f);
    } else {
        float value = texture(debug_texture, fragment_texcoord).r;
        out_color = vec4(vec3(value), 1.0f);
    }
}
