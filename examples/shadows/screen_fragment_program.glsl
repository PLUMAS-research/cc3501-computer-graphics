#version 330

in vec2 fragment_texcoord;
out vec4 out_color;
uniform sampler2D sampler_tex;

void main() {
    // El shadow map almacena profundidad en [0, 1]. Para visualizarlo
    // expandimos el rango cercano a la luz, donde está casi toda la
    // información útil (la cámara de luz tiene near/far comprimido).
    float depth = texture(sampler_tex, fragment_texcoord).r;
    float visualized = pow(depth, 16.0);
    out_color = vec4(vec3(visualized), 1.0);
}
