#version 330

in float frag_depth;
out vec4 out_color;

void main() {
    // Transición cálido→frío al estilo Gooch: naranja cerca, azul lejos
    vec3 warm = vec3(1.00, 0.60, 0.05);
    vec3 cool = vec3(0.05, 0.15, 0.65);
    out_color = vec4(mix(warm, cool, frag_depth), 1.0);
}