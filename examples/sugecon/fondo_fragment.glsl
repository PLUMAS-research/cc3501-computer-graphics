#version 330

in vec2 uv;
out vec4 outColor;

// Ruido hash barato para el grano del papel.
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec3 base = vec3(0.95, 0.93, 0.85);
    // Viñeta: el papel se oscurece hacia los bordes.
    float vignette = smoothstep(0.95, 0.25, length(uv - 0.5));
    float grain = (hash(uv * 900.0) - 0.5) * 0.04;
    vec3 color = base * mix(0.80, 1.0, vignette) + grain;
    outColor = vec4(color, 1.0);
}
