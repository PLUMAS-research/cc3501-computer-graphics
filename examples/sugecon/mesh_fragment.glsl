#version 330

in vec3 fragNormal;
in float fragValue;
out vec4 outColor;

uniform vec3 lightDir;
// 0 = superficie blanca, como en el paper; >0 = colormap del campo escalar fragValue.
uniform int fieldMode;

// Colormap divergente azul-blanco-rojo, centrado en cero. El valor cero queda
// blanco, así la banda blanca coincide con el cruce por cero de kr (que es
// justo el suggestive contour) y el signo de H/K se lee por el color.
vec3 diverging(float t) {
    vec3 negative = vec3(0.20, 0.32, 0.70);
    vec3 middle = vec3(0.96, 0.95, 0.90);
    vec3 positive = vec3(0.80, 0.20, 0.15);
    if (t < 0.0)
        return mix(middle, negative, clamp(-t, 0.0, 1.0));
    return mix(middle, positive, clamp(t, 0.0, 1.0));
}

void main() {
    if (fieldMode == 0) {
        // la superficie solo tapa las líneas ocultas: el dibujo son las líneas
        outColor = vec4(1.0);
        return;
    }
    // campo de curvatura, atenuado por la iluminación para conservar forma
    float intensity = max(dot(normalize(fragNormal), normalize(lightDir)), 0.0);
    outColor = vec4(diverging(fragValue) * (0.55 + 0.45 * intensity), 1.0);
}
