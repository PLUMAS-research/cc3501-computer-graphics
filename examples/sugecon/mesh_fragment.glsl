#version 330

in vec3 fragNormal;
in float fragValue;
out vec4 outColor;

uniform vec3 lightDir;
// 0 = relleno papel (cel-shading), >0 = colormap del campo escalar fragValue.
uniform int fieldMode;

// Colormap divergente azul-papel-rojo, centrado en cero. El valor cero queda
// blanco-papel, así la banda blanca coincide con el cruce por cero de kr (que
// es justo el suggestive contour) y el signo de H/K se lee por el color.
vec3 diverging(float t) {
    vec3 negative = vec3(0.20, 0.32, 0.70);
    vec3 middle = vec3(0.96, 0.95, 0.90);
    vec3 positive = vec3(0.80, 0.20, 0.15);
    if (t < 0.0)
        return mix(middle, negative, clamp(-t, 0.0, 1.0));
    return mix(middle, positive, clamp(t, 0.0, 1.0));
}

void main() {
    float intensity = max(dot(normalize(fragNormal), normalize(lightDir)), 0.0);

    vec3 base;
    if (fieldMode == 0) {
        // Relleno papel con cel-shading suave en tono cálido.
        vec3 paper = vec3(0.93, 0.90, 0.82);
        float celLevel;
        if (intensity > 0.6)
            celLevel = 1.0;
        else if (intensity > 0.3)
            celLevel = 0.86;
        else
            celLevel = 0.72;
        base = paper * celLevel;
    } else {
        // Campo de curvatura, atenuado por la iluminación para conservar forma.
        base = diverging(fragValue) * (0.55 + 0.45 * intensity);
    }

    outColor = vec4(base, 1.0);
}
