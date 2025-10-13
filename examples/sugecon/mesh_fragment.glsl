#version 330

in vec3 fragNormal;
out vec4 outColor;

uniform vec3 color;
uniform vec3 lightDir;

void main() {
    float intensity = max(dot(normalize(fragNormal), lightDir), 0.0);
    float celLevel;
    if (intensity > 0.7)
        celLevel = 1.0;
    else if (intensity > 0.35)
        celLevel = 0.6;
    else
        celLevel = 0.3;
    
    outColor = vec4(color * celLevel, 1.0);
}