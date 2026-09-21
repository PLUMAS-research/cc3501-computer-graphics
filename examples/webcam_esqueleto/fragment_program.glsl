#version 330
in vec3 normal_mundo;

uniform vec3 color_instancia;

out vec4 color_final;

// una luz direccional fija y un relleno constante: el ejemplo trata sobre la
// jerarquia de huesos, no sobre iluminacion
const vec3 DIRECCION_LUZ = normalize(vec3(0.4, 0.8, 0.6));

void main()
{
    vec3 normal = normalize(normal_mundo);
    float difusa = max(dot(normal, DIRECCION_LUZ), 0.0);
    color_final = vec4(color_instancia * (0.38 + 0.75 * difusa), 1.0);
}
