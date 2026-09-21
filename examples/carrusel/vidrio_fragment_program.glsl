#version 330
in vec3 posicion_mundo;
in vec3 normal_mundo;

uniform vec3 color_instancia;
uniform vec3 posicion_camara;

out vec4 color_final;

void main()
{
    vec3 normal = normalize(normal_mundo);
    vec3 hacia_camara = normalize(posicion_camara - posicion_mundo);

    // el vidrio tapa mas en el borde de la silueta que de frente, asi que la
    // opacidad crece donde la normal es casi perpendicular a la vista
    float borde = 1.0 - abs(dot(normal, hacia_camara));
    float opacidad = 0.10 + 0.55 * pow(borde, 2.0);

    color_final = vec4(color_instancia, opacidad);
}
