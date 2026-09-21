#version 330
in vec3 posicion_mundo;
in vec3 normal_mundo;

uniform vec3 color_instancia;
uniform vec3 posicion_ampolleta;
uniform vec3 posicion_camara;
uniform float emisivo;

out vec4 color_final;

void main()
{
    vec3 normal = normalize(normal_mundo);
    vec3 hacia_luz = normalize(posicion_ampolleta - posicion_mundo);
    vec3 hacia_camara = normalize(posicion_camara - posicion_mundo);

    float distancia = length(posicion_ampolleta - posicion_mundo);
    float atenuacion = 1.0 / (1.0 + 0.10 * distancia + 0.06 * distancia * distancia);

    float difusa = max(dot(normal, hacia_luz), 0.0) * atenuacion;

    vec3 vector_medio = normalize(hacia_luz + hacia_camara);
    float especular = pow(max(dot(normal, vector_medio), 0.0), 48.0) * atenuacion;

    // dos rellenos, porque la ampolleta sola deja negras todas las caras que
    // miran hacia afuera del carrusel: uno cenital y otro desde la camara
    float relleno = 0.34
                  + 0.20 * max(normal.y, 0.0)
                  + 0.34 * max(dot(normal, hacia_camara), 0.0);

    vec3 color = color_instancia * (relleno + 2.4 * difusa)
               + vec3(1.0, 0.95, 0.85) * 1.6 * especular;

    // la ampolleta es la fuente: iluminarla con ella misma la dejaria oscura,
    // porque su normal apunta en sentido contrario a su propio centro
    color = mix(color, color_instancia, emisivo);

    color_final = vec4(min(color, vec3(1.0)), 1.0);
}
