#version 330

in vec3 position;
in vec3 normal;
in vec3 normal_cara;
in vec2 uv_modelo;
in float capa_modelo;
in vec2 uv_contorno;
in float margen;
in float escala_parche;
in vec2 desplazamiento;
in float tono_parche;

uniform mat4 view;
uniform mat4 projection;
// dirección hacia la luz en el espacio de la cámara
uniform vec3 direccion_luz;
// función de transferencia de tono: el brillo es ambiente + exposicion * (N . L),
// recortado a [0, 1]. Con exposicion alta las caras que miran a la luz quedan
// en blanco y la luz ambiente aclara las que no la reciben
uniform float exposicion;
uniform float ambiente;
// lado de una repetición del TAM, en unidades del modelo
uniform float lado_tam;
// 0: tono según la luz; 1: un tono al azar por parche
uniform int tono_por_parche;
// 0: tono continuo por vértice (mezcla de 6 vías); 1: tono redondeado por
// vértice; 2: tono continuo por cara; 3: tono redondeado por cara
uniform int esquema_mezcla;

out vec2 uv_tam;
out vec2 uv_mascara;
out float alfa_margen;
out vec2 uv_textura;
flat out float capa_textura;
// pesos de los seis tonos del TAM: los tres claros y los tres oscuros. El
// blanco del papel es el séptimo tono, con el peso que falta para sumar 1
out vec3 pesos_claros;
out vec3 pesos_oscuros;
flat out vec3 pesos_claros_cara;
flat out vec3 pesos_oscuros_cara;

// la pre-pasada de profundidad calcula la misma posición con otro programa;
// invariant garantiza que ambos programas den la misma profundidad, bit a bit
invariant gl_Position;

float tono_con_luz(vec3 normal_mundo, float lado) {
    vec3 normal_vista = lado * normalize(mat3(view) * normal_mundo);
    float brillo = clamp(ambiente + exposicion * max(dot(normal_vista, direccion_luz), 0.0), 0.0, 1.0);
    // 0 es el papel blanco y 6 el tono más oscuro del TAM
    return 6.0 * (1.0 - brillo);
}

// cada tono k recibe 1 - |tono - k| si es positivo: entre los tonos k y k+1 la
// mezcla es lineal y el resto queda en 0
void pesos(float tono, out vec3 claros, out vec3 oscuros) {
    claros = max(1.0 - abs(tono - vec3(1.0, 2.0, 3.0)), 0.0);
    oscuros = max(1.0 - abs(tono - vec3(4.0, 5.0, 6.0)), 0.0);
}

void main() {
    gl_Position = projection * view * vec4(position, 1.0);

    // en una malla abierta se puede ver el reverso de una cara: entonces se
    // ilumina con las normales invertidas. Las tres esquinas de una cara
    // tienen la misma normal de cara, así que la decisión vale para toda ella
    vec3 posicion_vista = (view * vec4(position, 1.0)).xyz;
    float lado = dot(mat3(view) * normal_cara, -posicion_vista) < 0.0 ? -1.0 : 1.0;

    float tono_vertice = tono_por_parche == 1 ? tono_parche : tono_con_luz(normal, lado);
    float tono_cara = tono_por_parche == 1 ? tono_parche : tono_con_luz(normal_cara, lado);
    if (esquema_mezcla == 1) {
        tono_vertice = round(tono_vertice);
    }
    if (esquema_mezcla == 3) {
        tono_cara = round(tono_cara);
    }
    pesos(tono_vertice, pesos_claros, pesos_oscuros);
    pesos(tono_cara, pesos_claros_cara, pesos_oscuros_cara);

    uv_mascara = uv_contorno;
    alfa_margen = margen;
    uv_textura = uv_modelo;
    capa_textura = capa_modelo;
    // el contorno mide escala_parche unidades del modelo y el TAM, lado_tam:
    // así los trazos tienen el mismo tamaño en todos los parches
    uv_tam = uv_contorno * escala_parche / lado_tam + desplazamiento;
}
