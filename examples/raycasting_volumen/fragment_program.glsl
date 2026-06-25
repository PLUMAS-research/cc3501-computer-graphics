#version 330

// Ray casting de volumen sobre una textura 3D de unidades Hounsfield (HU).
//
// Cada pixel lanza un rayo desde la camara. El rayo se intersecta con la caja
// que contiene el volumen ([-box_half, box_half]) por el metodo de los slabs, y
// luego se recorre con paso fijo muestreando la textura 3D. Las muestras fuera
// de la caja de recorte ([clip_min, clip_max]) se descartan. Lo que se hace con
// las muestras restantes depende del modo de composicion (el apunte, "Funciones
// simples del perfil" y "Composicion volumetrica"):
//
//   modo 0  MIP         : el valor maximo de HU encontrado en el rayo.
//   modo 1  promedio    : el promedio de HU (parecido a una radiografia).
//   modo 2  compositing : composicion front-to-back con funcion de transferencia.
//
// La interpolacion (vecino mas cercano vs trilineal) la decide el filtro de la
// textura, no este programa.

out vec4 out_color;

uniform vec2 viewport_origin;    // esquina inferior izquierda del viewport 3D (pixeles).
uniform vec2 viewport_size;      // tamano del viewport 3D (pixeles).
uniform float camera_yaw;        // orbita horizontal de la camara (radianes).
uniform float camera_pitch;      // orbita vertical de la camara (radianes).
uniform int mode;                // 0 MIP, 1 promedio, 2 compositing.
uniform float step_size;         // paso de muestreo en unidades de mundo.
uniform float voxel_size;        // tamano de un voxel en unidades de mundo.
uniform int early_termination;   // 1 corta el rayo cuando el alpha se satura.
uniform vec3 box_half;           // medios lados: la caja es [-box_half, box_half] centrada en el origen.
uniform vec3 clip_min;           // esquina minima de la caja de recorte (en [0,1] de textura).
uniform vec3 clip_max;           // esquina maxima de la caja de recorte (en [0,1] de textura).
uniform sampler3D volume;        // valores en unidades Hounsfield (HU).

const int MAX_STEPS = 2048;
const float HU_MIN = -1000.0;
const float HU_MAX = 1000.0;

float normaliza_hu(float hu) {
    return clamp((hu - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0);
}

// Funcion de transferencia HU -> RGBA: color emitido y opacidad por muestra.
// Sigue el esquema del apunte: aire transparente, tejido rojo semitransparente,
// hueso blanco casi opaco.
// Las opacidades son por muestra y deliberadamente bajas: el medio se compone
// translucido para que el rayo acumule las capas en profundidad (el interior se
// ve a traves del reborde de hueso) en vez de saturarse en la primera capa.
vec4 transferencia(float hu) {
    if (hu > 300.0)  return vec4(1.00, 1.00, 1.00, 0.22);  // hueso
    if (hu > -100.0) return vec4(0.85, 0.25, 0.20, 0.02);  // tejido blando
    return vec4(0.0);                                       // aire
}

// Interseccion rayo-caja [-box_half, box_half] por el metodo de los slabs. Las
// divisiones por componentes nulas dan inf, que min/max manejan sin caso
// especial. La caja no es siempre cubica: un volumen CT anisotropico (menos
// cortes en z) usa medios lados distintos por eje para no salir aplastado.
bool intersecta_caja(vec3 origen, vec3 direccion, out float t_cerca, out float t_lejos) {
    vec3 reciproco = 1.0 / direccion;            // 1/d por componente: multiplicar en vez de dividir
    vec3 t0 = (-box_half - origen) * reciproco;  // cruce con el plano -box_half de cada eje
    vec3 t1 = ( box_half - origen) * reciproco;  // cruce con el plano +box_half de cada eje
    vec3 entrada = min(t0, t1);                 // por eje: el cruce menor es la entrada
    vec3 salida  = max(t0, t1);                 // por eje: el cruce mayor es la salida
    t_cerca = max(max(entrada.x, entrada.y), entrada.z);  // ultima entrada -> t_min
    t_lejos = min(min(salida.x, salida.y), salida.z);     // primera salida -> t_max
    return t_lejos >= max(t_cerca, 0.0);
}

void main() {
    // pixel a coordenada de pantalla del viewport 3D, centrada y por aspecto.
    vec2 screen = (gl_FragCoord.xy - viewport_origin - 0.5 * viewport_size) / viewport_size.y;

    // camara que orbita el origen a radio fijo segun yaw y pitch.
    float radius = 3.0;
    vec3 camera_position = radius * vec3(
        cos(camera_pitch) * sin(camera_yaw),
        sin(camera_pitch),
        cos(camera_pitch) * cos(camera_yaw)
    );
    vec3 forward = normalize(-camera_position);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 ray_direction = normalize(forward + screen.x * right + screen.y * up);

    vec3 fondo = vec3(0.05, 0.06, 0.09);

    float t_cerca, t_lejos;
    if (!intersecta_caja(camera_position, ray_direction, t_cerca, t_lejos)) {
        out_color = vec4(fondo, 1.0);
        return;
    }
    t_cerca = max(t_cerca, 0.0);

    // acumuladores del recorrido.
    float maximo = HU_MIN;
    float suma = 0.0;
    int conteo = 0;
    vec3 color_acumulado = vec3(0.0);
    float alpha_acumulado = 0.0;

    // correccion de opacidad: escala el alpha de cada muestra segun el tamano del
    // paso, asi la apariencia del compositing se mantiene estable al cambiar k.
    float exponente = step_size / voxel_size;

    float t = t_cerca;
    for (int i = 0; i < MAX_STEPS; i++) {
        if (t > t_lejos) break;
        vec3 punto = camera_position + t * ray_direction;
        vec3 coordenada = 0.5 * (punto / box_half + 1.0);  // caja -> [0,1] textura.

        // recorte: las muestras fuera de la caja de recorte no aportan al color.
        if (any(lessThan(coordenada, clip_min)) || any(greaterThan(coordenada, clip_max))) {
            t += step_size;
            continue;
        }
        float hu = texture(volume, coordenada).r;

        if (mode == 0) {
            maximo = max(maximo, hu);
        } else if (mode == 1) {
            suma += hu;
            conteo += 1;
        } else {
            vec4 muestra = transferencia(hu);
            float alpha = 1.0 - pow(1.0 - muestra.a, exponente);
            color_acumulado += (1.0 - alpha_acumulado) * muestra.rgb * alpha;
            alpha_acumulado += (1.0 - alpha_acumulado) * alpha;
            if (early_termination == 1 && alpha_acumulado > 0.99) break;
        }
        t += step_size;
    }

    vec3 resultado;
    if (mode == 0) {
        resultado = vec3(normaliza_hu(maximo));
    } else if (mode == 1) {
        float promedio = conteo > 0 ? suma / float(conteo) : HU_MIN;
        resultado = vec3(normaliza_hu(promedio));
    } else {
        resultado = color_acumulado + (1.0 - alpha_acumulado) * fondo;
    }

    out_color = vec4(resultado, 1.0);
}
