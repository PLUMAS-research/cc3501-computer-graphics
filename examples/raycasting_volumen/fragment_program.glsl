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
// Dos opciones cambian como se ve cada modo. Con iluminacion, el compositing
// sombrea cada muestra con Phong, usando el gradiente del volumen como normal. Con
// profundidad, el MIP pinta cada pixel segun la distancia a la que estaba el
// maximo, que es justo lo que el MIP descarta.
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
uniform sampler3D mascara;       // segmentacion: la etiqueta de cada voxel (0 = fondo), en R8.
uniform int usar_mascara;        // 1 si se cargo una segmentacion.
uniform int iluminacion;         // 1 sombrea el compositing con Phong.
uniform int profundidad_mip;     // 1 colorea el MIP segun la distancia del maximo.

const int MAX_STEPS = 4096;
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

// Lo segmentado reemplaza a la funcion de transferencia en las muestras que caen
// dentro de la mascara, con un color por etiqueta y una opacidad alta, para que se
// vea entre el tejido.
const float ALFA_MASCARA = 0.60;
const vec3 PALETA[8] = vec3[8](
    vec3(1.00, 0.80, 0.10), vec3(0.20, 0.85, 1.00), vec3(0.55, 1.00, 0.35),
    vec3(1.00, 0.45, 0.85), vec3(1.00, 0.55, 0.15), vec3(0.65, 0.55, 1.00),
    vec3(0.95, 1.00, 0.55), vec3(0.35, 1.00, 0.80));

// La textura guarda la etiqueta dividida por 255; se lee con GL_NEAREST, asi que
// al multiplicar por 255 vuelve a ser un entero.
float etiqueta(vec3 coordenada) {
    if (usar_mascara == 0) return 0.0;
    return round(texture(mascara, coordenada).r * 255.0);
}

vec3 color_etiqueta(float valor) {
    return PALETA[int(mod(valor - 1.0, 8.0))];
}

// Gradiente del volumen por diferencias centrales, en HU por voxel. Apunta hacia
// donde el valor crece: en el borde de un hueso, hacia adentro del hueso.
vec3 gradiente(vec3 coordenada) {
    vec3 paso = 1.0 / vec3(textureSize(volume, 0));
    vec3 dx = vec3(paso.x, 0.0, 0.0);
    vec3 dy = vec3(0.0, paso.y, 0.0);
    vec3 dz = vec3(0.0, 0.0, paso.z);
    return 0.5 * vec3(
        texture(volume, coordenada + dx).r - texture(volume, coordenada - dx).r,
        texture(volume, coordenada + dy).r - texture(volume, coordenada - dy).r,
        texture(volume, coordenada + dz).r - texture(volume, coordenada - dz).r);
}

// Phong con una luz direccional que acompana a la camara, arriba y a su izquierda,
// para que las caras que miran al ojo no queden todas iguales. El gradiente no
// distingue adentro de afuera, asi que el termino difuso usa el valor absoluto.
const float KA = 0.35;
const float KD = 0.65;
const float KS = 0.35;
const float BRILLO = 24.0;

vec3 phong(vec3 color, vec3 normal, vec3 hacia_luz, vec3 hacia_camara) {
    float difuso = abs(dot(normal, hacia_luz));
    vec3 reflejada = reflect(-hacia_luz, normal);
    float especular = pow(max(dot(reflejada, hacia_camara), 0.0), BRILLO);
    return color * (KA + KD * difuso) + KS * especular;
}

// Donde el volumen casi no cambia, el gradiente es ruido y no sirve como normal.
// Por eso la iluminacion se mezcla segun el largo del gradiente (en HU por voxel).
const float GRADIENTE_MINIMO = 30.0;
const float GRADIENTE_PLENO = 150.0;

// El maximo de un hueso se repite en la pared de adelante y en la de atras. Sin
// una tolerancia, el redondeo de la interpolacion haria saltar la profundidad de
// una pared a la otra entre pixeles vecinos.
const float TOLERANCIA_PROFUNDIDAD = 100.0;

// Escala de color del MIP con profundidad: amarillo cerca, azul lejos.
vec3 color_profundidad(float s) {
    vec3 cerca = vec3(1.00, 0.85, 0.25);
    vec3 medio = vec3(0.95, 0.30, 0.55);
    vec3 lejos = vec3(0.25, 0.40, 1.00);
    return s < 0.5 ? mix(cerca, medio, 2.0 * s) : mix(medio, lejos, 2.0 * s - 1.0);
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
    vec3 hacia_luz = normalize(-forward + 0.8 * up - 0.6 * right);

    vec3 fondo = vec3(0.05, 0.06, 0.09);

    float t_cerca, t_lejos;
    if (!intersecta_caja(camera_position, ray_direction, t_cerca, t_lejos)) {
        out_color = vec4(fondo, 1.0);
        return;
    }
    t_cerca = max(t_cerca, 0.0);

    // acumuladores del recorrido.
    float maximo = HU_MIN;
    float t_maximo = t_cerca;
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
        if (hu > maximo + TOLERANCIA_PROFUNDIDAD) t_maximo = t;  // para el MIP con profundidad.

        if (mode == 0) {
            maximo = max(maximo, hu);
        } else if (mode == 1) {
            suma += hu;
            conteo += 1;
        } else {
            vec4 muestra = transferencia(hu);
            float valor_etiqueta = etiqueta(coordenada);
            if (valor_etiqueta > 0.0) {
                muestra = vec4(color_etiqueta(valor_etiqueta), ALFA_MASCARA);
            }
            if (iluminacion == 1 && muestra.a > 0.0) {
                // el gradiente pasa de HU por voxel a HU por unidad de mundo, porque
                // los voxeles de una tomografia no miden lo mismo en los tres ejes.
                vec3 voxel_mundo = 2.0 * box_half / vec3(textureSize(volume, 0));
                vec3 g = gradiente(coordenada);
                float peso = smoothstep(GRADIENTE_MINIMO, GRADIENTE_PLENO, length(g));
                vec3 normal = -normalize(g / voxel_mundo + 1e-6);
                muestra.rgb = mix(muestra.rgb, phong(muestra.rgb, normal, hacia_luz, -ray_direction), peso);
            }
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
        if (profundidad_mip == 1) {
            // 0 a una distancia de la camara igual a la del centro menos el lado
            // mayor de la caja, 1 a la del centro mas ese lado.
            float lado = max(max(box_half.x, box_half.y), box_half.z);
            float s = clamp((t_maximo - (radius - lado)) / (2.0 * lado), 0.0, 1.0);
            resultado *= color_profundidad(s);
        }
    } else if (mode == 1) {
        float promedio = conteo > 0 ? suma / float(conteo) : HU_MIN;
        resultado = vec3(normaliza_hu(promedio));
    } else {
        resultado = color_acumulado + (1.0 - alpha_acumulado) * fondo;
    }

    out_color = vec4(resultado, 1.0);
}
