#version 330

// Ray casting de un volumen de microscopia de fluorescencia con dos canales.
//
// La textura 3D guarda un canal en r y el otro en g, como intensidades en [0,1].
// Se recorre igual que en raycasting_volumen: un rayo por pixel, recortado a la
// caja del volumen por el metodo de las losas y muestreado a paso fijo. Cambia lo
// que se hace con cada muestra:
//
//   modo 0  MIP por canal : el maximo de cada canal por separado. Cada maximo se
//                           pinta con el color de su canal y los dos se suman.
//   modo 2  mapa          : el embrion desplegado como un planisferio. Cada pixel
//                           es una longitud y una latitud; el rayo parte del centro
//                           de la esfera del embrion hacia afuera y se queda con el
//                           maximo de cada canal dentro de la cascara.
//   modo 1  composicion   : emision y absorcion front-to-back. La fluorescencia
//                           es luz emitida, asi que la funcion de transferencia
//                           es una rampa: la opacidad crece con el brillo y el
//                           color es el del canal.
//
// Igual que en raycasting_volumen, la composicion puede sombrearse con Phong usando
// el gradiente del brillo como normal, y el MIP puede pintarse segun la distancia a
// la que estaba el maximo, en vez del color de cada canal.

out vec4 out_color;

uniform vec2 viewport_size;      // tamano de la ventana (pixeles).
uniform float camera_yaw;        // orbita horizontal de la camara (radianes).
uniform float camera_pitch;      // orbita vertical de la camara (radianes).
uniform float distancia_camara;  // radio de la orbita.
uniform int mode;                // 0 MIP por canal, 1 composicion.
uniform float step_size;         // paso de muestreo en unidades de mundo.
uniform float voxel_size;        // tamano de un voxel en unidades de mundo.
uniform vec3 box_half;           // medios lados: la caja es [-box_half, box_half].
uniform sampler3D volume;        // canal 0 en r, canal 1 en g.
uniform vec2 minimo_canal;       // intensidad que se muestra con brillo 0, por canal.
uniform vec2 maximo_canal;       // intensidad que se muestra con brillo 1, por canal.
uniform vec2 canal_visible;      // 1 si el canal se dibuja, 0 si no.
uniform vec3 color_canal_0;
uniform vec3 color_canal_1;
uniform float densidad;          // opacidad de una muestra de brillo 1, por voxel.
uniform int iluminacion;         // 1 sombrea la composicion con Phong.
uniform int profundidad_mip;     // 1 colorea el MIP segun la distancia del maximo.
uniform vec3 centro_esfera;      // esfera ajustada a las celulas, en coordenadas de la caja.
uniform float radio_esfera;

const int MAX_STEPS = 4096;

bool intersecta_caja(vec3 origen, vec3 direccion, out float t_cerca, out float t_lejos) {
    vec3 reciproco = 1.0 / direccion;
    vec3 t0 = (-box_half - origen) * reciproco;
    vec3 t1 = ( box_half - origen) * reciproco;
    vec3 entrada = min(t0, t1);
    vec3 salida  = max(t0, t1);
    t_cerca = max(max(entrada.x, entrada.y), entrada.z);
    t_lejos = min(min(salida.x, salida.y), salida.z);
    return t_lejos >= max(t_cerca, 0.0);
}

// Brillo de cada canal en [0,1]: la ventana radiologica, aplicada a cada canal.
// Lo que queda bajo el minimo (el ruido de fondo) no aporta nada.
vec2 brillo(vec3 coordenada) {
    vec2 intensidad = texture(volume, coordenada).rg;
    vec2 normalizada = (intensidad - minimo_canal) / (maximo_canal - minimo_canal);
    return clamp(normalizada, 0.0, 1.0) * canal_visible;
}

// Gradiente de la suma de los brillos por diferencias centrales, en brillo por
// voxel. Apunta hacia donde la senal crece.
vec3 gradiente(vec3 coordenada) {
    vec3 paso = 1.0 / vec3(textureSize(volume, 0));
    vec3 dx = vec3(paso.x, 0.0, 0.0);
    vec3 dy = vec3(0.0, paso.y, 0.0);
    vec3 dz = vec3(0.0, 0.0, paso.z);
    vec2 uno = vec2(1.0);
    return 0.5 * vec3(
        dot(brillo(coordenada + dx) - brillo(coordenada - dx), uno),
        dot(brillo(coordenada + dy) - brillo(coordenada - dy), uno),
        dot(brillo(coordenada + dz) - brillo(coordenada - dz), uno));
}

// Phong con una luz que acompana a la camara, arriba y a su izquierda. El termino
// difuso usa el valor absoluto, porque el gradiente no distingue adentro de afuera.
const float KA = 0.35;
const float KD = 0.65;
const float KS = 0.35;
const float BRILLO_ESPECULAR = 24.0;

vec3 phong(vec3 color, vec3 normal, vec3 hacia_luz, vec3 hacia_camara) {
    float difuso = abs(dot(normal, hacia_luz));
    vec3 reflejada = reflect(-hacia_luz, normal);
    float especular = pow(max(dot(reflejada, hacia_camara), 0.0), BRILLO_ESPECULAR);
    return color * (KA + KD * difuso) + KS * especular;
}

// Donde la senal casi no cambia, el gradiente es ruido y no sirve como normal.
const float GRADIENTE_MINIMO = 0.02;
const float GRADIENTE_PLENO = 0.15;

// Un maximo puede repetirse en la cara de adelante y en la de atras del embrion.
// La tolerancia evita que la profundidad salte entre las dos de un pixel a otro.
const float TOLERANCIA_PROFUNDIDAD = 0.05;

// La cascara que recorre el mapa, como fraccion del radio de la esfera.
const float CASCARA_INTERIOR = 0.90;
const float CASCARA_EXTERIOR = 1.10;
const float PI = 3.14159265;

// Proyeccion equirectangular: el ancho de la ventana cubre 360 grados de longitud y
// un alto de la mitad del ancho cubre 180 grados de latitud, de modo que un grado
// mide lo mismo en las dos direcciones. Longitud y latitud 0 miran hacia la camara,
// asi que arrastrar el mouse gira el embrion bajo el mapa.
vec3 mapa(vec3 forward, vec3 right, vec3 up) {
    float alto_mapa = 0.5 * viewport_size.x;
    vec2 pixel = gl_FragCoord.xy - vec2(0.0, 0.5 * (viewport_size.y - alto_mapa));
    if (pixel.y < 0.0 || pixel.y > alto_mapa) return vec3(0.0);

    float longitud = (pixel.x / viewport_size.x - 0.5) * 2.0 * PI;
    float latitud = (pixel.y / alto_mapa - 0.5) * PI;
    vec3 direccion = cos(latitud) * (sin(longitud) * right - cos(longitud) * forward)
                   + sin(latitud) * up;

    vec2 maximo = vec2(0.0);
    for (float r = CASCARA_INTERIOR * radio_esfera; r <= CASCARA_EXTERIOR * radio_esfera; r += step_size) {
        vec3 coordenada = 0.5 * ((centro_esfera + r * direccion) / box_half + 1.0);
        if (any(lessThan(coordenada, vec3(0.0))) || any(greaterThan(coordenada, vec3(1.0)))) continue;
        maximo = max(maximo, brillo(coordenada));
    }
    vec3 color = min(maximo.x * color_canal_0 + maximo.y * color_canal_1, vec3(1.0));

    // reticula cada 30 grados: en el mapa las celdas son iguales, pero en la
    // esfera las de los polos son mucho mas chicas que las del ecuador.
    float paso_reticula = viewport_size.x / 12.0;
    vec2 distancia = abs(fract(pixel / paso_reticula + 0.5) - 0.5) * paso_reticula;
    if (min(distancia.x, distancia.y) < 0.75) color = mix(color, vec3(0.45), 0.6);
    return color;
}

// Escala de color del MIP con profundidad: amarillo cerca, azul lejos.
vec3 color_profundidad(float s) {
    vec3 cerca = vec3(1.00, 0.85, 0.25);
    vec3 medio = vec3(0.95, 0.30, 0.55);
    vec3 lejos = vec3(0.25, 0.40, 1.00);
    return s < 0.5 ? mix(cerca, medio, 2.0 * s) : mix(medio, lejos, 2.0 * s - 1.0);
}

void main() {
    vec2 screen = (gl_FragCoord.xy - 0.5 * viewport_size) / viewport_size.y;

    vec3 camera_position = distancia_camara * vec3(
        cos(camera_pitch) * sin(camera_yaw),
        sin(camera_pitch),
        cos(camera_pitch) * cos(camera_yaw)
    );
    vec3 forward = normalize(-camera_position);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 ray_direction = normalize(forward + screen.x * right + screen.y * up);
    vec3 hacia_luz = normalize(-forward + 0.8 * up - 0.6 * right);

    if (mode == 2) {
        out_color = vec4(mapa(forward, right, up), 1.0);
        return;
    }

    vec3 fondo = vec3(0.0);  // la fluorescencia se mira sobre negro.

    float t_cerca, t_lejos;
    if (!intersecta_caja(camera_position, ray_direction, t_cerca, t_lejos)) {
        out_color = vec4(fondo, 1.0);
        return;
    }
    t_cerca = max(t_cerca, 0.0);

    vec2 maximo = vec2(0.0);
    float maximo_total = 0.0;
    float t_maximo = t_cerca;
    vec3 color_acumulado = vec3(0.0);
    float alpha_acumulado = 0.0;
    float exponente = step_size / voxel_size;  // correccion de opacidad por el paso.

    float t = t_cerca;
    for (int i = 0; i < MAX_STEPS; i++) {
        if (t > t_lejos) break;
        vec3 punto = camera_position + t * ray_direction;
        vec3 coordenada = 0.5 * (punto / box_half + 1.0);
        vec2 b = brillo(coordenada);

        if (mode == 0) {
            maximo = max(maximo, b);
            float total = max(b.x, b.y);
            if (total > maximo_total + TOLERANCIA_PROFUNDIDAD) t_maximo = t;
            maximo_total = max(maximo_total, total);
        } else {
            float suma = b.x + b.y;
            if (suma > 0.0) {
                vec3 emision = (b.x * color_canal_0 + b.y * color_canal_1) / suma;
                if (iluminacion == 1) {
                    vec3 voxel_mundo = 2.0 * box_half / vec3(textureSize(volume, 0));
                    vec3 g = gradiente(coordenada);
                    float peso = smoothstep(GRADIENTE_MINIMO, GRADIENTE_PLENO, length(g));
                    vec3 normal = -normalize(g / voxel_mundo + 1e-6);
                    emision = mix(emision, phong(emision, normal, hacia_luz, -ray_direction), peso);
                }
                float alpha_referencia = clamp(densidad * suma, 0.0, 1.0);
                float alpha = 1.0 - pow(1.0 - alpha_referencia, exponente);
                color_acumulado += (1.0 - alpha_acumulado) * emision * alpha;
                alpha_acumulado += (1.0 - alpha_acumulado) * alpha;
                if (alpha_acumulado > 0.99) break;
            }
        }
        t += step_size;
    }

    vec3 resultado;
    if (mode == 0 && profundidad_mip == 1) {
        // 0 a la distancia del centro menos el lado mayor de la caja, 1 a la del centro mas ese lado.
        float lado = max(max(box_half.x, box_half.y), box_half.z);
        float s = clamp((t_maximo - (distancia_camara - lado)) / (2.0 * lado), 0.0, 1.0);
        resultado = maximo_total * color_profundidad(s);
    } else if (mode == 0) {
        resultado = maximo.x * color_canal_0 + maximo.y * color_canal_1;
    } else {
        resultado = color_acumulado + (1.0 - alpha_acumulado) * fondo;
    }
    out_color = vec4(min(resultado, vec3(1.0)), 1.0);
}
