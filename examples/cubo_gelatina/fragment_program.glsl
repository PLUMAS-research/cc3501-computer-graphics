#version 330

// Cubo de gelatina: un solido SDF (caja redondeada) dibujado con apariencia
// acuosa y semitransparente. La transparencia no es solo alpha: el rayo entra
// al cubo, se refracta, viaja por dentro acumulando absorcion de color
// (ley de Beer-Lambert: mientras mas espeso, mas verde se pone) y sale por la
// cara de atras hacia el ambiente. Encima se mezcla un reflejo segun el angulo
// (Fresnel: los bordes reflejan mas, el centro deja ver a traves).
//
// El temblor de gelatina es una deformacion del dominio: antes de evaluar la
// caja se desplaza el punto con una suma de senos que depende del tiempo. Eso
// hace ondular la superficie sin cambiar la primitiva.

out vec4 out_color;

uniform vec2 resolution;
uniform float camera_yaw;
uniform float camera_pitch;
uniform float tiempo;
uniform float wobble;          // amplitud del temblor de gelatina.
uniform float indice_refraccion;  // 1.0 = aire (invisible), ~1.33 agua, ~1.5 vidrio.

const int MAX_STEPS = 96;
const float MAX_DISTANCE = 30.0;
const float SURFACE_EPSILON = 0.0015;
const float PISO_Y = -1.55;
const vec3 SUN_DIRECTION = vec3(0.5, 0.78, 0.38);
// absorcion por canal: el rojo se apaga rapido, el verde casi no, asi el
// interior espeso tira a verde-cian de gelatina. El espesor manda: los bordes
// finos quedan casi transparentes y el cuerpo grueso toma color.
const vec3 ABSORCION = vec3(2.3, 0.5, 1.25);

// SDF de una caja con esquinas redondeadas (radio rounding).
float sdf_caja_redondeada(vec3 point, vec3 half_extents, float rounding) {
    vec3 q = abs(point) - half_extents;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - rounding;
}

// campo de distancia del cubo, con la deformacion de gelatina aplicada.
float scene(vec3 point) {
    vec3 deformado = point + wobble * sin(point.yzx * 3.0 + tiempo * 2.5);
    // factor < 1 para compensar que la deformacion del dominio rompe la metrica
    // exacta de distancia y el sphere tracing podria pasarse de largo.
    return sdf_caja_redondeada(deformado, vec3(0.95), 0.18) * 0.6;
}

vec3 estimate_normal(vec3 point) {
    vec2 epsilon = vec2(0.002, 0.0);
    return normalize(vec3(
        scene(point + epsilon.xyy) - scene(point - epsilon.xyy),
        scene(point + epsilon.yxy) - scene(point - epsilon.yxy),
        scene(point + epsilon.yyx) - scene(point - epsilon.yyx)
    ));
}

// cielo: gradiente azulado con un sol para los brillos especulares.
vec3 cielo(vec3 ray_direction) {
    vec3 color = mix(vec3(0.55, 0.65, 0.75), vec3(0.20, 0.42, 0.62),
                     clamp(ray_direction.y * 0.5 + 0.5, 0.0, 1.0));
    float sol = pow(max(dot(ray_direction, normalize(SUN_DIRECTION)), 0.0), 220.0);
    return color + sol * vec3(1.0, 0.95, 0.8);
}

// ambiente que ve un rayo: piso a cuadros si apunta hacia abajo, si no, cielo.
// Sirve para el rayo primario y para los rayos refractado y reflejado.
vec3 ambiente(vec3 origin, vec3 ray_direction) {
    if (ray_direction.y < -0.001) {
        float t = (PISO_Y - origin.y) / ray_direction.y;
        if (t > 0.0 && t < 60.0) {
            vec3 hit = origin + t * ray_direction;
            float tablero = mod(floor(hit.x * 0.7) + floor(hit.z * 0.7), 2.0);
            vec3 piso = mix(vec3(0.14, 0.15, 0.19), vec3(0.32, 0.34, 0.40), tablero);
            float niebla = exp(-t * 0.06);
            return mix(cielo(ray_direction), piso, niebla);
        }
    }
    return cielo(ray_direction);
}

// avanza por el rayo hasta tocar el cubo. Devuelve la distancia recorrida o -1.
float raymarch(vec3 origin, vec3 ray_direction) {
    float traveled = 0.0;
    for (int step_index = 0; step_index < MAX_STEPS; step_index++) {
        float distance = scene(origin + traveled * ray_direction);
        if (distance < SURFACE_EPSILON) {
            return traveled;
        }
        traveled += distance;
        if (traveled > MAX_DISTANCE) {
            break;
        }
    }
    return -1.0;
}

// espesor del cubo en la direccion refractada: avanza por dentro (donde el
// campo es negativo) hasta volver a salir por la cara de atras. El paso minimo
// garantiza progreso aun en la entrada, donde el campo vale casi cero y la
// condicion de salida se cumpliria de inmediato dejando un espesor nulo.
float espesor_interior(vec3 origin, vec3 ray_direction) {
    float traveled = 0.0;
    for (int step_index = 0; step_index < 80; step_index++) {
        float distance = scene(origin + traveled * ray_direction);
        if (distance > SURFACE_EPSILON) {
            break;  // ya salio del cubo.
        }
        traveled += max(abs(distance), 0.01);
    }
    return traveled;
}

void main() {
    vec2 screen = (gl_FragCoord.xy - 0.5 * resolution) / resolution.y;

    float radius = 4.5;
    vec3 camera_position = radius * vec3(
        cos(camera_pitch) * sin(camera_yaw),
        sin(camera_pitch),
        cos(camera_pitch) * cos(camera_yaw)
    );
    vec3 forward = normalize(-camera_position);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 ray_direction = normalize(forward + screen.x * right + screen.y * up);

    float traveled = raymarch(camera_position, ray_direction);

    // el rayo no toca el cubo: se ve el ambiente directo.
    if (traveled < 0.0) {
        out_color = vec4(pow(ambiente(camera_position, ray_direction), vec3(0.4545)), 1.0);
        return;
    }

    vec3 hit_point = camera_position + traveled * ray_direction;
    vec3 normal = estimate_normal(hit_point);

    // refraccion al entrar: el rayo se dobla y cruza el interior del cubo.
    vec3 refractado = refract(ray_direction, normal, 1.0 / indice_refraccion);
    float espesor = espesor_interior(hit_point, refractado);
    vec3 punto_salida = hit_point + refractado * espesor;
    vec3 normal_salida = estimate_normal(punto_salida);  // apunta hacia afuera.

    // refraccion al salir; si hay reflexion total interna, se refleja adentro.
    vec3 sale = refract(refractado, -normal_salida, indice_refraccion);
    if (dot(sale, sale) < 0.0001) {
        sale = reflect(refractado, normal_salida);
    }
    vec3 absorcion = exp(-espesor * ABSORCION);  // Beer-Lambert: tinte por espesor.
    vec3 color_refractado = ambiente(punto_salida, sale) * absorcion;

    // reflejo del ambiente en la cara frontal.
    vec3 color_reflejado = ambiente(hit_point, reflect(ray_direction, normal));

    // Fresnel (aproximacion de Schlick): mas reflejo en los bordes rasantes.
    float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(normal, -ray_direction), 0.0), 5.0);
    vec3 color = mix(color_refractado, color_reflejado, fresnel);

    // brillo especular del sol sobre la superficie humeda.
    vec3 reflejo = reflect(ray_direction, normal);
    float especular = pow(max(dot(reflejo, normalize(SUN_DIRECTION)), 0.0), 80.0);
    color += especular * vec3(1.0, 0.95, 0.85);

    out_color = vec4(pow(color, vec3(0.4545)), 1.0);  // correccion gamma.
}
