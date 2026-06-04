#version 330

// CSG por raymarching sobre funciones de distancia con signo (SDF).
//
// Cada primitiva es una funcion f(p) que devuelve la distancia con signo del
// punto p a su superficie: f(p) < 0 dentro, f(p) > 0 fuera, f(p) = 0 en la
// superficie. Las operaciones booleanas del arbol CSG son operaciones sobre
// estos numeros:
//   union        = min(f_a, f_b)
//   interseccion = max(f_a, f_b)
//   diferencia   = max(f_a, -f_b)
//
// El raymarching avanza por el rayo de la camara en pasos del tamano de la
// distancia mas cercana (sphere tracing) hasta tocar la superficie f(p) ~ 0.

out vec4 out_color;

uniform vec2 resolution;
uniform float camera_yaw;     // orbita horizontal de la camara (radianes).
uniform float camera_pitch;   // orbita vertical de la camara (radianes).
uniform int operation;        // 0 union, 1 interseccion, 2 diferencia.
uniform float separation;     // desplazamiento de la esfera respecto de la caja.

const int MAX_STEPS = 96;
const float MAX_DISTANCE = 20.0;
const float SURFACE_EPSILON = 0.0008;

// Primitiva A: caja centrada en el origen. mat = 0.
float sdf_box(vec3 point, vec3 half_extents) {
    vec3 q = abs(point) - half_extents;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Primitiva B: esfera. mat = 1.
float sdf_sphere(vec3 point, vec3 center, float radius) {
    return length(point - center) - radius;
}

// Escena CSG completa. Devuelve (distancia, material) en un vec2 para saber a
// que primitiva pertenece la superficie tocada y colorearla distinto.
vec2 scene(vec3 point) {
    float distance_box = sdf_box(point, vec3(0.75));
    float distance_sphere = sdf_sphere(point, vec3(separation, 0.0, 0.0), 0.95);

    if (operation == 0) {
        // union: gana la primitiva mas cercana.
        return distance_box < distance_sphere ? vec2(distance_box, 0.0)
                                              : vec2(distance_sphere, 1.0);
    } else if (operation == 1) {
        // interseccion: gana la primitiva mas lejana.
        return distance_box > distance_sphere ? vec2(distance_box, 0.0)
                                              : vec2(distance_sphere, 1.0);
    }
    // diferencia caja - esfera = max(f_box, -f_sphere).
    return distance_box > -distance_sphere ? vec2(distance_box, 0.0)
                                           : vec2(-distance_sphere, 1.0);
}

// Normal por gradiente numerico del campo de distancia.
vec3 estimate_normal(vec3 point) {
    vec2 epsilon = vec2(0.0015, 0.0);
    return normalize(vec3(
        scene(point + epsilon.xyy).x - scene(point - epsilon.xyy).x,
        scene(point + epsilon.yxy).x - scene(point - epsilon.yxy).x,
        scene(point + epsilon.yyx).x - scene(point - epsilon.yyx).x
    ));
}

void main() {
    // pixel a coordenada de pantalla centrada, corregida por aspecto.
    vec2 screen = (gl_FragCoord.xy - 0.5 * resolution) / resolution.y;

    // camara que orbita el origen a radio fijo segun yaw y pitch.
    float radius = 4.0;
    vec3 camera_position = radius * vec3(
        cos(camera_pitch) * sin(camera_yaw),
        sin(camera_pitch),
        cos(camera_pitch) * cos(camera_yaw)
    );
    vec3 forward = normalize(-camera_position);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 ray_direction = normalize(forward + screen.x * right + screen.y * up);

    // sphere tracing.
    float traveled = 0.0;
    float material = -1.0;
    for (int step_index = 0; step_index < MAX_STEPS; step_index++) {
        vec3 sample_point = camera_position + traveled * ray_direction;
        vec2 result = scene(sample_point);
        if (result.x < SURFACE_EPSILON) {
            material = result.y;
            break;
        }
        traveled += result.x;
        if (traveled > MAX_DISTANCE) {
            break;
        }
    }

    // fondo plano cuando el rayo no toca el solido.
    vec3 background = vec3(0.10, 0.11, 0.16);
    if (material < 0.0) {
        out_color = vec4(background, 1.0);
        return;
    }

    vec3 hit_point = camera_position + traveled * ray_direction;
    vec3 normal = estimate_normal(hit_point);
    vec3 light_direction = normalize(vec3(0.6, 0.8, 0.5));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float ambient = 0.25;

    // dos tintes para distinguir que primitiva aporta cada parte de la superficie.
    vec3 tint_box = vec3(0.95, 0.55, 0.30);
    vec3 tint_sphere = vec3(0.35, 0.65, 0.95);
    vec3 base = material < 0.5 ? tint_box : tint_sphere;

    vec3 shaded = base * (ambient + diffuse);
    out_color = vec4(pow(shaded, vec3(0.4545)), 1.0);  // correccion gamma.
}
