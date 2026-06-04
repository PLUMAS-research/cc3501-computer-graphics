#version 330

// Metaballs por raymarching: varias esferas SDF unidas con union suave.
//
// La union dura del CSG es min(f_a, f_b): produce un borde anguloso donde dos
// primitivas se tocan. La union SUAVE reemplaza ese min por una mezcla
// continua, la smooth-min:
//
//   smin(a, b, k) = mezcla de a y b que se redondea en una franja de ancho k.
//
// Con k pequeno se recupera el min duro (las esferas se tocan con un borde
// marcado). Con k grande las esferas se "derriten" entre si y forman una sola
// gota. Esa fusion continua es lo que da el aspecto de lampara de lava.

#define N_BLOBS 7

out vec4 out_color;

uniform vec2 resolution;
uniform float camera_yaw;
uniform float camera_pitch;
uniform float smooth_k;        // ancho de la franja de fusion.
uniform vec4 blobs[N_BLOBS];   // xyz = centro, w = radio.

const int MAX_STEPS = 110;
const float MAX_DISTANCE = 24.0;
const float SURFACE_EPSILON = 0.001;

// smooth-min polinomial (Inigo Quilez). Devuelve la distancia fusionada.
float smooth_min(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// campo de distancia de toda la escena: union suave de las esferas.
float scene(vec3 point) {
    float distance = 1e5;
    for (int i = 0; i < N_BLOBS; i++) {
        float sphere = length(point - blobs[i].xyz) - blobs[i].w;
        distance = smooth_min(distance, sphere, smooth_k);
    }
    return distance;
}

vec3 estimate_normal(vec3 point) {
    vec2 epsilon = vec2(0.0015, 0.0);
    return normalize(vec3(
        scene(point + epsilon.xyy) - scene(point - epsilon.xyy),
        scene(point + epsilon.yxy) - scene(point - epsilon.yxy),
        scene(point + epsilon.yyx) - scene(point - epsilon.yyx)
    ));
}

// gradiente vertical calido para el fondo y para tintar las gotas por altura.
vec3 warm_gradient(float height) {
    vec3 bottom = vec3(0.95, 0.35, 0.10);  // naranja encendido abajo.
    vec3 top = vec3(0.85, 0.10, 0.45);     // magenta arriba.
    return mix(bottom, top, clamp(height * 0.5 + 0.5, 0.0, 1.0));
}

void main() {
    vec2 screen = (gl_FragCoord.xy - 0.5 * resolution) / resolution.y;

    float radius = 6.0;
    vec3 camera_position = radius * vec3(
        cos(camera_pitch) * sin(camera_yaw),
        sin(camera_pitch),
        cos(camera_pitch) * cos(camera_yaw)
    );
    vec3 forward = normalize(-camera_position);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 ray_direction = normalize(forward + screen.x * right + screen.y * up);

    float traveled = 0.0;
    bool hit = false;
    for (int step_index = 0; step_index < MAX_STEPS; step_index++) {
        vec3 sample_point = camera_position + traveled * ray_direction;
        float distance = scene(sample_point);
        if (distance < SURFACE_EPSILON) {
            hit = true;
            break;
        }
        traveled += distance;
        if (traveled > MAX_DISTANCE) {
            break;
        }
    }

    // fondo: gradiente calido oscuro de la lampara.
    if (!hit) {
        vec3 background = warm_gradient(ray_direction.y) * 0.12;
        out_color = vec4(background, 1.0);
        return;
    }

    vec3 hit_point = camera_position + traveled * ray_direction;
    vec3 normal = estimate_normal(hit_point);
    vec3 light_direction = normalize(vec3(0.4, 0.9, 0.6));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float ambient = 0.25;

    // tinte por altura del punto + realce de borde (fresnel) para el brillo.
    vec3 base = warm_gradient(hit_point.y);
    float rim = pow(1.0 - max(dot(normal, -ray_direction), 0.0), 2.5);
    vec3 shaded = base * (ambient + diffuse) + rim * vec3(1.0, 0.7, 0.4);

    out_color = vec4(pow(shaded, vec3(0.4545)), 1.0);  // correccion gamma.
}
