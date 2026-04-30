#version 330

in float view_z;
out vec4 out_color;

uniform float depth_near;
uniform float depth_far;

void main() {
    // Visualización del z-buffer: cada fragmento se colorea según su
    // profundidad lineal en cámara. El rango (depth_near, depth_far) se ajusta
    // dinámicamente al bounding sphere del modelo, así el gradiente cubre toda
    // la superficie y no solo una franja del frustum.
    float t = clamp((view_z - depth_near) / (depth_far - depth_near), 0.0, 1.0);

    // Paleta Gooch (azul frío, amarillo cálido). Lo cercano queda cálido,
    // lo lejano frío.
    vec3 cool = vec3(0.00, 0.00, 0.55);
    vec3 warm = vec3(0.55, 0.55, 0.00);
    out_color = vec4(mix(warm, cool, t), 1.0);
}
