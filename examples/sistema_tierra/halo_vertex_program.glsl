#version 330
in vec2 position;
in vec2 uv;

uniform vec3 halo_world_position;
uniform float halo_size;
uniform vec3 camera_right;
uniform vec3 camera_up;
uniform mat4 view;
uniform mat4 projection;

out vec2 frag_uv;

// Billboard alineado a la cámara: el quad vive en mundo, en la
// posición del Sol, y se orienta para enfrentarse a la cámara.
// Como recibe view y projection estándar, el depth test funciona:
// la Tierra (más cercana a la cámara) ocluye al halo en los píxeles
// donde se solapan en pantalla, así el halo queda dibujado solo
// alrededor de la silueta de la Tierra, no encima de ella
void main()
{
    vec3 world_pos = halo_world_position
        + camera_right * position.x * halo_size
        + camera_up * position.y * halo_size;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    frag_uv = uv;
}
