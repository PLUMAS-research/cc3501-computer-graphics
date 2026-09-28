#version 330

in vec3 position;
in vec3 normal;
in vec4 color;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
// La vista de la cámara del jugador. Coincide con view salvo en la vista
// desde arriba, donde las cascadas siguen siendo las del jugador.
uniform mat4 camera_view;

// Una matriz por cascada del sol: ndc_to_texture * projection_cascada *
// view_sol. Las tres comparten la vista del sol y difieren en la proyección.
uniform mat4 shadow_transform_0;
uniform mat4 shadow_transform_1;
uniform mat4 shadow_transform_2;

// Una matriz por foco: ndc_to_texture * projection_foco * view_foco.
uniform mat4 spot_shadow_transform_0;
uniform mat4 spot_shadow_transform_1;
uniform mat4 spot_shadow_transform_2;

out vec3 fragment_color;
out vec3 fragment_normal;
out vec3 fragment_world_position;
out float fragment_view_depth;
out vec4 position_in_shadow_map_0;
out vec4 position_in_shadow_map_1;
out vec4 position_in_shadow_map_2;
out vec4 position_in_spot_map_0;
out vec4 position_in_spot_map_1;
out vec4 position_in_spot_map_2;

void main() {
    vec4 world_position = transform * vec4(position, 1.0);

    fragment_color = color.rgb / 255.0;
    fragment_normal = normalize(transpose(inverse(mat3(transform))) * normal);
    fragment_world_position = world_position.xyz;
    // distancia a la cámara medida sobre su eje: decide qué cascada consultar
    fragment_view_depth = -(camera_view * world_position).z;

    position_in_shadow_map_0 = shadow_transform_0 * world_position;
    position_in_shadow_map_1 = shadow_transform_1 * world_position;
    position_in_shadow_map_2 = shadow_transform_2 * world_position;
    position_in_spot_map_0 = spot_shadow_transform_0 * world_position;
    position_in_spot_map_1 = spot_shadow_transform_1 * world_position;
    position_in_spot_map_2 = spot_shadow_transform_2 * world_position;

    gl_Position = projection * view * world_position;
}
