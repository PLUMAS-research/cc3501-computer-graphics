#version 330

// Cuarta pasada: la escena iluminada. Es el mismo Phong de `disco_bunny`, con
// una diferencia: el término ambiental se multiplica por el factor de
// oclusión que calcularon las pasadas anteriores.

in vec3 position;
in vec3 normal;
in vec4 color;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 view_position;
out vec3 view_normal;
out vec3 fragment_color;

void main() {
    vec4 position_in_view = view * transform * vec4(position, 1.0f);
    view_position = position_in_view.xyz;
    view_normal = mat3(transpose(inverse(view * transform))) * normal;
    // el cargador entrega el color en 0..255
    fragment_color = color.rgb / 255.0f;
    gl_Position = projection * position_in_view;
}
