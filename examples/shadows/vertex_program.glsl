#version 330

in vec3 position;
in vec3 normal;
in vec4 color;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 light_transform;

out vec4 fragment_color;
out vec3 fragment_normal;
out vec3 fragment_world_position;
out vec4 position_in_light_space;

void main() {
    vec4 world_position = transform * vec4(position, 1.0);

    // matriz para transformar normales correctamente bajo escalado no
    // uniforme o esquileo.
    mat3 normal_matrix = transpose(inverse(mat3(transform)));

    fragment_normal = normalize(normal_matrix * normal);
    fragment_color = color / 255.0;
    fragment_world_position = vec3(world_position);
    position_in_light_space = light_transform * world_position;

    gl_Position = projection * view * world_position;
}
