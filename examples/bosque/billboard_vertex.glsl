#version 330

in vec3 base_position;
in vec2 local_offset;
in vec2 texcoord;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_position;

out vec2 frag_texcoord;

void main()
{
    // Billboard cilindrico: el quad rota alrededor del eje Y para mirar a la
    // camara, conservando su altura en el mundo.
    vec3 to_camera = camera_position - base_position;
    to_camera.y = 0.0;
    float largo = length(to_camera);
    if (largo < 0.0001) {
        to_camera = vec3(0.0, 0.0, 1.0);
    } else {
        to_camera = to_camera / largo;
    }

    vec3 right = vec3(-to_camera.z, 0.0, to_camera.x);
    vec3 up = vec3(0.0, 1.0, 0.0);

    vec3 world_pos = base_position + right * local_offset.x + up * local_offset.y;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    frag_texcoord = texcoord;
}
