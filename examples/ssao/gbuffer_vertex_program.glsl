#version 330

// Primera pasada: en vez de un color, esta pasada escribe la geometría que
// la oclusión necesita. Todo se calcula en espacio de vista, porque ahí la
// cámara está en el origen y las comparaciones de profundidad son directas.

in vec3 position;
in vec3 normal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

out vec3 view_position;
out vec3 view_normal;

void main() {
    vec4 position_in_view = view * transform * vec4(position, 1.0f);
    view_position = position_in_view.xyz;

    // La matriz de normales: la inversa traspuesta de la parte lineal, para
    // que un escalado no uniforme no deje las normales torcidas.
    mat3 normal_matrix = mat3(transpose(inverse(view * transform)));
    view_normal = normal_matrix * normal;

    gl_Position = projection * position_in_view;
}
