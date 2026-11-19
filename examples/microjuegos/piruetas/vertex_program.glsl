#version 330

in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 joints;   // los cuatro huesos que mueven al vértice
in vec4 weights;  // el peso de cada uno; suman 1

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
// el largo de un arreglo uniform tiene que ser constante: el esqueleto del
// zorzal tiene 91 huesos
uniform mat4 huesos[91];

out vec3 normal_mundo;
out vec2 uv_textura;

void main()
{
    // linear blend skinning: la matriz del vértice es el promedio de las
    // matrices de sus huesos, ponderado por los pesos
    mat4 piel = weights.x * huesos[int(joints.x)]
              + weights.y * huesos[int(joints.y)]
              + weights.z * huesos[int(joints.z)]
              + weights.w * huesos[int(joints.w)];
    mat4 modelo = transform * piel;

    normal_mundo = mat3(transpose(inverse(modelo))) * normal;
    uv_textura = uv;
    gl_Position = projection * view * modelo * vec4(position, 1.0);
}
