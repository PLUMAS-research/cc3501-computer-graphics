#version 330

// Linear Blend Skinning sobre un esqueleto de tres huesos.
//
// Cada vertice trae hasta cuatro influencias (joints) con sus pesos
// (weights, suman 1). La CPU ya calculo, por hueso, la matriz de skinning
//   skin_matrices[j] = M_j(actual) * inverse_bind(j),
// que lleva el vertice desde el espacio de objeto en reposo al espacio de
// objeto deformado pasando por el espacio local del hueso. El vertex shader
// solo mezcla esas matrices: combinacion lineal ponderada por los pesos.
//
// El color del vertice tambien se mezcla con los pesos, usando un color por
// hueso (joint_colors). Asi se ve directamente que hueso domina cada region
// de la malla y como las bandas de mezcla se deforman con el esqueleto.

in vec3 position;
in vec3 normal;
in vec4 joints;
in vec4 weights;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 skin_matrices[3];
uniform vec3 joint_colors[3];

out vec3 world_normal;
out vec3 weight_color;

void main()
{
    mat4 skin_matrix =
          weights.x * skin_matrices[int(joints.x)]
        + weights.y * skin_matrices[int(joints.y)]
        + weights.z * skin_matrices[int(joints.z)]
        + weights.w * skin_matrices[int(joints.w)];

    vec4 skinned_position = skin_matrix * vec4(position, 1.0);

    // la normal vive en el mismo espacio que la posicion, asi que se
    // transforma con la matriz de normales de la propia matriz de skinning
    mat3 normal_matrix = transpose(inverse(mat3(skin_matrix)));
    world_normal = normalize(normal_matrix * normal);

    weight_color =
          weights.x * joint_colors[int(joints.x)]
        + weights.y * joint_colors[int(joints.y)]
        + weights.z * joint_colors[int(joints.z)]
        + weights.w * joint_colors[int(joints.w)];

    gl_Position = projection * view * skinned_position;
}
