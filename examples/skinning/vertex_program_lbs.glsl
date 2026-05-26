#version 330

// Linear Blend Skinning (LBS)
//
// Cada vertice tiene hasta cuatro huesos que lo influencian (joints) y un
// peso por hueso (weights, suman 1). La matriz de skinning final es la
// combinacion lineal de las matrices de cada hueso ponderada por sus pesos.
// LBS es barato y trivial pero degenera en torsiones fuertes: la suma de
// matrices de rotacion no es una rotacion, lo que produce el clasico efecto
// candy-wrapper (los volumenes se aplastan en articulaciones torcidas).

in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 joints;
in vec4 weights;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 skin_matrices[91];

out vec3 world_position;
out vec3 world_normal;
out vec2 frag_uv;

void main()
{
    mat4 skin_matrix =
          weights.x * skin_matrices[int(joints.x)]
        + weights.y * skin_matrices[int(joints.y)]
        + weights.z * skin_matrices[int(joints.z)]
        + weights.w * skin_matrices[int(joints.w)];

    vec4 skinned_position = skin_matrix * vec4(position, 1.0);
    vec4 world = transform * skinned_position;
    world_position = world.xyz;

    // la matriz de normales se construye sobre skin*transform porque la
    // normal vive en el mismo espacio que la posicion antes de proyectar
    mat3 normal_matrix = transpose(inverse(mat3(transform) * mat3(skin_matrix)));
    world_normal = normalize(normal_matrix * normal);

    frag_uv = uv;
    gl_Position = projection * view * world;
}
