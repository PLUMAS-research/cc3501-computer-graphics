#version 330

// Dual Quaternion Skinning (DQS)
//
// En lugar de mezclar matrices (que pierde unitaridad bajo combinacion lineal)
// representamos cada hueso como un dual quaternion: la parte real codifica la
// rotacion como cuaternion unitario y la parte dual codifica la translacion.
// Mezclamos los dual quaternions con suma ponderada y renormalizamos al final,
// lo que produce siempre una transformacion rigida.
//
// DQS preserva volumenes en torsiones donde LBS los aplasta. Tiene tambien
// un costo: la mezcla es esferica (interpolacion sobre la esfera de
// cuaterniones) y deja un patron de "joint bulging" caracteristico cuando los
// huesos se cruzan en angulos grandes. Mas costoso que LBS pero todavia barato
// para tiempo real.

in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 joints;
in vec4 weights;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 dual_quat_real[91];
uniform vec4 dual_quat_dual[91];

out vec3 world_position;
out vec3 world_normal;
out vec2 frag_uv;

vec3 quaternion_rotate(vec4 q, vec3 v)
{
    // formula que evita construir la matriz 3x3 de rotacion:
    // v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    vec3 q_xyz = q.xyz;
    return v + 2.0 * cross(q_xyz, cross(q_xyz, v) + q.w * v);
}

vec3 dual_quat_transform_position(vec4 q_real, vec4 q_dual, vec3 p)
{
    vec3 rotated = quaternion_rotate(q_real, p);
    // el componente de translacion de un dual quaternion (q_r, q_d) es
    // t = 2 * (q_r.w * q_d.xyz - q_d.w * q_r.xyz + cross(q_r.xyz, q_d.xyz))
    vec3 translation =
        2.0 * (q_real.w * q_dual.xyz
             - q_dual.w * q_real.xyz
             + cross(q_real.xyz, q_dual.xyz));
    return rotated + translation;
}

void main()
{
    vec4 blended_real = vec4(0.0);
    vec4 blended_dual = vec4(0.0);

    // usamos el primer cuaternion como referencia de hemisferio. Si otro
    // cuaternion apunta al hemisferio opuesto (producto interno negativo)
    // lo invertimos antes de sumar. Sin este paso la suma ponderada puede
    // dar un cuaternion casi cero y la normalizacion explota
    vec4 reference_real = dual_quat_real[int(joints.x)];

    for (int influence_index = 0; influence_index < 4; ++influence_index)
    {
        float weight = weights[influence_index];
        if (weight <= 0.0) continue;

        int joint_index = int(joints[influence_index]);
        vec4 q_real = dual_quat_real[joint_index];
        vec4 q_dual = dual_quat_dual[joint_index];

        float hemisphere_sign = (dot(q_real, reference_real) < 0.0) ? -1.0 : 1.0;
        blended_real += hemisphere_sign * weight * q_real;
        blended_dual += hemisphere_sign * weight * q_dual;
    }

    float real_length = length(blended_real);
    if (real_length > 1e-6)
    {
        blended_real /= real_length;
        blended_dual /= real_length;
    }

    vec3 skinned_position = dual_quat_transform_position(blended_real, blended_dual, position);
    vec3 skinned_normal = quaternion_rotate(blended_real, normal);

    vec4 world = transform * vec4(skinned_position, 1.0);
    world_position = world.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(transform)));
    world_normal = normalize(normal_matrix * skinned_normal);

    frag_uv = uv;
    gl_Position = projection * view * world;
}
