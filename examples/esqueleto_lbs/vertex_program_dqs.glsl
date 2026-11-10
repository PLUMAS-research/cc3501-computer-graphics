#version 330

// Dual Quaternion Skinning (DQS) sobre el mismo esqueleto de tres huesos.
//
// Recibe los mismos atributos que vertex_program.glsl, pero en vez de una
// matriz de skinning por hueso recibe su cuaternion dual: dual_quat_real[j] es
// el cuaternion unitario de la rotacion y dual_quat_dual[j] codifica la
// traslacion. Se mezclan con los pesos y se dividen por el largo de la parte
// real, de modo que el resultado es otra vez una transformacion rigida: el tubo
// no se estrecha al torcer un hueso, como si pasa con la mezcla de matrices.

in vec3 position;
in vec3 normal;
in vec4 joints;
in vec4 weights;
in float franja;   // 1 en las franjas oscuras, 0 en el resto

uniform mat4 view;
uniform mat4 projection;
uniform vec4 dual_quat_real[3];
uniform vec4 dual_quat_dual[3];
uniform vec3 joint_colors[3];

out vec3 world_normal;
out vec3 weight_color;

vec3 quaternion_rotate(vec4 q, vec3 v)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main()
{
    // q y -q son la misma rotacion: cada cuaternion se suma con el signo del
    // lado de la primera influencia, para que la suma no se anule
    vec4 reference_real = dual_quat_real[int(joints.x)];
    vec4 blended_real = vec4(0.0);
    vec4 blended_dual = vec4(0.0);
    for (int influence_index = 0; influence_index < 4; ++influence_index)
    {
        int joint_index = int(joints[influence_index]);
        float weight = weights[influence_index];
        vec4 q_real = dual_quat_real[joint_index];
        float hemisphere_sign = (dot(q_real, reference_real) < 0.0) ? -1.0 : 1.0;
        blended_real += hemisphere_sign * weight * q_real;
        blended_dual += hemisphere_sign * weight * dual_quat_dual[joint_index];
    }
    float real_length = length(blended_real);
    blended_real /= real_length;
    blended_dual /= real_length;

    // traslacion de un cuaternion dual (q_r, q_d):
    // t = 2 * (q_r.w * q_d.xyz - q_d.w * q_r.xyz + cross(q_r.xyz, q_d.xyz))
    vec3 translation = 2.0 * (blended_real.w * blended_dual.xyz
                              - blended_dual.w * blended_real.xyz
                              + cross(blended_real.xyz, blended_dual.xyz));
    vec3 skinned_position = quaternion_rotate(blended_real, position) + translation;

    // la transformacion es rigida, asi que la normal solo se rota
    world_normal = normalize(quaternion_rotate(blended_real, normal));

    weight_color =
          weights.x * joint_colors[int(joints.x)]
        + weights.y * joint_colors[int(joints.y)]
        + weights.z * joint_colors[int(joints.z)]
        + weights.w * joint_colors[int(joints.w)];
    weight_color *= mix(1.0, 0.55, franja);

    gl_Position = projection * view * vec4(skinned_position, 1.0);
}
