"""Cuaterniones duales para el Dual Quaternion Skinning (DQS).

Un cuaternion dual (q_r, q_d) guarda una transformacion rigida: q_r es el
cuaternion unitario de la rotacion y q_d = 0.5 * (0, t) * q_r codifica la
traslacion t. DQS mezcla los cuaterniones duales de los huesos con los pesos del
vertice y divide por el largo de la parte real. El resultado vuelve a ser una
transformacion rigida, de modo que no encoge la malla como la mezcla de matrices
del Linear Blend Skinning.

Lo usan esqueleto_lbs (en vertex_program_dqs.glsl y en la AABB del overlay de
colision) y skinning. Los cuaterniones van en formato xyzw, como en GLSL.
"""

import numpy as np


def matrix_to_quaternion(rotation_matrix_3x3):
    """
    Convierte una matriz 3x3 de rotacion (ortonormal, det = 1) a un
    cuaternion en formato xyzw. Es la formula clasica basada en la traza
    con casos para evitar perdida de precision cuando el coeficiente
    dominante es chico.
    """
    m = rotation_matrix_3x3
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def skin_matrix_to_dual_quaternion(skin_matrix):
    """
    Descompone una matriz de skinning en un dual quaternion (parte real,
    parte dual). Asume que la parte 3x3 es una rotacion pura (sin escala
    ni shear). Si hay escala uniforme la extrae primero; DQS clasico no
    maneja escala asi que el resultado sera fiel solo para transformaciones
    rigidas.

    Devuelve (real_xyzw, dual_xyzw).
    """
    rotation_part = skin_matrix[:3, :3]
    translation = skin_matrix[:3, 3]

    # extraemos escala uniforme si existe. Sin este paso, matrices con escala
    # rompen la formula de cuaternion (la traza ya no esta en el rango valido)
    column_scales = np.linalg.norm(rotation_part, axis=0)
    uniform_scale = float(np.mean(column_scales))
    if uniform_scale > 1e-8:
        rotation_normalized = rotation_part / uniform_scale
    else:
        rotation_normalized = rotation_part

    real_quat = matrix_to_quaternion(rotation_normalized)

    # dual = 0.5 * (0, t) * real_quat, con t = translation
    # (0, t) * (qw, qx, qy, qz) en formato xyzw:
    # resultado.w = -t . q_xyz
    # resultado.xyz = qw * t + cross(t, q_xyz)
    q_xyz = real_quat[:3]
    q_w = real_quat[3]
    dual_w = -float(np.dot(translation, q_xyz))
    dual_xyz = q_w * translation + np.cross(translation, q_xyz)
    dual_quat = 0.5 * np.array([dual_xyz[0], dual_xyz[1], dual_xyz[2], dual_w], dtype=np.float32)

    return real_quat, dual_quat


def _rotar(q, v):
    """quaternion_rotate() del shader para muchos vectores: v + 2 q.xyz x (q.xyz x v + q.w v)."""
    q_xyz, q_w = q[:, :3], q[:, 3:]
    return v + 2.0 * np.cross(q_xyz, np.cross(q_xyz, v) + q_w * v)


def aplicar_dqs(posiciones, juntas, pesos, reales, duales):
    """Lo mismo que vertex_program_dqs.glsl, en la CPU y para todos los vertices a la vez.

    posiciones es (n, 3); juntas y pesos, (n, 4); reales y duales, (huesos, 4).
    Devuelve las posiciones deformadas, (n, 3).
    """
    indices = juntas.astype(int)
    referencia = reales[indices[:, 0]]
    mezcla_real = np.zeros((len(posiciones), 4))
    mezcla_dual = np.zeros((len(posiciones), 4))
    for influencia in range(indices.shape[1]):
        q_real = reales[indices[:, influencia]]
        q_dual = duales[indices[:, influencia]]
        # q y -q son la misma rotacion: se suma con el signo del lado de la referencia
        signo = np.where(np.sum(q_real * referencia, axis=1) < 0.0, -1.0, 1.0)[:, None]
        peso = pesos[:, influencia][:, None]
        mezcla_real += signo * peso * q_real
        mezcla_dual += signo * peso * q_dual
    largo = np.linalg.norm(mezcla_real, axis=1, keepdims=True)
    mezcla_real /= largo
    mezcla_dual /= largo
    traslacion = 2.0 * (mezcla_real[:, 3:] * mezcla_dual[:, :3]
                        - mezcla_dual[:, 3:] * mezcla_real[:, :3]
                        + np.cross(mezcla_real[:, :3], mezcla_dual[:, :3]))
    return _rotar(mezcla_real, posiciones) + traslacion
