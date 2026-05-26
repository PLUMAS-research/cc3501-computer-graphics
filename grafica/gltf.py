"""
Cargador minimalista de archivos GLTF 2.0 (formato `.gltf` + `.bin`).

Cubre lo que necesitan los ejemplos del curso sobre animacion esqueletica
y skinning: nodos con TRS, jerarquia, mallas con POSITION/NORMAL/TEXCOORD_0
(opcionalmente JOINTS_0 + WEIGHTS_0), materiales con baseColorTexture, y
animaciones de keyframes (LINEAR) sobre traslacion/rotacion/escala.

No es un loader general. No soporta:
- archivos .glb (binarios; aqui solo .gltf con .bin separado)
- interpolacion STEP ni CUBICSPLINE en animaciones
- pbrMetallicRoughness completo (solo baseColorFactor + baseColorTexture)
- normal maps, occlusion, emissive
- targets de morph
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from grafica.textures import texture_2D_setup


COMPONENT_TYPE_TO_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

ELEMENT_COUNT_BY_TYPE = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _read_accessor(gltf, buffers, accessor_index):
    """
    Devuelve un numpy array (count, components) leyendo el accessor indicado.
    Para SCALAR devuelve forma (count,).
    """
    accessor = gltf["accessors"][accessor_index]
    component_count = ELEMENT_COUNT_BY_TYPE[accessor["type"]]
    dtype = COMPONENT_TYPE_TO_DTYPE[accessor["componentType"]]
    count = accessor["count"]

    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    buffer_bytes = buffers[buffer_view["buffer"]]

    offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    element_bytes = np.dtype(dtype).itemsize * component_count
    byte_stride = buffer_view.get("byteStride", element_bytes)

    if byte_stride == element_bytes:
        # datos contiguos: lectura directa
        flat = np.frombuffer(
            buffer_bytes, dtype=dtype, count=count * component_count, offset=offset
        )
        data = flat.reshape(count, component_count) if component_count > 1 else flat
    else:
        # datos interleaved: hay que saltar entre elementos
        data = np.empty((count, component_count), dtype=dtype)
        for element_index in range(count):
            element_offset = offset + element_index * byte_stride
            chunk = np.frombuffer(
                buffer_bytes,
                dtype=dtype,
                count=component_count,
                offset=element_offset,
            )
            data[element_index] = chunk
        if component_count == 1:
            data = data.ravel()

    if accessor.get("normalized", False):
        if dtype == np.int8:
            data = np.maximum(data.astype(np.float32) / 127.0, -1.0)
        elif dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        elif dtype == np.int16:
            data = np.maximum(data.astype(np.float32) / 32767.0, -1.0)
        elif dtype == np.uint16:
            data = data.astype(np.float32) / 65535.0

    return data


def _quaternion_to_matrix(quat):
    """xyzw -> matriz 4x4 de rotacion (row-major, lista para usar con @)."""
    x, y, z, w = quat
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        return np.eye(4, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[0, 1] = 2.0 * (xy - wz)
    matrix[0, 2] = 2.0 * (xz + wy)
    matrix[1, 0] = 2.0 * (xy + wz)
    matrix[1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[1, 2] = 2.0 * (yz - wx)
    matrix[2, 0] = 2.0 * (xz - wy)
    matrix[2, 1] = 2.0 * (yz + wx)
    matrix[2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def _slerp(a, b, t):
    """Interpolacion esferica de cuaterniones (xyzw)."""
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        # casi colineales: lerp + renormalizacion
        result = a + t * (b - a)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s_a = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s_b = np.sin(theta) / sin_theta_0
    return s_a * a + s_b * b


def _compose_trs(translation, rotation_quat, scale):
    """Construye una matriz 4x4 a partir de T, R (xyzw), S."""
    matrix = _quaternion_to_matrix(rotation_quat)
    matrix[:3, 0] *= scale[0]
    matrix[:3, 1] *= scale[1]
    matrix[:3, 2] *= scale[2]
    matrix[:3, 3] = translation
    return matrix


class GltfNode:
    __slots__ = (
        "index",
        "name",
        "children",
        "mesh_index",
        "skin_index",
        "translation",
        "rotation",
        "scale",
        "matrix_override",
    )

    def __init__(self, index, name, children, mesh_index, skin_index, translation, rotation, scale, matrix_override):
        self.index = index
        self.name = name
        self.children = children
        self.mesh_index = mesh_index
        self.skin_index = skin_index
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.matrix_override = matrix_override

    def local_matrix(self):
        if self.matrix_override is not None:
            return self.matrix_override
        return _compose_trs(self.translation, self.rotation, self.scale)


class GltfPrimitive:
    __slots__ = ("positions", "normals", "uvs", "colors", "indices", "material_index", "joints", "weights")

    def __init__(self, positions, normals, uvs, colors, indices, material_index, joints, weights):
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.colors = colors
        self.indices = indices
        self.material_index = material_index
        self.joints = joints
        self.weights = weights


class GltfSkin:
    __slots__ = ("joints", "inverse_bind_matrices", "skeleton_root")

    def __init__(self, joints, inverse_bind_matrices, skeleton_root):
        self.joints = joints
        self.inverse_bind_matrices = inverse_bind_matrices
        self.skeleton_root = skeleton_root


class GltfAnimationChannel:
    __slots__ = ("node_index", "path", "times", "values", "interpolation")

    def __init__(self, node_index, path, times, values, interpolation):
        self.node_index = node_index
        self.path = path
        self.times = times
        self.values = values
        self.interpolation = interpolation


class GltfModel:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path) as f:
            gltf = json.load(f)

        base_dir = self.path.parent
        buffers = []
        for buffer in gltf["buffers"]:
            uri = buffer["uri"]
            buffer_path = base_dir / uri
            buffers.append(buffer_path.read_bytes())

        self.nodes = []
        for node_index, node_json in enumerate(gltf["nodes"]):
            translation = np.array(node_json.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            rotation = np.array(node_json.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
            scale = np.array(node_json.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
            matrix_override = None
            if "matrix" in node_json:
                # GLTF guarda matrices column-major; numpy las quiere row-major
                raw = np.array(node_json["matrix"], dtype=np.float32).reshape(4, 4, order="F")
                matrix_override = raw
            self.nodes.append(GltfNode(
                index=node_index,
                name=node_json.get("name", f"node_{node_index}"),
                children=list(node_json.get("children", [])),
                mesh_index=node_json.get("mesh"),
                skin_index=node_json.get("skin"),
                translation=translation,
                rotation=rotation,
                scale=scale,
                matrix_override=matrix_override,
            ))

        self.scene_roots = list(gltf["scenes"][gltf.get("scene", 0)]["nodes"])

        self.meshes = []
        for mesh_json in gltf["meshes"]:
            primitives = []
            for primitive_json in mesh_json["primitives"]:
                attributes = primitive_json["attributes"]
                positions = _read_accessor(gltf, buffers, attributes["POSITION"]).astype(np.float32)
                normals = (
                    _read_accessor(gltf, buffers, attributes["NORMAL"]).astype(np.float32)
                    if "NORMAL" in attributes
                    else None
                )
                uvs = (
                    _read_accessor(gltf, buffers, attributes["TEXCOORD_0"]).astype(np.float32)
                    if "TEXCOORD_0" in attributes
                    else None
                )
                colors = None
                if "COLOR_0" in attributes:
                    raw_colors = _read_accessor(gltf, buffers, attributes["COLOR_0"]).astype(np.float32)
                    if raw_colors.shape[1] == 3:
                        # COLOR_0 puede venir como vec3; lo extendemos a vec4 con alpha=1
                        colors = np.hstack([raw_colors, np.ones((raw_colors.shape[0], 1), dtype=np.float32)])
                    else:
                        colors = raw_colors
                joints = (
                    _read_accessor(gltf, buffers, attributes["JOINTS_0"]).astype(np.int32)
                    if "JOINTS_0" in attributes
                    else None
                )
                weights = (
                    _read_accessor(gltf, buffers, attributes["WEIGHTS_0"]).astype(np.float32)
                    if "WEIGHTS_0" in attributes
                    else None
                )
                indices = (
                    _read_accessor(gltf, buffers, primitive_json["indices"]).astype(np.uint32)
                    if "indices" in primitive_json
                    else None
                )
                primitives.append(GltfPrimitive(
                    positions=positions,
                    normals=normals,
                    uvs=uvs,
                    colors=colors,
                    indices=indices,
                    material_index=primitive_json.get("material"),
                    joints=joints,
                    weights=weights,
                ))
            self.meshes.append(primitives)

        # skins: cada uno guarda la lista de joints (indices a nodos) y sus
        # inverse bind matrices. Necesario para skinning lineal/dual quaternion
        self.skins = []
        for skin_json in gltf.get("skins", []):
            joints = list(skin_json["joints"])
            if "inverseBindMatrices" in skin_json:
                ibm_flat = _read_accessor(gltf, buffers, skin_json["inverseBindMatrices"])
                # GLTF guarda mat4 en orden column-major; numpy las lee como (count, 16)
                # y las queremos como (count, 4, 4) row-major para usar con @
                inverse_bind_matrices = ibm_flat.reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float32)
            else:
                inverse_bind_matrices = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
            self.skins.append(GltfSkin(
                joints=joints,
                inverse_bind_matrices=inverse_bind_matrices,
                skeleton_root=skin_json.get("skeleton"),
            ))

        # materiales: solo baseColorFactor + baseColorTexture
        self.materials = []
        for material_json in gltf.get("materials", []):
            pbr = material_json.get("pbrMetallicRoughness", {})
            base_color_factor = np.array(
                pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32
            )
            base_color_texture_index = None
            if "baseColorTexture" in pbr:
                base_color_texture_index = pbr["baseColorTexture"]["index"]
            self.materials.append({
                "name": material_json.get("name", ""),
                "base_color_factor": base_color_factor,
                "base_color_texture": base_color_texture_index,
            })

        # texturas: cargamos imagenes una vez y devolvemos texture IDs de OpenGL
        # Se hace por demanda en `upload_textures` porque GL exige contexto activo.
        self._gltf = gltf
        self._base_dir = base_dir
        self.texture_gl_ids = None

        # animaciones: una sola por ahora (la primera del archivo). Lista de canales.
        self.animations = []
        for animation_json in gltf.get("animations", []):
            channels = []
            samplers = animation_json["samplers"]
            for channel_json in animation_json["channels"]:
                sampler = samplers[channel_json["sampler"]]
                times = _read_accessor(gltf, buffers, sampler["input"]).astype(np.float32)
                values = _read_accessor(gltf, buffers, sampler["output"]).astype(np.float32)
                channels.append(GltfAnimationChannel(
                    node_index=channel_json["target"]["node"],
                    path=channel_json["target"]["path"],
                    times=times,
                    values=values,
                    interpolation=sampler.get("interpolation", "LINEAR"),
                ))
            self.animations.append({
                "name": animation_json.get("name", ""),
                "channels": channels,
                "duration": max((c.times[-1] for c in channels if len(c.times) > 0), default=0.0),
            })

    def upload_textures(self):
        """Sube las imagenes a GPU como texturas 2D. Llamar con contexto OpenGL activo."""
        gltf = self._gltf
        base_dir = self._base_dir
        image_texture_ids = {}

        # Cada entry de `textures` referencia un `images[source]` y opcionalmente un `samplers[]`.
        # Aqui solo usamos el `source`; el sampler GLTF lo ignoramos (`texture_2D_setup` usa LINEAR + CLAMP).
        self.texture_gl_ids = []
        for texture_json in gltf.get("textures", []):
            image_index = texture_json["source"]
            if image_index in image_texture_ids:
                self.texture_gl_ids.append(image_texture_ids[image_index])
                continue
            image_json = gltf["images"][image_index]
            if "uri" not in image_json:
                # imagen embedded en bufferView (no soportado por ahora)
                self.texture_gl_ids.append(None)
                continue
            image_path = base_dir / image_json["uri"]
            pil_image = Image.open(image_path)
            if pil_image.mode not in ("RGB", "RGBA"):
                pil_image = pil_image.convert("RGBA")
            texture_id = texture_2D_setup(pil_image)
            image_texture_ids[image_index] = texture_id
            self.texture_gl_ids.append(texture_id)

    def sample_animation(self, animation_index, time_seconds):
        """
        Evalua todos los canales de la animacion en el instante dado.
        Devuelve un dict {node_index: {"translation": vec3, "rotation": quat, "scale": vec3}}
        con valores interpolados. Solo se llenan los nodos animados.
        """
        animation = self.animations[animation_index]
        duration = animation["duration"]
        if duration > 0.0:
            time_seconds = time_seconds % duration

        out = {}
        for channel in animation["channels"]:
            times = channel.times
            values = channel.values
            if len(times) == 0:
                continue

            if time_seconds <= times[0]:
                value = values[0]
            elif time_seconds >= times[-1]:
                value = values[-1]
            else:
                next_index = int(np.searchsorted(times, time_seconds, side="right"))
                prev_index = next_index - 1
                t0 = times[prev_index]
                t1 = times[next_index]
                alpha = (time_seconds - t0) / (t1 - t0) if t1 > t0 else 0.0
                v0 = values[prev_index]
                v1 = values[next_index]
                if channel.path == "rotation":
                    value = _slerp(v0, v1, alpha)
                else:
                    value = v0 + alpha * (v1 - v0)

            node_entry = out.setdefault(channel.node_index, {})
            node_entry[channel.path] = value

        return out

    def node_trs_with_overrides(self, animation_overrides):
        """
        Devuelve la matriz local de cada nodo, combinando los valores de bind pose
        del archivo GLTF con los overrides de animacion (cuando existen).
        """
        local_matrices = []
        for node in self.nodes:
            override = animation_overrides.get(node.index)
            if override is None:
                local_matrices.append(node.local_matrix())
            else:
                translation = override.get("translation", node.translation)
                rotation = override.get("rotation", node.rotation)
                scale = override.get("scale", node.scale)
                local_matrices.append(_compose_trs(translation, rotation, scale))
        return local_matrices
