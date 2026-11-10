"""
Skinning con un modelo GLTF: comparacion entre Linear Blend Skinning (LBS)
y Dual Quaternion Skinning (DQS).

Cada vertice del modelo tiene hasta cuatro huesos que lo influencian, con
pesos que suman 1. Cada hueso provee una matriz de skinning que es el
producto entre la pose global actual del hueso y su inverse bind matrix.
LBS mezcla esas matrices linealmente: rapido pero degenera bajo torsion
(efecto candy-wrapper). DQS las convierte a dual quaternions, mezcla en la
esfera de cuaterniones y renormaliza: mantiene volumen mejor pero cuesta
mas.

El modelo por omision (un zorzal con animacion de aleteo) tiene un quirk de export:
el archivo tiene dos esqueletos con los mismos nombres y la animacion mueve
el segundo, pero la malla esta vinculada al primero. Cada joint de la malla toma
la pose global del nodo de igual nombre que mueve la animacion.

Teclas:
    m             alterna LBS / DQS
    e             muestra / oculta el esqueleto (lineas entre huesos)
    a             pasa a la animacion siguiente, si el archivo trae varias
    espacio       pausa / reanuda
    , y .         retrocede / avanza 0.05 s
    r             reset a t = 0
    mouse izq     rotar camara
    mouse der     pan
    rueda         zoom

Otros modelos:
    uv run python caja_de_juguetes.py skinning ruta/al/modelo.gltf --animacion 2

    El archivo tiene que ser un .gltf con su .bin aparte, con un esqueleto
    (skin) y a lo mas cuatro huesos por vertice. Un .glb, un .fbx o un .blend
    se convierten con Blender usando tools/convertir_a_gltf.py, que ademas junta
    en un solo archivo las animaciones que vienen en archivos separados. El
    ejemplo dibuja solo las mallas con skin, usa el primer skin que encuentra y
    asume el eje Y hacia arriba. El largo de los arreglos de huesos de los
    shaders se fija al compilarlos, con la cantidad de huesos del modelo.

    Modelos probados con este ejemplo:
    - Fox, de los modelos de muestra de Khronos (modelo CC0; rig y animaciones
      CC-BY 4.0). Se carga tal cual, con 24 huesos y tres animaciones:
      github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox/glTF
    - Animated Human, de Quaternius (CC0 en poly.pizza/m/c3Ibh9I3udk). Viene
      en .glb, con 41 huesos y ocho animaciones.
    - Low poly pigeon, de mujtaba-io (CC0 en opengameart.org/content/
      low-poly-3d-pigeon-model-rigged-animated-untextured). Viene en .blend,
      con 13 huesos y tres animaciones.
    - Animated Characters 1, de Kenney (CC0 en kenney.nl/assets/
      animated-characters-1). El modelo y cada animacion vienen en un .fbx
      aparte, y las texturas, en una carpeta que hay que conectar en Blender.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from pyglet.gl import GLfloat
from PIL import Image

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.background import GradientBackground
from grafica.gltf import GltfModel, _compose_trs
from grafica.scenegraph import Scenegraph
from grafica.textures import texture_2D_setup
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline

from examples.esqueleto_lbs.cuaterniones_duales import skin_matrix_to_dual_quaternion


DEFAULT_GLTF = "assets/zorzal-aleteo.gltf"

# El export GLTF del zorzal no incluyo los baseColorTexture. Los hueso
# por nombre estan en zorzal.mtl / zorzal.obj. Mapeamos por nombre de
# material del GLTF al archivo de textura que usaba el OBJ original.
# Si pasas otro modelo con texturas en su GLTF, este mapeo se ignora y
# se usan las texturas del archivo
ZORZAL_MATERIAL_TEXTURE_MAP = {
    "M_Blackbird": "zorzal.png",
    "M_BlackbirdEyes": "zorzal.png",
    "M_BlackbirdWings": "plumas.png",
}


def animated_joint_nodes(model, animation_index):
    """
    Para cada joint del skin de la malla, el nodo cuya pose global usa el skinning.

    El zorzal viene con dos esqueletos con los mismos nombres: la malla esta
    pegada al primero y el aleteo mueve el segundo. Copiar las transformaciones
    locales del segundo al primero no basta, porque la raiz del segundo se llama
    distinto (blackbird_Pelvis_bone.001) y su movimiento se pierde: el cuerpo
    quedaba sin la inclinacion de 13 grados y sin el sube y baja del aleteo. Por
    eso cada joint toma la pose global del nodo de igual nombre que mueve la
    animacion. Si la animacion mueve los joints del skin, cada joint es su propio
    nodo.
    """
    mesh_skin = next(model.skins[n.skin_index] for n in model.nodes if n.skin_index is not None)
    if animation_index is None or not model.animations:
        return list(mesh_skin.joints)
    animated = {channel.node_index for channel in model.animations[animation_index]["channels"]}
    by_name = {model.nodes[index].name: index for index in animated}
    return [j if j in animated else by_name.get(model.nodes[j].name, j) for j in mesh_skin.joints]


def load_skinning_pipeline(vertex_path, fragment_path, n_joints):
    """
    Como load_pipeline(), pero define NUM_JOINTS en el vertex program antes de
    compilarlo. GLSL exige que el largo de un arreglo uniform sea una constante,
    asi que se fija aqui, con la cantidad de huesos del modelo cargado.
    """
    vertex_source = Path(vertex_path).read_text()
    assert vertex_source.startswith("#version 330"), vertex_path
    vertex_source = vertex_source.replace(
        "#version 330", f"#version 330\n#define NUM_JOINTS {n_joints}", 1
    )
    fragment_source = Path(fragment_path).read_text()
    return pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(vertex_source, "vertex"),
        pyglet.graphics.shader.Shader(fragment_source, "fragment"),
    )


def compute_skin_matrices(model, graph, node_to_scenegraph_name, joint_nodes=None):
    """
    Devuelve (skin_matrices, real_quats, dual_quats) listos para subir a
    los shaders. skin_matrices: array (N, 4, 4) row-major; real_quats y
    dual_quats: arrays (N, 4) en xyzw. joint_nodes es, para cada joint, el
    nodo cuya pose global se usa (ver animated_joint_nodes); por omision, el
    propio joint.
    """
    graph.calculate_global_transforms()
    mesh_skin = None
    for node in model.nodes:
        if node.skin_index is not None:
            mesh_skin = model.skins[node.skin_index]
            break
    assert mesh_skin is not None, "el modelo no tiene skin"

    n_joints = len(mesh_skin.joints)
    skin_matrices = np.empty((n_joints, 4, 4), dtype=np.float32)
    real_quats = np.empty((n_joints, 4), dtype=np.float32)
    dual_quats = np.empty((n_joints, 4), dtype=np.float32)

    if joint_nodes is None:
        joint_nodes = mesh_skin.joints
    for joint_index, joint_node_index in enumerate(joint_nodes):
        joint_global = graph.global_transforms[node_to_scenegraph_name[joint_node_index]]
        skin_matrix = joint_global @ mesh_skin.inverse_bind_matrices[joint_index]
        skin_matrices[joint_index] = skin_matrix
        real, dual = skin_matrix_to_dual_quaternion(skin_matrix)
        real_quats[joint_index] = real
        dual_quats[joint_index] = dual

    return skin_matrices, real_quats, dual_quats


def compute_skinned_bbox(skin_matrices, primitives_data):
    """
    Calcula la AABB del modelo en pose actual aplicando skinning en CPU.
    Solo se usa una vez al inicio para encuadrar la camara
    """
    min_pt = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    max_pt = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for prim in primitives_data:
        positions = prim["positions"]
        joints = prim["joints"]
        weights = prim["weights"]
        ones = np.ones((positions.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([positions, ones])
        skinned = np.zeros_like(homogeneous)
        for influence_index in range(4):
            joint_indices = joints[:, influence_index].astype(int)
            w = weights[:, influence_index][:, None]
            per_vertex_matrices = skin_matrices[joint_indices]
            contributions = np.einsum("nij,nj->ni", per_vertex_matrices, homogeneous)
            skinned += w * contributions
        world_positions = skinned[:, :3]
        min_pt = np.minimum(min_pt, world_positions.min(axis=0))
        max_pt = np.maximum(max_pt, world_positions.max(axis=0))
    return min_pt, max_pt


def gather_skinned_primitives(model):
    """
    Recolecta cada primitiva skinneada como un dict independiente, conservando
    su material para que el renderer pueda atar la textura correcta.
    """
    primitives_data = []
    for mesh_index, primitives in enumerate(model.meshes):
        for primitive_index, primitive in enumerate(primitives):
            if primitive.joints is None:
                continue
            n = primitive.positions.shape[0]
            normals = (
                primitive.normals.astype(np.float32)
                if primitive.normals is not None
                else np.zeros((n, 3), dtype=np.float32)
            )
            uvs = (
                primitive.uvs.astype(np.float32)
                if primitive.uvs is not None
                else np.zeros((n, 2), dtype=np.float32)
            )
            colors = (
                primitive.colors.astype(np.float32)
                if primitive.colors is not None
                else np.ones((n, 4), dtype=np.float32)
            )
            indices = (
                primitive.indices.astype(np.uint32)
                if primitive.indices is not None
                else np.arange(n, dtype=np.uint32)
            )
            primitives_data.append({
                "positions": primitive.positions.astype(np.float32),
                "normals": normals,
                "uvs": uvs,
                "colors": colors,
                "joints": primitive.joints.astype(np.float32),
                "weights": primitive.weights.astype(np.float32),
                "indices": indices,
                "material_index": primitive.material_index,
            })
    return primitives_data


@click.command("skinning", short_help="LBS vs DQS sobre un zorzal con animacion de aleteo")
@click.argument("gltf_path", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--width", type=int, default=1280)
@click.option("--height", type=int, default=800)
@click.option("--animacion", "animation_index", type=int, default=0,
              help="indice de la animacion inicial; la tecla a pasa a la siguiente")
def skinning(gltf_path, width, height, animation_index):
    if gltf_path is None:
        gltf_path = str(Path(__file__).parent.parent.parent / DEFAULT_GLTF)

    print(f"[skinning] cargando {gltf_path}")
    model = GltfModel(gltf_path)

    # encuentra el skin que la malla usa
    mesh_skin_index = None
    mesh_node_for_skin = None
    for node in model.nodes:
        if node.skin_index is not None:
            mesh_skin_index = node.skin_index
            mesh_node_for_skin = node
            break
    if mesh_skin_index is None:
        raise click.ClickException("el modelo no tiene primitivas con skinning")
    n_joints = len(model.skins[mesh_skin_index].joints)
    print(f"[skinning] nodos={len(model.nodes)} skins={len(model.skins)} joints_efectivos={n_joints}")
    for index, animation in enumerate(model.animations):
        print(
            f"[skinning] animacion {index}: '{animation['name']}' "
            f"duracion={animation['duration']:.2f}s canales={len(animation['channels'])}"
        )
    if model.animations and not 0 <= animation_index < len(model.animations):
        raise click.ClickException(
            f"--animacion {animation_index} no existe: el archivo trae {len(model.animations)}"
        )

    joint_nodes = animated_joint_nodes(model, animation_index if model.animations else None)
    otros = sum(1 for j, n in zip(model.skins[mesh_skin_index].joints, joint_nodes) if j != n)
    if otros:
        print(f"[skinning] {otros} joints toman la pose de otro esqueleto animado con los mismos nombres")

    window = pyglet.window.Window(width=width, height=height, caption="skinning: LBS vs DQS")
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    background = GradientBackground()

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    # construimos las dos pipelines
    lbs_pipeline = load_skinning_pipeline(
        Path(__file__).parent / "vertex_program_lbs.glsl",
        Path(__file__).parent / "fragment_program.glsl",
        n_joints,
    )
    dqs_pipeline = load_skinning_pipeline(
        Path(__file__).parent / "vertex_program_dqs.glsl",
        Path(__file__).parent / "fragment_program.glsl",
        n_joints,
    )

    # el grafo de escena gestiona la jerarquia de nodos (incluidos los
    # huesos). No usamos sus mesh-instances aqui: dibujamos el mesh
    # con un vertex_list propio porque necesitamos subir uniforms de skin
    graph = Scenegraph("root")

    node_to_scenegraph_name = []
    for node in model.nodes:
        scenegraph_name = f"gltf_node_{node.index}"
        graph.add_transform(scenegraph_name, node.local_matrix())
        node_to_scenegraph_name.append(scenegraph_name)
    for node in model.nodes:
        for child_index in node.children:
            graph.add_edge(node_to_scenegraph_name[node.index], node_to_scenegraph_name[child_index])
    for root_index in model.scene_roots:
        graph.add_edge("root", node_to_scenegraph_name[root_index])

    # primitivas + texturas. Si el GLTF trae baseColorTexture las usamos.
    # En caso contrario (el zorzal exportado quedo asi), miramos el nombre
    # del material en el mapeo manual y cargamos el PNG correspondiente
    primitives_data = gather_skinned_primitives(model)
    n_vertices_total = sum(p["positions"].shape[0] for p in primitives_data)
    n_triangles_total = sum(p["indices"].shape[0] // 3 for p in primitives_data)
    print(f"[skinning] primitivas={len(primitives_data)} vertices={n_vertices_total} triangulos={n_triangles_total}")

    # Algunos exports (los de FBX2glTF, por ejemplo) dejan una escala uniforme s
    # en las matrices de skinning, incluso en reposo. LBS la aplica sin problema,
    # pero un cuaternion dual solo guarda rotacion y traslacion, de modo que DQS
    # dibujaria el modelo s veces mas chico. Si s es la misma en todos los huesos,
    # se pasa a la malla: posiciones por s e inverse bind matrices por diag(1/s).
    # El producto de la matriz de skinning por la posicion no cambia, y las
    # matrices de skinning quedan rigidas
    rest_skin_matrices, _, _ = compute_skin_matrices(model, graph, node_to_scenegraph_name)
    joint_scales = np.cbrt(np.abs(np.linalg.det(rest_skin_matrices[:, :3, :3])))
    skin_scale = float(np.median(joint_scales))
    if abs(skin_scale - 1.0) > 1e-3:
        if np.ptp(joint_scales) > 1e-3 * skin_scale:
            print(
                f"[skinning] aviso: las matrices de skinning traen escalas distintas por hueso "
                f"({joint_scales.min():.3f} a {joint_scales.max():.3f}) y DQS no las representa"
            )
        else:
            for prim in primitives_data:
                prim["positions"] *= skin_scale
            mesh_skin = model.skins[mesh_skin_index]
            mesh_skin.inverse_bind_matrices = mesh_skin.inverse_bind_matrices @ np.diag(
                [1.0 / skin_scale, 1.0 / skin_scale, 1.0 / skin_scale, 1.0]
            ).astype(np.float32)
            print(f"[skinning] escala {skin_scale:.3f} de las matrices de skinning pasada a la malla")

    assets_dir = Path(__file__).parent.parent.parent / "assets"
    texture_cache_by_filename = {}

    def load_texture_by_filename(filename):
        if filename in texture_cache_by_filename:
            return texture_cache_by_filename[filename]
        path = assets_dir / filename
        image = Image.open(path)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
        # GLTF define el origen de las coordenadas UV en la esquina superior
        # izquierda de la imagen. PIL ya entrega los datos con la fila 0
        # arriba: si subimos sin flip, sample con v=0 lee la fila 0 (arriba
        # de la imagen original) y los UVs del GLTF caen donde corresponde.
        # Con flip vertical (el default de texture_2D_setup) terminariamos
        # leyendo la imagen invertida y la textura se ve mal aunque no haya
        # transparencia
        texture_id = texture_2D_setup(image, flip_top_bottom=False)
        texture_cache_by_filename[filename] = texture_id
        print(f"[skinning] textura cargada: {filename} ({image.mode}, {image.size[0]}x{image.size[1]})")
        return texture_id

    # Blender y los modelos de muestra de Khronos siguen la especificacion: V = 0
    # es la fila de arriba de la imagen, asi que la textura se sube sin invertir
    model.upload_textures(flip_top_bottom=False)
    white_pixel_texture = texture_2D_setup(Image.new("RGB", (1, 1), color=(255, 255, 255)))

    for prim in primitives_data:
        texture_id = None
        material = model.materials[prim["material_index"]] if prim["material_index"] is not None else None
        if material is not None and material["base_color_texture"] is not None:
            texture_id = model.texture_gl_ids[material["base_color_texture"]]
        material_name = material["name"] if material is not None else ""
        if texture_id is None and material_name in ZORZAL_MATERIAL_TEXTURE_MAP:
            texture_id = load_texture_by_filename(ZORZAL_MATERIAL_TEXTURE_MAP[material_name])
        if texture_id is None:
            texture_id = white_pixel_texture
        prim["texture_id"] = texture_id
        prim["base_color"] = (
            material["base_color_factor"][:3] if material is not None else np.ones(3, dtype=np.float32)
        )
        # las alas usan plumas.png con alpha. Usamos blending para los
        # bordes suaves de las plumas + un discard chico para evitar
        # escribir al depth buffer pixeles invisibles. Las primitivas
        # opacas (cuerpo, ojos) no necesitan ninguna de las dos cosas
        prim["is_transparent"] = "Wings" in material_name
        prim["alpha_cutoff"] = 0.01 if prim["is_transparent"] else -1.0

    # un par de vertex_list por primitiva: una por LBS y otra por DQS
    # (pyglet asocia cada VAO a un ShaderProgram concreto)
    def make_vertex_list(pipeline, prim):
        n = prim["positions"].shape[0]
        vl = pipeline.vertex_list_indexed(
            n, GL.GL_TRIANGLES, prim["indices"].tolist()
        )
        vl.position[:] = prim["positions"].flatten()
        vl.normal[:] = prim["normals"].flatten()
        vl.uv[:] = prim["uvs"].flatten()
        vl.joints[:] = prim["joints"].flatten()
        vl.weights[:] = prim["weights"].flatten()
        return vl

    for prim in primitives_data:
        prim["lbs_vertex_list"] = make_vertex_list(lbs_pipeline, prim)
        prim["dqs_vertex_list"] = make_vertex_list(dqs_pipeline, prim)

    # transparentes al final del listado: con depth test estandar +
    # blending el orden importa cuando hay piezas semitransparentes.
    # Las opacas (cuerpo, ojos) escriben al depth y luego las alas se
    # mezclan encima respetando ese depth
    primitives_data.sort(key=lambda p: p["is_transparent"])

    # pipeline y geometria del esqueleto: una arista por cada par
    # (padre, hijo) que conecta huesos del skin. Las posiciones se
    # reescriben por frame desde los transforms globales del grafo
    skeleton_pipeline = load_pipeline(
        Path(__file__).parent / "skeleton_vertex_program.glsl",
        Path(__file__).parent / "skeleton_fragment_program.glsl",
    )
    joint_set = set(model.skins[mesh_skin_index].joints)
    skeleton_edges = []
    for node in model.nodes:
        if node.index not in joint_set:
            continue
        for child_index in node.children:
            if child_index in joint_set:
                skeleton_edges.append((node.index, child_index))
    n_skeleton_positions = len(skeleton_edges) * 2
    skeleton_gpu = skeleton_pipeline.vertex_list(n_skeleton_positions, GL.GL_LINES)
    skeleton_positions_buffer = np.zeros((n_skeleton_positions, 3), dtype=np.float32)
    print(f"[skinning] esqueleto: {len(skeleton_edges)} aristas entre huesos")

    # GLTF asume Y up por defecto y este zorzal lo cumple (X y Z son el
    # plano horizontal por donde se extienden las alas). No aplicamos
    # detector automatico aqui porque "el eje con mayor extension" falla
    # para personajes en pose de vuelo: el wingspan > altura. Si usas otro
    # modelo Z-up edita el root con tr.rotationX(-pi/2) manualmente
    bind_skin_matrices, _, _ = compute_skin_matrices(model, graph, node_to_scenegraph_name)
    min_pt, max_pt = compute_skinned_bbox(bind_skin_matrices, primitives_data)
    print(f"[skinning] bind pose bbox: min={min_pt} max={max_pt}")

    center = (min_pt + max_pt) * 0.5
    diagonal = float(np.linalg.norm(max_pt - min_pt))
    if diagonal <= 0:
        diagonal = 1.0

    # la camara queda a la distancia en que la esfera que envuelve al modelo en
    # reposo cabe en el campo de vision vertical de 45 grados
    camera_direction = np.array([0.7, 0.3, 0.8], dtype=np.float32)
    camera_direction /= np.linalg.norm(camera_direction)
    camera_distance = 0.5 * diagonal / np.sin(np.radians(45.0 / 2.0))
    initial_eye = center + camera_direction * camera_distance
    initial_view = tr.lookAt(initial_eye, center.astype(np.float32), np.array([0.0, 1.0, 0.0]))

    arcball = Arcball(
        np.linalg.inv(initial_view),
        np.array((width, height), dtype=float),
        diagonal,
        center.astype(np.float64),
    )
    arcball.set_initial_state()
    arcball.set_distance_limits(min_distance=diagonal * 0.2, max_distance=diagonal * 6.0)

    projection = tr.perspective(45.0, width / height, diagonal * 0.01, diagonal * 10.0)

    light_direction = np.array([0.4, 0.7, 0.5], dtype=np.float32)
    ambient_strength = 0.35

    # pyglet exige que las assignaciones a uniformes-array tengan tipo ctypes
    # con la forma exacta. Pre-alocamos los buffers una vez y los rellenamos
    # via vista numpy en cada frame para evitar convertir un sub-array por hueso
    Mat4Array = (GLfloat * 16) * n_joints
    Vec4Array = (GLfloat * 4) * n_joints
    skin_matrices_buffer = Mat4Array()
    skin_matrices_view = np.frombuffer(skin_matrices_buffer, dtype=np.float32).reshape(n_joints, 16)
    real_quat_buffer = Vec4Array()
    real_quat_view = np.frombuffer(real_quat_buffer, dtype=np.float32).reshape(n_joints, 4)
    dual_quat_buffer = Vec4Array()
    dual_quat_view = np.frombuffer(dual_quat_buffer, dtype=np.float32).reshape(n_joints, 4)

    label_method = pyglet.text.Label(
        text="", font_name="Fira Code", font_size=14,
        x=12, y=height - 24, color=(255, 240, 200, 255),
    )
    label_time = pyglet.text.Label(
        text="", font_name="Fira Code", font_size=12,
        x=12, y=height - 48, color=(230, 230, 230, 255),
    )
    label_help = pyglet.text.Label(
        text="", font_name="Fira Code", font_size=11,
        x=12, y=height - 68, color=(180, 200, 255, 255),
    )

    state = {
        "time": 0.0,
        "paused": False,
        "method": "LBS",
        "show_skeleton": False,
        "animation": animation_index,
        "joint_nodes": joint_nodes,
    }

    def apply_animation_to_graph():
        if not model.animations:
            return
        overrides = model.sample_animation(state["animation"], state["time"])
        for node_index, trs in overrides.items():
            translation = trs.get("translation", model.nodes[node_index].translation)
            rotation = trs.get("rotation", model.nodes[node_index].rotation)
            scale = trs.get("scale", model.nodes[node_index].scale)
            graph.nodes[node_to_scenegraph_name[node_index]]["transform"] = _compose_trs(
                translation, rotation, scale
            )

    def update_labels():
        label_method.text = f"metodo: {state['method']}"
        if model.animations:
            animation = model.animations[state["animation"]]
            duration = animation["duration"]
            label_time.text = (
                f"{animation['name']}   t = {state['time']:5.2f} / {duration:5.2f} s   "
                f"{'[pausa]' if state['paused'] else '[play] '}"
                f"   esqueleto: {'on' if state['show_skeleton'] else 'off'}"
            )
        else:
            label_time.text = f"sin animacion   esqueleto: {'on' if state['show_skeleton'] else 'off'}"
        label_help.text = "m LBS/DQS  e esqueleto  a animacion  espacio pausa  , . scrub  r reset  mouse rotar/pan/zoom"

    apply_animation_to_graph()
    update_labels()

    @window.event
    def on_draw():
        window.clear()
        background.draw()
        view = np.linalg.inv(arcball.pose)

        skin_matrices, real_quats, dual_quats = compute_skin_matrices(
            model, graph, node_to_scenegraph_name, state["joint_nodes"]
        )

        if state["method"] == "LBS":
            pipeline = lbs_pipeline
            vertex_list_key = "lbs_vertex_list"
        else:
            pipeline = dqs_pipeline
            vertex_list_key = "dqs_vertex_list"

        pipeline.use()
        pipeline["view"] = view.reshape(16, 1, order="F")
        pipeline["projection"] = projection.reshape(16, 1, order="F")
        pipeline["transform"] = tr.identity().reshape(16, 1, order="F")
        pipeline["light_direction"] = light_direction.reshape(3, 1, order="F")
        pipeline["ambient_strength"] = ambient_strength

        if state["method"] == "LBS":
            skin_matrices_view[:] = skin_matrices.transpose(0, 2, 1).reshape(n_joints, 16)
            pipeline["skin_matrices"] = skin_matrices_buffer
        else:
            real_quat_view[:] = real_quats
            dual_quat_view[:] = dual_quats
            pipeline["dual_quat_real"] = real_quat_buffer
            pipeline["dual_quat_dual"] = dual_quat_buffer

        # una pasada por primitiva: cada una con su textura.
        # En la pasada transparente desactivamos depth write: los pixeles
        # de pluma se mezclan con lo que hay detras (cuerpo) sin pelearse
        # entre ellos en z. Si dejaramos depth write encendido, dos planos
        # de pluma se ocluirian uno al otro en orden arbitrario y veriamos
        # bordes negros donde uno tapa al otro
        for prim in primitives_data:
            if prim["is_transparent"]:
                GL.glDepthMask(GL.GL_FALSE)
            else:
                GL.glDepthMask(GL.GL_TRUE)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, prim["texture_id"])
            pipeline["alpha_cutoff"] = prim["alpha_cutoff"]
            pipeline["base_color_factor"] = prim["base_color"].reshape(3, 1, order="F")
            prim[vertex_list_key].draw(GL.GL_TRIANGLES)
        GL.glDepthMask(GL.GL_TRUE)

        if state["show_skeleton"]:
            # cada arista del esqueleto: posicion del nodo padre y del nodo
            # hijo, leidas del transform global actualizado. Apagamos depth
            # test para que el esqueleto sea visible aunque este dentro de
            # la malla, asi se ve como un rayos X pedagogico
            # cada joint se dibuja donde lo deja el nodo que mueve la animacion
            source = dict(zip(model.skins[mesh_skin_index].joints, state["joint_nodes"]))
            for edge_index, (parent_node_index, child_node_index) in enumerate(skeleton_edges):
                parent_pos = graph.global_transforms[node_to_scenegraph_name[source[parent_node_index]]][:3, 3]
                child_pos = graph.global_transforms[node_to_scenegraph_name[source[child_node_index]]][:3, 3]
                skeleton_positions_buffer[2 * edge_index] = parent_pos
                skeleton_positions_buffer[2 * edge_index + 1] = child_pos
            skeleton_gpu.position[:] = skeleton_positions_buffer.flatten()

            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glLineWidth(1.5)
            skeleton_pipeline.use()
            skeleton_pipeline["view"] = view.reshape(16, 1, order="F")
            skeleton_pipeline["projection"] = projection.reshape(16, 1, order="F")
            skeleton_pipeline["line_color"] = np.array([1.0, 0.85, 0.2], dtype=np.float32).reshape(3, 1, order="F")
            skeleton_gpu.draw(GL.GL_LINES)
            GL.glEnable(GL.GL_DEPTH_TEST)

        with ui_overlay():
            label_method.draw()
            label_time.draw()
            label_help.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.M:
            state["method"] = "DQS" if state["method"] == "LBS" else "LBS"
            print(f"[skinning] metodo: {state['method']}")
        elif symbol == pyglet.window.key.E:
            state["show_skeleton"] = not state["show_skeleton"]
        elif symbol == pyglet.window.key.A and model.animations:
            # los nodos que la animacion nueva no mueve vuelven a su pose del archivo
            state["animation"] = (state["animation"] + 1) % len(model.animations)
            state["joint_nodes"] = animated_joint_nodes(model, state["animation"])
            state["time"] = 0.0
            for node in model.nodes:
                graph.nodes[node_to_scenegraph_name[node.index]]["transform"] = node.local_matrix()
            apply_animation_to_graph()
        elif symbol == pyglet.window.key.SPACE:
            state["paused"] = not state["paused"]
        elif symbol == pyglet.window.key.R:
            state["time"] = 0.0
            apply_animation_to_graph()
        elif symbol == pyglet.window.key.COMMA:
            state["paused"] = True
            state["time"] = max(0.0, state["time"] - 0.05)
            apply_animation_to_graph()
        elif symbol == pyglet.window.key.PERIOD:
            state["paused"] = True
            if model.animations:
                duration = model.animations[state["animation"]]["duration"]
                state["time"] = min(duration, state["time"] + 0.05)
            apply_animation_to_graph()
        update_labels()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        elif button == pyglet.window.mouse.MIDDLE:
            arcball.set_state(Arcball.STATE_ZOOM)
        arcball.down((x, y))

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        arcball.set_state(Arcball.STATE_ROTATE)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    def update(dt):
        if not model.animations or state["paused"]:
            return
        state["time"] += dt
        duration = model.animations[state["animation"]]["duration"]
        if duration > 0.0:
            state["time"] = state["time"] % duration
        apply_animation_to_graph()
        update_labels()

    pyglet.clock.schedule_interval(update, 1.0 / 60.0)
    pyglet.app.run()
