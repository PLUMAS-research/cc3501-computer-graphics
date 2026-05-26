"""
Animacion esqueletica rigida con un modelo GLTF.

Demuestra los conceptos previos al skinning: jerarquia de huesos, interpolacion
de keyframes TRS por canal, acumulacion de transformaciones via Scenegraph.
El modelo (un Valkyrie VF-1S Battroid de Macross) no tiene weight blending:
todas las piezas son rigidas, parenteadas a nodos del esqueleto que la
animacion mueve. Es decir, este ejemplo trabaja al nivel de la jerarquia
de transformaciones; los pesos vienen en el ejemplo `skinning`.

Teclas:
    espacio       pausa / reanuda la reproduccion
    , y .         retrocede / avanza 0.1 s manualmente
    r             reset a t = 0
    mouse izq     rotar camara
    mouse der     pan
    rueda         zoom
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from PIL import Image

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.gltf import GltfModel, _compose_trs
from grafica.scenegraph import Scenegraph
from grafica.textures import texture_2D_setup
from grafica.ui import ui_overlay


DEFAULT_GLTF = (
    "assets/macrossrobotech_vf-1s_batt-roid_fightermech/scene.gltf"
)


def _build_mesh_node(positions, normals, uvs, indices, texture_id, n_vertices):
    """Construye el dict de malla que espera el Scenegraph (estructura plana, sin children)."""
    return {
        "object": None,
        "mesh": {
            "n_vertices": n_vertices,
            "texture": texture_id,
            "textures": {},
        },
        "attributes": {
            "position": positions.flatten().astype(np.float32),
            "uv": uvs.flatten().astype(np.float32),
            "normal": normals.flatten().astype(np.float32),
            "color": None,
        },
        "indices": indices.astype(np.uint32).tolist(),
        "GL_TYPE": GL.GL_TRIANGLES,
        "transform": tr.identity(),
        "id": None,
        "children": [],
        "parent": None,
        "has_texture": texture_id is not None,
    }


def _compute_bbox(model, graph, mesh_instance_records):
    """
    Calcula la AABB del modelo en bind pose usando las transformaciones globales
    del Scenegraph y las posiciones crudas de cada primitiva.
    """
    graph.calculate_global_transforms()
    min_pt = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    max_pt = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for record in mesh_instance_records:
        positions = model.meshes[record["mesh_index"]][record["primitive_index"]].positions
        global_transform = graph.global_transforms[record["instance_name"]]
        homogeneous = np.hstack([positions, np.ones((positions.shape[0], 1), dtype=np.float32)])
        world_positions = (global_transform @ homogeneous.T).T[:, :3]
        min_pt = np.minimum(min_pt, world_positions.min(axis=0))
        max_pt = np.maximum(max_pt, world_positions.max(axis=0))
    return min_pt, max_pt


@click.command("animacion_esqueletica", short_help="Animacion esqueletica rigida con un GLTF (mecha Macross VF-1S)")
@click.argument("gltf_path", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--width", type=int, default=1280)
@click.option("--height", type=int, default=800)
def animacion_esqueletica(gltf_path, width, height):
    if gltf_path is None:
        gltf_path = str(Path(__file__).parent.parent.parent / DEFAULT_GLTF)

    print(f"[animacion_esqueletica] cargando {gltf_path}")
    model = GltfModel(gltf_path)
    print(
        f"[animacion_esqueletica] nodos={len(model.nodes)} "
        f"meshes={len(model.meshes)} animaciones={len(model.animations)}"
    )
    if model.animations:
        animation = model.animations[0]
        print(
            f"[animacion_esqueletica] animacion='{animation['name']}' "
            f"duracion={animation['duration']:.2f}s canales={len(animation['channels'])}"
        )

    window = pyglet.window.Window(width=width, height=height, caption="animacion esqueletica")

    GL.glClearColor(0.08, 0.09, 0.12, 1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    model.upload_textures()

    # textura blanca 1x1 como fallback. asi el shader siempre puede muestrear
    # sin necesidad de branch por "has_texture" (las primitivas sin material
    # con baseColorTexture la usan y su base_color_factor decide el color)
    white_pixel = Image.new("RGB", (1, 1), color=(255, 255, 255))
    white_pixel_texture_id = texture_2D_setup(white_pixel)

    graph = Scenegraph("root")
    graph.load_and_register_pipeline(
        "default",
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    print("[animacion_esqueletica] registrando primitivas...")
    skipped_skinned = 0
    for mesh_index, primitives in enumerate(model.meshes):
        for primitive_index, primitive in enumerate(primitives):
            if primitive.joints is not None:
                # las primitivas con weights van en el ejemplo de skinning;
                # aqui no las dibujamos porque tendriamos que aplicar matrices
                # de hueso por vertice y todavia no tenemos shader de skinning
                skipped_skinned += 1
                continue
            mesh_name = f"mesh_{mesh_index}_prim_{primitive_index}"
            if primitive.uvs is None:
                uvs = np.zeros((primitive.positions.shape[0], 2), dtype=np.float32)
            else:
                uvs = primitive.uvs
            if primitive.normals is None:
                normals = np.zeros_like(primitive.positions)
            else:
                normals = primitive.normals
            material = (
                model.materials[primitive.material_index]
                if primitive.material_index is not None
                else None
            )
            texture_id = None
            if material is not None and material["base_color_texture"] is not None:
                texture_id = model.texture_gl_ids[material["base_color_texture"]]
            if texture_id is None:
                texture_id = white_pixel_texture_id
            mesh_node = _build_mesh_node(
                positions=primitive.positions,
                normals=normals,
                uvs=uvs,
                indices=(
                    primitive.indices
                    if primitive.indices is not None
                    else np.arange(primitive.positions.shape[0], dtype=np.uint32)
                ),
                texture_id=texture_id,
                n_vertices=primitive.positions.shape[0],
            )
            graph.register_mesh(mesh_name, mesh_node)
    if skipped_skinned > 0:
        print(
            f"[animacion_esqueletica] {skipped_skinned} primitiva(s) con skinning omitidas "
            "(este ejemplo no aplica weight blending)"
        )

    # un nodo de transformacion por cada nodo GLTF. el nombre lleva el indice
    # para que el match con la animacion sea directo
    node_names = []
    for node in model.nodes:
        scenegraph_name = f"gltf_node_{node.index}"
        graph.add_transform(scenegraph_name, node.local_matrix())
        node_names.append(scenegraph_name)

    mesh_instance_records = []
    for node in model.nodes:
        if node.mesh_index is not None:
            for primitive_index, primitive in enumerate(model.meshes[node.mesh_index]):
                if primitive.joints is not None:
                    continue
                instance_name = f"node_{node.index}_prim_{primitive_index}"
                material = (
                    model.materials[primitive.material_index]
                    if primitive.material_index is not None
                    else None
                )
                if material is not None:
                    base_color_factor = material["base_color_factor"][:3]
                else:
                    base_color_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                graph.add_mesh_instance(
                    instance_name,
                    f"mesh_{node.mesh_index}_prim_{primitive_index}",
                    "default",
                    base_color_factor=base_color_factor.astype(np.float32),
                )
                graph.add_edge(node_names[node.index], instance_name)
                mesh_instance_records.append({
                    "instance_name": instance_name,
                    "mesh_index": node.mesh_index,
                    "primitive_index": primitive_index,
                })

        for child_index in node.children:
            graph.add_edge(node_names[node.index], node_names[child_index])

    for root_index in model.scene_roots:
        graph.add_edge("root", node_names[root_index])

    # bbox del bind pose. Lo usamos para dos cosas: detectar si el modelo
    # asume otro eje como vertical (Sketchfab + Blender suelen exportar Z up),
    # y luego encuadrar la camara
    min_pt, max_pt = _compute_bbox(model, graph, mesh_instance_records)
    extents = max_pt - min_pt
    tallest_axis = int(np.argmax(extents))
    if tallest_axis == 2:
        root_fixup = tr.rotationX(-np.pi / 2)
        print("[animacion_esqueletica] modelo Z-up: rotando -90 sobre X")
    elif tallest_axis == 0:
        root_fixup = tr.rotationZ(np.pi / 2)
        print("[animacion_esqueletica] modelo X-up: rotando +90 sobre Z")
    else:
        root_fixup = tr.identity()
    graph.nodes["root"]["transform"] = root_fixup
    min_pt, max_pt = _compute_bbox(model, graph, mesh_instance_records)
    center = (min_pt + max_pt) * 0.5
    diagonal = float(np.linalg.norm(max_pt - min_pt))
    print(f"[animacion_esqueletica] bbox: min={min_pt} max={max_pt} diagonal={diagonal:.2f}")

    if diagonal <= 0:
        diagonal = 1.0

    camera_offset = np.array([diagonal * 0.8, diagonal * 0.3, diagonal * 0.8], dtype=np.float32)
    initial_eye = center + camera_offset
    initial_view = tr.lookAt(initial_eye, center.astype(np.float32), np.array([0.0, 1.0, 0.0]))

    arcball = Arcball(
        np.linalg.inv(initial_view),
        np.array((width, height), dtype=float),
        diagonal,
        center.astype(np.float64),
    )
    arcball.set_initial_state()
    arcball.set_distance_limits(min_distance=diagonal * 0.2, max_distance=diagonal * 5.0)

    projection = tr.perspective(
        45.0,
        width / height,
        max(diagonal * 0.01, 0.05),
        diagonal * 10.0,
    )
    graph.set_global_attributes(
        projection=projection,
        light_direction=np.array([0.4, 0.7, 0.5], dtype=np.float32),
        ambient_strength=0.35,
    )
    graph.register_view_transform(np.linalg.inv(arcball.pose))

    label_time = pyglet.text.Label(
        text="",
        font_name="Fira Code",
        font_size=12,
        x=12,
        y=height - 24,
        color=(230, 230, 230, 255),
    )
    label_state = pyglet.text.Label(
        text="",
        font_name="Fira Code",
        font_size=11,
        x=12,
        y=height - 44,
        color=(180, 200, 255, 255),
    )

    state = {
        "time": 0.0,
        "paused": False,
        "animation_index": 0 if model.animations else None,
    }

    def apply_animation():
        animation_index = state["animation_index"]
        if animation_index is None:
            return
        overrides = model.sample_animation(animation_index, state["time"])
        for node_index, trs in overrides.items():
            translation = trs.get("translation", model.nodes[node_index].translation)
            rotation = trs.get("rotation", model.nodes[node_index].rotation)
            scale = trs.get("scale", model.nodes[node_index].scale)
            graph.nodes[node_names[node_index]]["transform"] = _compose_trs(
                translation, rotation, scale
            )

    def update_labels():
        if state["animation_index"] is None:
            label_time.text = "sin animacion"
        else:
            duration = model.animations[state["animation_index"]]["duration"]
            label_time.text = (
                f"t = {state['time']:6.2f} / {duration:5.2f} s   "
                f"{'[pausa]' if state['paused'] else '[play] '}"
            )
        label_state.text = ", . avance manual  espacio pausa  r reset  mouse: rotar / pan / zoom"

    apply_animation()
    update_labels()

    @window.event
    def on_draw():
        window.clear()
        graph.views[graph.current_view] = np.linalg.inv(arcball.pose)
        graph.render()

        with ui_overlay():
            label_time.draw()
            label_state.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            state["paused"] = not state["paused"]
        elif symbol == pyglet.window.key.R:
            state["time"] = 0.0
            apply_animation()
        elif symbol == pyglet.window.key.COMMA:
            state["paused"] = True
            state["time"] = max(0.0, state["time"] - 0.1)
            apply_animation()
        elif symbol == pyglet.window.key.PERIOD:
            state["paused"] = True
            if state["animation_index"] is not None:
                duration = model.animations[state["animation_index"]]["duration"]
                state["time"] = min(duration, state["time"] + 0.1)
            apply_animation()
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
        if state["animation_index"] is None or state["paused"]:
            return
        state["time"] += dt
        duration = model.animations[state["animation_index"]]["duration"]
        if duration > 0.0:
            state["time"] = state["time"] % duration
        apply_animation()
        update_labels()

    pyglet.clock.schedule_interval(update, 1.0 / 60.0)
    pyglet.app.run()
