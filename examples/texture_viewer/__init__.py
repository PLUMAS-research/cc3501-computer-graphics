"""Visor de modelos 3D texturizados con paneles de espacio UV.

La ventana se divide en dos regiones:
- Un viewport principal (izquierda) muestra el modelo 3D controlado por arcball.
- Una columna de viewports (derecha) muestra, por cada imagen de textura usada
  por el modelo, la imagen con la(s) parametrización(es) UV superpuestas como
  líneas. Si varias submallas comparten la misma imagen, sus parametrizaciones
  caen en el mismo panel con colores distintos.
"""

import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.background import GradientBackground
from grafica.scenegraph import Scenegraph
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline


# Paleta para distinguir parametrizaciones que comparten textura
WIREFRAME_PALETTE = np.array(
    [
        [1.00, 0.85, 0.20, 1.0],  # amarillo
        [0.35, 0.90, 1.00, 1.0],  # cyan
        [1.00, 0.45, 0.70, 1.0],  # magenta
        [0.55, 1.00, 0.60, 1.0],  # verde lima
        [1.00, 0.65, 0.35, 1.0],  # naranja
        [0.75, 0.70, 1.00, 1.0],  # lavanda
    ],
    dtype=np.float32,
)


def _build_uv_line_list(child_node, pipeline):
    """Arma un vertex_list de GL_LINES con las aristas de la parametrización UV.

    Para cada triángulo (i0, i1, i2) del nodo se emiten tres aristas:
    (i0,i1), (i1,i2), (i2,i0). Esto duplica índices pero reaprovecha los
    vértices que ya existen en GPU.
    """
    uv_data = np.asarray(child_node["attributes"]["uv"], dtype=np.float32)
    triangle_indices = np.asarray(child_node["indices"], dtype=np.uint32).reshape(-1, 3)

    line_indices = np.empty((triangle_indices.shape[0], 6), dtype=np.uint32)
    line_indices[:, 0] = triangle_indices[:, 0]
    line_indices[:, 1] = triangle_indices[:, 1]
    line_indices[:, 2] = triangle_indices[:, 1]
    line_indices[:, 3] = triangle_indices[:, 2]
    line_indices[:, 4] = triangle_indices[:, 2]
    line_indices[:, 5] = triangle_indices[:, 0]

    n_vertices = uv_data.size // 2
    vertex_list = pipeline.vertex_list_indexed(
        n_vertices, GL.GL_LINES, line_indices.flatten()
    )
    vertex_list.uv[:] = uv_data
    return vertex_list


def _build_panel_quad(pipeline):
    """Cuadrilátero [-1,1]² con UVs [0,1]² para dibujar la textura de fondo."""
    vertex_list = pipeline.vertex_list_indexed(
        4,
        GL.GL_TRIANGLES,
        np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32),
    )
    vertex_list.position[:] = np.array(
        [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32
    )
    vertex_list.uv[:] = np.array(
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32
    )
    return vertex_list


def _material_label(child_node, fallback_index):
    """Nombre legible para una submalla, prefiriendo el material si existe."""
    mesh = child_node.get("object")
    if mesh is not None:
        material = getattr(getattr(mesh, "visual", None), "material", None)
        if material is not None:
            name = getattr(material, "name", None)
            if name:
                return str(name)
    return f"submalla {fallback_index + 1}"


def _image_key(child_node):
    """Llave que identifica la imagen de textura de una submalla.

    Se hashean los bytes de la imagen PIL para que submallas que apuntan al
    mismo archivo caigan en el mismo panel aunque trimesh haya construido
    objetos PIL distintos.
    """
    mesh = child_node.get("object")
    material = getattr(getattr(mesh, "visual", None), "material", None)
    image = getattr(material, "image", None)
    if image is None:
        return id(child_node)
    return hash((image.size, image.mode, image.tobytes()))


def _group_children_by_image(textured_children):
    """Agrupa las submallas que comparten una misma imagen de textura.

    Devuelve una lista de diccionarios, cada uno con la textura GL a usar
    para el fondo del panel y las submallas (con índice original y nombre)
    que parametrizan esa imagen.
    """
    groups = {}
    for child_index, child in enumerate(textured_children):
        key = _image_key(child)
        if key not in groups:
            groups[key] = {
                "texture_id": child["mesh"]["texture"],
                "submeshes": [],
            }
        groups[key]["submeshes"].append(
            {
                "child": child,
                "index": child_index,
                "name": _material_label(child, child_index),
            }
        )
    return list(groups.values())


@click.command(
    "texture_viewer",
    short_help="Visor 3D con paneles laterales del espacio UV",
)
@click.argument("filename", type=str)
@click.option("--width", type=int, default=1280)
@click.option("--height", type=int, default=720)
def texture_viewer(filename, width, height):
    window = pyglet.window.Window(width, height)
    base_path = Path(os.path.dirname(__file__))

    font_path = Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf"
    pyglet.font.add_file(str(font_path))

    # --- escena 3D ----------------------------------------------------------
    graph = Scenegraph("root")
    graph.load_and_register_mesh("object", filename)
    graph.load_and_register_pipeline(
        "model_pipeline",
        base_path / "model_vertex_program.glsl",
        base_path / "model_fragment_program.glsl",
    )
    graph.add_mesh_instance("object", "object", "model_pipeline")
    graph.add_edge("root", "object")

    model_node = graph.meshes["object"]
    textured_children = [c for c in model_node["children"] if c.get("has_texture")]

    if not textured_children:
        raise click.ClickException(
            f"El modelo '{filename}' no tiene submallas con UVs y textura. "
            "Prueba con assets/dice_cube.obj o assets/zorzal.obj."
        )

    # --- pipelines auxiliares para los paneles UV --------------------------
    quad_pipeline = load_pipeline(
        base_path / "quad_vertex_program.glsl",
        base_path / "quad_fragment_program.glsl",
    )
    line_pipeline = load_pipeline(
        base_path / "uv_wireframe_vertex_program.glsl",
        base_path / "uv_wireframe_fragment_program.glsl",
    )

    panel_quad = _build_panel_quad(quad_pipeline)

    # --- agrupación de parametrizaciones por imagen -----------------------
    image_groups = _group_children_by_image(textured_children)

    # --- layout de viewports ------------------------------------------------
    panel_width = max(width // 3, 240)
    main_viewport = (0, 0, width - panel_width, height)
    panel_x = width - panel_width
    panel_height = height // len(image_groups)

    panels = []
    for panel_index, group in enumerate(image_groups):
        panel_y = height - (panel_index + 1) * panel_height
        wireframes = []
        for submesh in group["submeshes"]:
            wireframes.append(
                {
                    "vertex_list": _build_uv_line_list(submesh["child"], line_pipeline),
                    "color": WIREFRAME_PALETTE[submesh["index"] % len(WIREFRAME_PALETTE)],
                    "name": submesh["name"],
                }
            )
        panels.append(
            {
                "viewport": (panel_x, panel_y, panel_width, panel_height),
                "texture_id": group["texture_id"],
                "wireframes": wireframes,
            }
        )

    # --- cámara y arcball ---------------------------------------------------
    # El arcball lleva la pose cámara-a-mundo. Cada frame derivamos la matriz
    # view (mundo-a-cámara) como inv(pose) y la subimos al scenegraph; así
    # no se aplica la traslación dos veces, a diferencia del patrón que fija
    # view y además escribe inv(pose) en el transform del nodo raíz.
    initial_view = tr.lookAt(
        np.array([0.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    projection = tr.perspective(
        45.0, main_viewport[2] / main_viewport[3], 0.1, 50.0
    )
    graph.register_view_transform(initial_view)
    graph.set_global_attributes(projection=projection)

    arcball = Arcball(
        np.linalg.inv(initial_view),
        np.array((main_viewport[2], main_viewport[3]), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    def _mouse_in_main_viewport(mouse_x, mouse_y):
        vx, vy, vw, vh = main_viewport
        return vx <= mouse_x < vx + vw and vy <= mouse_y < vy + vh

    def _to_viewport_coords(mouse_x, mouse_y):
        return (mouse_x - main_viewport[0], mouse_y - main_viewport[1])

    # --- eventos ------------------------------------------------------------
    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if not _mouse_in_main_viewport(x, y):
            return
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        elif button == pyglet.window.mouse.MIDDLE:
            arcball.set_state(Arcball.STATE_ZOOM)
        arcball.down(_to_viewport_coords(x, y))

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag(_to_viewport_coords(x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        if _mouse_in_main_viewport(x, y):
            arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.R:
            arcball.reset()

    # --- overlays -----------------------------------------------------------
    panel_labels = []
    for panel_index, panel in enumerate(panels):
        vx, vy, vw, vh = panel["viewport"]
        header_text = f"textura {panel_index + 1}/{len(panels)}  ·  {len(panel['wireframes'])} parametrización(es)"
        panel_labels.append(
            pyglet.text.Label(
                header_text,
                font_name="Fira Code",
                font_size=11,
                x=vx + 10,
                y=vy + vh - 20,
                color=(240, 240, 240, 255),
            )
        )
        for legend_index, wireframe in enumerate(panel["wireframes"]):
            line_rgba = wireframe["color"]
            legend_color = (
                int(line_rgba[0] * 255),
                int(line_rgba[1] * 255),
                int(line_rgba[2] * 255),
                255,
            )
            panel_labels.append(
                pyglet.text.Label(
                    f"■ {wireframe['name']}",
                    font_name="Fira Code",
                    font_size=10,
                    x=vx + 10,
                    y=vy + vh - 38 - legend_index * 16,
                    color=legend_color,
                )
            )

    hint_label = pyglet.text.Label(
        "arrastra con el mouse (L: rotar, R: pan, M: zoom) | rueda: zoom | R: reset",
        font_name="Fira Code",
        font_size=10,
        x=10,
        y=10,
        color=(200, 200, 210, 255),
    )

    background = GradientBackground()
    background_dim = 0.85

    @window.event
    def on_draw():
        window.clear()

        # ------ fondo de ventana (degradado) ------
        GL.glViewport(0, 0, width, height)
        background.draw()

        # ------ viewport 3D ------
        GL.glViewport(*main_viewport)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        graph.views[graph.current_view] = np.linalg.inv(arcball.pose)
        graph.render()

        # ------ paneles UV ------
        GL.glDisable(GL.GL_DEPTH_TEST)
        for panel in panels:
            GL.glViewport(*panel["viewport"])

            quad_pipeline.use()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, panel["texture_id"])
            quad_pipeline["sampler_tex"] = 0
            quad_pipeline["background_dim"] = background_dim
            panel_quad.draw(GL.GL_TRIANGLES)
            quad_pipeline.stop()

            line_pipeline.use()
            for wireframe in panel["wireframes"]:
                line_pipeline["line_color"] = wireframe["color"]
                wireframe["vertex_list"].draw(GL.GL_LINES)
            line_pipeline.stop()

        # ------ overlay de texto a ventana completa ------
        GL.glViewport(0, 0, width, height)
        with ui_overlay():
            for label in panel_labels:
                label.draw()
            hint_label.draw()

    pyglet.app.run()
