import pyglet
import pyglet.gl as GL
import numpy as np
import os
from pathlib import Path
import click

from grafica.utils import load_pipeline
from grafica.arcball import Arcball
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_premade import line_node, ring_node
from grafica.ui import ui_overlay
import grafica.transformations as tr


NEAR_PLANE = 0.1
FAR_PLANE  = 5.0


def _arcball_angles(pose):
    """Azimut, elevación y roll extraídos de la pose cámara-a-mundo."""
    eye = pose[:3, 3]
    horizontal_distance = np.sqrt(eye[0] ** 2 + eye[2] ** 2)
    azimuth   = np.degrees(np.arctan2(eye[0], eye[2]))
    elevation = np.degrees(np.arctan2(eye[1], horizontal_distance))
    world_up  = pose[:3, 1]
    roll      = np.degrees(np.arctan2(world_up[0], world_up[1]))
    return azimuth, elevation, roll


def _project_to_viewport_pixel(point_3d, view_matrix, projection_matrix, viewport):
    """Proyecta un punto 3D a coordenadas de píxel dentro de un viewport."""
    clip_position = projection_matrix @ view_matrix @ np.array([*point_3d, 1.0])
    ndc_position  = clip_position[:3] / clip_position[3]
    pixel_x = viewport[0] + (ndc_position[0] + 1) * 0.5 * viewport[2]
    pixel_y = viewport[1] + (ndc_position[1] + 1) * 0.5 * viewport[3]
    return int(pixel_x), int(pixel_y)


@click.command("arcball_example", short_help="Visor interactivo de modelos 3D con control de cámara")
@click.argument("filename", type=str)
@click.option("--width",  type=int, default=960)
@click.option("--height", type=int, default=720)
def arcball_example(filename, width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    graph = Scenegraph("root")
    graph.load_and_register_mesh("object", filename)

    base_path = Path(os.path.dirname(__file__))

    textured_pipeline = load_pipeline(
        base_path / "vertex_program.glsl",
        base_path / "fragment_program.glsl",
    )
    untextured_pipeline = load_pipeline(
        base_path / "vertex_program_notex.glsl",
        base_path / "fragment_program_notex.glsl",
    )
    depth_pipeline = load_pipeline(
        base_path / "z_vertex_program.glsl",
        base_path / "z_fragment_program.glsl",
    )
    axes_pipeline = load_pipeline(
        base_path / "axes_vertex_program.glsl",
        base_path / "axes_fragment_program.glsl",
    )

    has_texture = graph.meshes["object"]["has_texture"]
    main_pipeline = textured_pipeline if has_texture else untextured_pipeline
    pipeline_index = 0
    available_pipelines = [main_pipeline, depth_pipeline]
    graph.register_pipeline("pipeline", main_pipeline)

    # Bounding sphere del modelo. _node_from_file lo normaliza a [-1, 1]^3,
    # así que el radio máximo es sqrt(3). Lo usamos para ajustar el rango de
    # profundidad de la visualización z-buffer al modelo, no al frustum entero.
    MODEL_RADIUS = float(np.sqrt(3.0))

    graph.add_mesh_instance("object", "object", "pipeline")
    graph.add_edge("root", "object")

    # --- Grafo de escena para el gizmo de orientación ---
    # Se renderiza en paralelo al grafo principal, en su propio viewport y con
    # su propia cámara. Una sola línea y un solo anillo se reutilizan para los
    # tres ejes y los tres anillos del gimbal mediante rotaciones por instancia.
    AXIS_LENGTH    = 1.0
    RING_RADIUS    = 0.85
    RING_SEGMENTS  = 128

    color_x = np.array([1.0,  0.25, 0.25], dtype=np.float32)
    color_y = np.array([0.25, 1.0,  0.25], dtype=np.float32)
    color_z = np.array([0.35, 0.60, 1.0],  dtype=np.float32)

    overlay_graph = Scenegraph("root")
    overlay_graph.register_pipeline("axes", axes_pipeline)
    overlay_graph.register_mesh("axis_line", line_node((0, 0, 0), (AXIS_LENGTH, 0, 0)))
    overlay_graph.register_mesh("ring", ring_node(RING_RADIUS, RING_SEGMENTS))

    # Ejes: la línea base apunta hacia +X. Las rotaciones la llevan a Y y Z.
    overlay_graph.add_object("axis_x", "axis_line", "axes", parent="root",
                             transform=tr.identity(), axis_color=color_x)
    overlay_graph.add_object("axis_y", "axis_line", "axes", parent="root",
                             transform=tr.rotationZ(np.pi / 2), axis_color=color_y)
    overlay_graph.add_object("axis_z", "axis_line", "axes", parent="root",
                             transform=tr.rotationY(-np.pi / 2), axis_color=color_z)

    # Anillos: el anillo base vive en el plano XY (roll). Las rotaciones lo
    # llevan a XZ (azimut) y YZ (elevación).
    overlay_graph.add_object("ring_z", "ring", "axes", parent="root",
                             transform=tr.identity(), axis_color=color_z)
    overlay_graph.add_object("ring_y", "ring", "axes", parent="root",
                             transform=tr.rotationX(np.pi / 2), axis_color=color_y)
    overlay_graph.add_object("ring_x", "ring", "axes", parent="root",
                             transform=tr.rotationY(np.pi / 2), axis_color=color_x)

    # --- Viewport del gizmo (esquina superior derecha) ---
    OVERLAY_SIZE = 160
    overlay_viewport   = (width - OVERLAY_SIZE, height - OVERLAY_SIZE, OVERLAY_SIZE, OVERLAY_SIZE)
    overlay_projection = tr.ortho(-1.6, 1.6, -1.6, 1.6, 0.1, 10.0)
    overlay_graph.set_global_attributes(projection=overlay_projection)
    overlay_graph.register_view_transform(tr.identity())

    # --- Cámara ---
    projection = tr.perspective(45, float(width) / float(height), NEAR_PLANE, FAR_PLANE)
    view = tr.lookAt(np.array([0, 0, 2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    graph.register_view_transform(view)
    graph.set_global_attributes(projection=projection, alpha_cutoff=0.0)

    arcball = Arcball(
        np.linalg.inv(view),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    state = {
        "depth_test":  True,
        "transparent": False,
        "wireframe":   False,
        "mouse_mode":  "ninguno",
    }

    # --- HUD ---
    FONT = "FiraCode"

    def _label(x, y, anchor_y="top", fs=12, color=(220, 220, 220, 255)):
        return pyglet.text.Label(
            "", font_name=FONT, font_size=fs, color=color,
            x=x, y=y, anchor_x="left", anchor_y=anchor_y,
        )

    label_eye       = _label(10, height - 10)
    label_look      = _label(10, height - 28)
    label_distance  = _label(10, height - 46)
    label_status    = _label(10, height - 64)
    label_mode      = _label(10, height - 82)
    label_azimuth   = _label(10, height - 106, color=(80,  210, 100, 255))
    label_elevation = _label(10, height - 124, color=(255, 100, 100, 255))
    label_roll      = _label(10, height - 142, color=(90,  150, 255, 255))

    label_instructions_main      = _label(10, 20, anchor_y="bottom", fs=11)
    label_instructions_secondary = _label(10,  4, anchor_y="bottom", fs=11)
    label_instructions_main.text = (
        "[clic izq] Rotar   [clic der] Trasladar   [scroll / clic centro] Zoom   [R] Reset"
    )
    label_instructions_secondary.text = (
        "[W] Wireframe   [ESPACIO] Pipeline z-buffer   [D] Depth test   [B] Transparencia"
    )

    # Etiquetas X/Y/Z del gizmo de esquina, con posición dinámica.
    def _axis_label(text, color):
        return pyglet.text.Label(
            text, font_name=FONT, font_size=11, color=color,
            x=0, y=0, anchor_x="center", anchor_y="center",
        )

    label_gizmo_x = _axis_label("X", (255,  80,  80, 230))
    label_gizmo_y = _axis_label("Y", ( 80, 210,  80, 230))
    label_gizmo_z = _axis_label("Z", ( 90, 150, 255, 230))

    def update_hud(current_view, view_gizmo):
        eye  = arcball.pose[:3, 3]
        look = -arcball.pose[:3, 2]
        camera_distance = arcball.get_camera_distance()
        label_eye.text      = f"Ojo:   ({eye[0]:+.3f}, {eye[1]:+.3f}, {eye[2]:+.3f})"
        label_look.text     = f"Mira:  ({look[0]:+.3f}, {look[1]:+.3f}, {look[2]:+.3f})"
        label_distance.text = f"Dist:  {camera_distance:.4f}"

        pipeline_name = "z-buffer" if pipeline_index else "normal"
        label_status.text = (
            f"Pipeline: {pipeline_name}   Depth: {'ON' if state['depth_test'] else 'OFF'}   "
            f"Transparencia: {'ON' if state['transparent'] else 'OFF'}"
        )
        label_mode.text = f"Mouse: {state['mouse_mode']}"
        azimuth, elevation, roll = _arcball_angles(arcball.pose)
        label_azimuth.text   = f"Azimut (Y):    {azimuth:+8.2f}°"
        label_elevation.text = f"Elevación:     {elevation:+8.2f}°"
        label_roll.text      = f"Roll (Z cám):  {roll:+8.2f}°"

        axis_tip = AXIS_LENGTH + 0.2
        label_gizmo_x.x, label_gizmo_x.y = _project_to_viewport_pixel(
            [axis_tip, 0, 0], view_gizmo, overlay_projection, overlay_viewport)
        label_gizmo_y.x, label_gizmo_y.y = _project_to_viewport_pixel(
            [0, axis_tip, 0], view_gizmo, overlay_projection, overlay_viewport)
        label_gizmo_z.x, label_gizmo_z.y = _project_to_viewport_pixel(
            [0, 0, axis_tip], view_gizmo, overlay_projection, overlay_viewport)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
            state["mouse_mode"] = "rotando"
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
            state["mouse_mode"] = "trasladando"
        elif button == pyglet.window.mouse.MIDDLE:
            arcball.set_state(Arcball.STATE_ZOOM)
            state["mouse_mode"] = "zoom"
        arcball.down((x, y))

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        arcball.set_state(Arcball.STATE_ROTATE)
        state["mouse_mode"] = "ninguno"

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal pipeline_index

        if symbol == pyglet.window.key.SPACE:
            pipeline_index = not pipeline_index
            graph.register_pipeline("pipeline", available_pipelines[pipeline_index])

        elif symbol == pyglet.window.key.D:
            state["depth_test"] = not state["depth_test"]

        elif symbol == pyglet.window.key.B:
            state["transparent"] = not state["transparent"]
            cutoff = 0.1 if state["transparent"] else 0.0
            graph.set_global_attributes(alpha_cutoff=cutoff)

        elif symbol == pyglet.window.key.W:
            state["wireframe"] = not state["wireframe"]

        elif symbol == pyglet.window.key.R:
            arcball.reset()

    @window.event
    def on_draw():
        GL.glClearColor(0.15, 0.15, 0.18, 1.0)

        if state["depth_test"]:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

        if state["transparent"]:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        else:
            GL.glDisable(GL.GL_BLEND)

        if state["wireframe"]:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

        window.clear()

        # ---- Escena principal ----
        GL.glViewport(0, 0, width, height)
        current_view = np.linalg.inv(arcball.pose)
        graph.views[graph.current_view] = current_view

        # Rango de profundidad ajustado al bounding sphere del modelo: así la
        # paleta Gooch del z-buffer cubre toda la superficie y la variación
        # entre la parte más cercana y la más lejana es notoria.
        camera_distance = float(arcball.get_camera_distance())
        depth_near = max(NEAR_PLANE, camera_distance - MODEL_RADIUS)
        depth_far  = camera_distance + MODEL_RADIUS
        graph.set_global_attributes(depth_near=float(depth_near), depth_far=float(depth_far))

        graph.render()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # ---- View para el viewport del gizmo (rotación pura, sin pan) ----
        view_gizmo = np.eye(4, dtype=np.float64)
        view_gizmo[:3, :3] = current_view[:3, :3]
        view_gizmo[2, 3]   = -3.0

        # ---- Gizmo de orientación (ejes + anillos) en su propio scenegraph ----
        GL.glViewport(*overlay_viewport)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1.5)
        overlay_graph.views[overlay_graph.current_view] = view_gizmo
        overlay_graph.render()
        GL.glLineWidth(1.0)

        # ---- Overlay UI ----
        GL.glViewport(0, 0, width, height)
        update_hud(current_view, view_gizmo)

        with ui_overlay():
            label_eye.draw()
            label_look.draw()
            label_distance.draw()
            label_status.draw()
            label_mode.draw()
            label_azimuth.draw()
            label_elevation.draw()
            label_roll.draw()
            label_instructions_main.draw()
            label_instructions_secondary.draw()
            label_gizmo_x.draw()
            label_gizmo_y.draw()
            label_gizmo_z.draw()

    pyglet.app.run()
