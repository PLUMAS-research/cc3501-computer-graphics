import pyglet
import pyglet.gl as GL
import numpy as np
import os
from pathlib import Path
import click

from grafica.utils import load_pipeline
from grafica.arcball import Arcball
from grafica.scenegraph import Scenegraph
from grafica.ui import ui_overlay
import grafica.transformations as tr


NEAR_PLANE = 0.1
FAR_PLANE  = 5.0


def _arcball_angles(pose):
    """Azimut, elevación y roll extraídos de la pose cámara-a-mundo."""
    eye = pose[:3, 3]
    dist_xz  = np.sqrt(eye[0] ** 2 + eye[2] ** 2)
    azimut   = np.degrees(np.arctan2(eye[0], eye[2]))
    elevacion = np.degrees(np.arctan2(eye[1], dist_xz))
    up_world = pose[:3, 1]
    roll = np.degrees(np.arctan2(up_world[0], up_world[1]))
    return azimut, elevacion, roll


def _project_corner(pt_3d, view_m, proj_m, vp):
    """Proyecta un punto 3D a coordenadas de píxel dentro de un viewport."""
    v   = proj_m @ view_m @ np.array([*pt_3d, 1.0])
    ndc = v[:3] / v[3]
    px  = vp[0] + (ndc[0] + 1) * 0.5 * vp[2]
    py  = vp[1] + (ndc[1] + 1) * 0.5 * vp[3]
    return int(px), int(py)


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

    tex_pipeline = load_pipeline(
        base_path / "vertex_program.glsl",
        base_path / "fragment_program.glsl",
    )
    notex_pipeline = load_pipeline(
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
    circle_pipeline = load_pipeline(
        base_path / "circle_vertex_program.glsl",
        base_path / "circle_fragment_program.glsl",
    )

    main_pipeline = tex_pipeline if graph.meshes["object"]["has_texture"] else notex_pipeline
    current_pipeline = 0
    pipelines = [main_pipeline, depth_pipeline]
    graph.register_pipeline("pipeline", main_pipeline)

    graph.add_mesh_instance("object", "object", "pipeline")
    graph.add_edge("root", "object")

    # --- Ejes (escena principal y gizmo de esquina) ---
    AXIS_LENGTH = 1.0
    axis_x_gpu = axes_pipeline.vertex_list(2, GL.GL_LINES)
    axis_x_gpu.position[:] = [0, 0, 0, AXIS_LENGTH, 0, 0]
    axis_y_gpu = axes_pipeline.vertex_list(2, GL.GL_LINES)
    axis_y_gpu.position[:] = [0, 0, 0, 0, AXIS_LENGTH, 0]
    axis_z_gpu = axes_pipeline.vertex_list(2, GL.GL_LINES)
    axis_z_gpu.position[:] = [0, 0, 0, 0, 0, AXIS_LENGTH]

    # --- Anillos del gimbal ---
    N_RING = 128
    R_RING = 0.85
    a_ring = np.linspace(0, 2 * np.pi, N_RING, endpoint=False)

    def _make_ring(axis):
        gpu = axes_pipeline.vertex_list(N_RING, GL.GL_LINE_LOOP)
        pts = np.zeros(N_RING * 3, dtype=np.float32)
        i0, i1 = {"y": (0, 2), "x": (1, 2), "z": (0, 1)}[axis]
        pts[i0::3] = R_RING * np.cos(a_ring)
        pts[i1::3] = R_RING * np.sin(a_ring)
        gpu.position[:] = pts
        return gpu

    ring_y_gpu = _make_ring("y")  # XZ → azimut
    ring_x_gpu = _make_ring("x")  # YZ → elevación
    ring_z_gpu = _make_ring("z")  # XY → roll

    # --- Círculo de la esfera virtual del arcball ---
    N_CIRCLE = 128
    circle_r = 0.3 * min(width, height)
    c_ang    = np.linspace(0, 2 * np.pi, N_CIRCLE, endpoint=False)
    c_pts    = np.empty(N_CIRCLE * 2, dtype=np.float32)
    c_pts[0::2] = width  / 2.0 + circle_r * np.cos(c_ang)
    c_pts[1::2] = height / 2.0 + circle_r * np.sin(c_ang)
    circle_gpu  = circle_pipeline.vertex_list(N_CIRCLE, GL.GL_LINE_LOOP)
    circle_gpu.position[:] = c_pts

    # --- Viewports de esquina ---
    GIZMO_SIZE  = 120
    GIMBAL_SIZE = 140
    gizmo_vp  = (width - GIZMO_SIZE,  height - GIZMO_SIZE, GIZMO_SIZE,  GIZMO_SIZE)
    gimbal_vp = (width - GIMBAL_SIZE, 0,                   GIMBAL_SIZE, GIMBAL_SIZE)
    proj_corner = tr.ortho(-1.4, 1.4, -1.4, 1.4, 0.1, 10.0)

    # --- Cámara ---
    projection = tr.perspective(45, float(width) / float(height), NEAR_PLANE, FAR_PLANE)
    view = tr.lookAt(np.array([0, 0, 2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    graph.register_view_transform(view)
    graph.set_global_attributes(
        projection=projection,
        near_plane=NEAR_PLANE,
        far_plane=FAR_PLANE,
        alpha_cutoff=0.0,
    )

    arcball = Arcball(
        np.linalg.inv(view),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    state = {
        "depth_test":  True,
        "transparent": False,
        "show_circle": False,
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

    label_eye      = _label(10, height - 10)
    label_look     = _label(10, height - 28)
    label_dist     = _label(10, height - 46)
    label_status   = _label(10, height - 64)
    label_mode     = _label(10, height - 82)
    label_azimut   = _label(10, height - 106, color=(80,  210, 100, 255))
    label_elev     = _label(10, height - 124, color=(255, 100, 100, 255))
    label_roll     = _label(10, height - 142, color=(90,  150, 255, 255))

    label_instr1 = _label(10, 20, anchor_y="bottom", fs=11)
    label_instr2 = _label(10,  4, anchor_y="bottom", fs=11)
    label_instr1.text = "[clic izq] Rotar   [clic der] Trasladar   [scroll / clic centro] Zoom   [R] Reset"
    label_instr2.text = "[C] Círculo arcball   [W] Wireframe   [ESPACIO] Pipeline z-buffer   [D] Depth test   [B] Transparencia"

    # Etiquetas X/Y/Z del gizmo de esquina — posición dinámica
    def _axis_label(text, color):
        return pyglet.text.Label(
            text, font_name=FONT, font_size=11, color=color,
            x=0, y=0, anchor_x="center", anchor_y="center",
        )

    label_gx = _axis_label("X", (255,  80,  80, 230))
    label_gy = _axis_label("Y", ( 80, 210,  80, 230))
    label_gz = _axis_label("Z", ( 90, 150, 255, 230))

    def update_hud(current_view, view_gizmo):
        eye  = arcball.pose[:3, 3]
        look = -arcball.pose[:3, 2]
        dist = arcball.get_camera_distance()
        label_eye.text  = f"Ojo:   ({eye[0]:+.3f}, {eye[1]:+.3f}, {eye[2]:+.3f})"
        label_look.text = f"Mira:  ({look[0]:+.3f}, {look[1]:+.3f}, {look[2]:+.3f})"
        label_dist.text = f"Dist:  {dist:.4f}"

        pl_name = "z-buffer" if current_pipeline else "normal"
        label_status.text = (
            f"Pipeline: {pl_name}   Depth: {'ON' if state['depth_test'] else 'OFF'}   "
            f"Transparencia: {'ON' if state['transparent'] else 'OFF'}"
        )
        label_mode.text = (
            f"Círculo: {'ON' if state['show_circle'] else 'OFF'}   "
            f"Mouse: {state['mouse_mode']}"
        )
        az, el, ro = _arcball_angles(arcball.pose)
        label_azimut.text = f"Azimut (Y):    {az:+8.2f}°"
        label_elev.text   = f"Elevación:     {el:+8.2f}°"
        label_roll.text   = f"Roll (Z cám):  {ro:+8.2f}°"

        tip = AXIS_LENGTH + 0.2
        label_gx.x, label_gx.y = _project_corner([tip, 0,   0  ], view_gizmo, proj_corner, gizmo_vp)
        label_gy.x, label_gy.y = _project_corner([0,   tip, 0  ], view_gizmo, proj_corner, gizmo_vp)
        label_gz.x, label_gz.y = _project_corner([0,   0,   tip], view_gizmo, proj_corner, gizmo_vp)

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
        nonlocal current_pipeline

        if symbol == pyglet.window.key.SPACE:
            current_pipeline = not current_pipeline
            graph.register_pipeline("pipeline", pipelines[current_pipeline])

        elif symbol == pyglet.window.key.D:
            state["depth_test"] = not state["depth_test"]

        elif symbol == pyglet.window.key.B:
            state["transparent"] = not state["transparent"]
            cutoff = 0.1 if state["transparent"] else 0.0
            graph.set_global_attributes(alpha_cutoff=cutoff)

        elif symbol == pyglet.window.key.C:
            state["show_circle"] = not state["show_circle"]

        elif symbol == pyglet.window.key.W:
            state["wireframe"] = not state["wireframe"]

        elif symbol == pyglet.window.key.R:
            arcball.reset()

    def _draw_axes(view_m, proj_m, lw=2.0):
        axes_pipeline.use()
        axes_pipeline["view"]       = view_m.reshape(16, 1, order="F")
        axes_pipeline["projection"] = proj_m.reshape(16, 1, order="F")
        GL.glLineWidth(lw)
        axes_pipeline["axis_color"] = (1.0, 0.25, 0.25)
        axis_x_gpu.draw(GL.GL_LINES)
        axes_pipeline["axis_color"] = (0.25, 1.0, 0.25)
        axis_y_gpu.draw(GL.GL_LINES)
        axes_pipeline["axis_color"] = (0.35, 0.60, 1.0)
        axis_z_gpu.draw(GL.GL_LINES)
        GL.glLineWidth(1.0)

    def _draw_rings(view_m, proj_m):
        axes_pipeline.use()
        axes_pipeline["view"]       = view_m.reshape(16, 1, order="F")
        axes_pipeline["projection"] = proj_m.reshape(16, 1, order="F")
        GL.glLineWidth(1.5)
        axes_pipeline["axis_color"] = (1.0, 0.25, 0.25)
        ring_x_gpu.draw(GL.GL_LINE_LOOP)
        axes_pipeline["axis_color"] = (0.25, 1.0, 0.25)
        ring_y_gpu.draw(GL.GL_LINE_LOOP)
        axes_pipeline["axis_color"] = (0.35, 0.60, 1.0)
        ring_z_gpu.draw(GL.GL_LINE_LOOP)
        GL.glLineWidth(1.0)

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
        graph.render()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # ---- View para los viewports de esquina (rotación pura, sin pan) ----
        view_gizmo = np.eye(4, dtype=np.float64)
        view_gizmo[:3, :3] = current_view[:3, :3]
        view_gizmo[2, 3]   = -3.0

        # ---- Gizmo ejes (esquina superior derecha) ----
        GL.glViewport(*gizmo_vp)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        _draw_axes(view_gizmo, proj_corner)

        # ---- Gimbal anillos (esquina inferior derecha) ----
        GL.glViewport(*gimbal_vp)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        _draw_rings(view_gizmo, proj_corner)

        # ---- Overlay UI ----
        GL.glViewport(0, 0, width, height)
        update_hud(current_view, view_gizmo)

        with ui_overlay():
            label_eye.draw()
            label_look.draw()
            label_dist.draw()
            label_status.draw()
            label_mode.draw()
            label_azimut.draw()
            label_elev.draw()
            label_roll.draw()
            label_instr1.draw()
            label_instr2.draw()
            label_gx.draw()
            label_gy.draw()
            label_gz.draw()

            if state["show_circle"]:
                circle_pipeline.use()
                circle_pipeline["resolution"]   = (float(width), float(height))
                circle_pipeline["circle_color"] = (1.0, 1.0, 1.0, 0.35)
                circle_gpu.draw(GL.GL_LINE_LOOP)
                circle_pipeline.stop()

    pyglet.app.run()
