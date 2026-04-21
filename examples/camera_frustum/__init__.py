import os
from pathlib import Path
from itertools import chain

import numpy as np
import pyglet
import pyglet.gl as GL
import click

import grafica.transformations as tr
from grafica.utils import load_pipeline
from grafica.ui import ui_overlay
from examples.projection.elementos import stanford_bunny, regular_grid, rectangulo


def compute_frustum_vertices(eye, target, up_vec, fovy_degrees, aspect, near_dist, far_dist):
    """
    Calcula las 8 esquinas del frustum en coordenadas de mundo.
    Devuelve un array plano de 24 floats (8 vértices × 3 componentes).
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_vec)
    right = right / np.linalg.norm(right)
    camera_up = np.cross(right, forward)

    half_fovy = np.radians(fovy_degrees) / 2
    near_half_h = np.tan(half_fovy) * near_dist
    near_half_w = near_half_h * aspect
    far_half_h  = np.tan(half_fovy) * far_dist
    far_half_w  = far_half_h * aspect

    near_center = eye + forward * near_dist
    far_center  = eye + forward * far_dist

    corners = np.array([
        near_center + camera_up * near_half_h - right * near_half_w,  # 0: near sup-izq
        near_center + camera_up * near_half_h + right * near_half_w,  # 1: near sup-der
        near_center - camera_up * near_half_h + right * near_half_w,  # 2: near inf-der
        near_center - camera_up * near_half_h - right * near_half_w,  # 3: near inf-izq
        far_center  + camera_up * far_half_h  - right * far_half_w,   # 4: far sup-izq
        far_center  + camera_up * far_half_h  + right * far_half_w,   # 5: far sup-der
        far_center  - camera_up * far_half_h  + right * far_half_w,   # 6: far inf-der
        far_center  - camera_up * far_half_h  - right * far_half_w,   # 7: far inf-izq
    ], dtype=np.float32)

    return corners.flatten()


# Índices de las 12 aristas del frustum:
# plano near (4), plano far (4), aristas laterales (4)
FRUSTUM_INDICES = [
    0, 1, 1, 2, 2, 3, 3, 0,  # near
    4, 5, 5, 6, 6, 7, 7, 4,  # far
    0, 4, 1, 5, 2, 6, 3, 7,  # laterales
]


@click.command("camera_frustum", short_help="Frustum de cámara con picture-in-picture")
@click.option("--width",  type=int, default=960)
@click.option("--height", type=int, default=720)
def camera_frustum(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(str(
        Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf"
    ))

    # ── Geometría ──────────────────────────────────────────────────────────────
    bg_data    = rectangulo()
    bunny_data = stanford_bunny()
    grid_data  = regular_grid(resolution=25)

    # ── Pipelines ──────────────────────────────────────────────────────────────
    examples_dir = Path(os.path.dirname(__file__)) / ".."
    hello_world_dir = examples_dir / "hello_world"
    fragment_program = hello_world_dir / "fragment_program.glsl"

    bg_pipeline = load_pipeline(
        hello_world_dir / "vertex_program.glsl",
        fragment_program,
    )
    bunny_pipeline = load_pipeline(
        examples_dir / "projection" / "normal_vertex_program.glsl",
        fragment_program,
    )
    flat_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "flat_vertex_program.glsl",
        fragment_program,
    )

    # ── Buffers GPU ────────────────────────────────────────────────────────────
    bg_gpu = bg_pipeline.vertex_list_indexed(
        bg_data["n_vertices"], bg_data["gl_type"], bg_data["indices"]
    )
    bg_gpu.position[:] = bg_data["position"]
    bg_gpu.color[:]    = bg_data["color"]

    bunny_gpu = bunny_pipeline.vertex_list_indexed(
        bunny_data["n_vertices"], bunny_data["gl_type"], bunny_data["indices"]
    )
    bunny_gpu.position[:] = bunny_data["position"]
    bunny_gpu.normal[:]   = bunny_data["normal"]

    grid_gpu = flat_pipeline.vertex_list_indexed(
        grid_data["n_vertices"], grid_data["gl_type"], grid_data["indices"]
    )
    grid_gpu.position[:] = grid_data["position"]

    # El frustum se actualiza cada frame: 8 vértices, 24 índices
    frustum_gpu = flat_pipeline.vertex_list_indexed(
        8, GL.GL_LINES, FRUSTUM_INDICES
    )

    # Marcador de posición de la cámara observada: 3 segmentos de ejes (6 vértices)
    marker_gpu = flat_pipeline.vertex_list(6, GL.GL_LINES)

    # ── Estado de la cámara observada ──────────────────────────────────────────
    observed_fov   = 60.0
    near_dist      = 0.3
    far_dist       = 5.0
    orbit_radius   = 2.0
    orbit_height   = 0.7
    observed_target = np.array([0.0, 0.0, 0.25])
    up_vec          = np.array([0.0, 0.0, 1.0])
    main_aspect     = width / height
    total_time      = 0.0

    def get_observed_eye():
        x = orbit_radius * np.cos(total_time * 0.5)
        y = orbit_radius * np.sin(total_time * 0.5)
        return np.array([x, y, orbit_height])

    # ── Cámara externa (fija) ─────────────────────────────────────────────────
    external_eye    = np.array([4.5, -3.0, 3.5])
    external_target = np.array([0.0,  0.0, 0.25])
    pip_w = width  // 3
    pip_h = height // 3
    pip_aspect = pip_w / pip_h
    external_view       = tr.lookAt(external_eye, external_target, up_vec)
    external_projection = tr.perspective(50, pip_aspect, 0.1, 20.0)

    # Posición y borde del PiP (esquina inferior derecha)
    pip_margin = 12
    pip_x      = width  - pip_w - pip_margin
    pip_y      = pip_margin
    pip_border = 3

    # ── Transformación de la grilla ────────────────────────────────────────────
    # regular_grid genera vértices en [0,1]×[0,1]; la escalamos a [-2,2]×[-2,2]
    grid_transform = tr.scale(4.0, 4.0, 1.0) @ tr.translate(-0.5, -0.5, 0.0)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def set_uniforms(pipeline, model_transform, view, projection):
        pipeline["transform"]  = model_transform.reshape(16, 1, order="F")
        pipeline["view"]       = view.reshape(16, 1, order="F")
        pipeline["projection"] = projection.reshape(16, 1, order="F")

    def draw_matrix(matrix, title, x, y, font_size=11):
        """Dibuja título + matriz 4×4 alineados a la izquierda en (x, y).
        Devuelve la y del último elemento dibujado."""
        line_height = font_size + 5
        pyglet.text.Label(
            title,
            font_name="Fira Code", font_size=font_size,
            x=x, y=y, color=(200, 200, 200, 255),
        ).draw()
        for row_index in range(4):
            row = matrix[row_index]
            text = f"[{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f} {row[3]:7.3f}]"
            pyglet.text.Label(
                text,
                font_name="Fira Code", font_size=font_size,
                x=x, y=y - (row_index + 1) * line_height,
                color=(160, 210, 160, 255),
            ).draw()
        return y - 5 * line_height

    MARKER_SIZE = 0.13

    def update_dynamic_geometry():
        eye = get_observed_eye()

        frustum_gpu.position[:] = compute_frustum_vertices(
            eye, observed_target, up_vec,
            observed_fov, main_aspect, near_dist, far_dist,
        )

        s = MARKER_SIZE
        marker_gpu.position[:] = np.array([
            eye[0]-s, eye[1],   eye[2],
            eye[0]+s, eye[1],   eye[2],
            eye[0],   eye[1]-s, eye[2],
            eye[0],   eye[1]+s, eye[2],
            eye[0],   eye[1],   eye[2]-s,
            eye[0],   eye[1],   eye[2]+s,
        ], dtype=np.float32)

    # Inicializar antes del primer frame
    update_dynamic_geometry()

    # ── Eventos ────────────────────────────────────────────────────────────────
    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal observed_fov, far_dist
        if symbol in (pyglet.window.key.PLUS, pyglet.window.key.EQUAL):
            observed_fov = min(120.0, observed_fov + 5.0)
            update_dynamic_geometry()
        elif symbol == pyglet.window.key.MINUS:
            observed_fov = max(10.0, observed_fov - 5.0)
            update_dynamic_geometry()
        elif symbol == pyglet.window.key.UP:
            far_dist = min(6.0, far_dist + 0.25)
            update_dynamic_geometry()
        elif symbol == pyglet.window.key.DOWN:
            far_dist = max(0.6, far_dist - 0.25)
            update_dynamic_geometry()

    @window.event
    def on_draw():
        eye = get_observed_eye()
        observed_view       = tr.lookAt(eye, observed_target, up_vec)
        observed_projection = tr.perspective(observed_fov, main_aspect, near_dist, far_dist)

        # ── Vista principal (ventana completa) ─────────────────────────────────
        GL.glClearColor(0.06, 0.08, 0.25, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        window.clear()
        GL.glViewport(0, 0, width, height)

        GL.glDisable(GL.GL_DEPTH_TEST)
        bg_pipeline.use()
        bg_gpu.draw(bg_data["gl_type"])

        GL.glEnable(GL.GL_DEPTH_TEST)

        bunny_pipeline.use()
        set_uniforms(bunny_pipeline, tr.identity(), observed_view, observed_projection)
        bunny_gpu.draw(bunny_data["gl_type"])

        flat_pipeline.use()
        flat_pipeline["color"] = (0.32, 0.34, 0.52)
        GL.glLineWidth(1.0)
        set_uniforms(flat_pipeline, grid_transform, observed_view, observed_projection)
        grid_gpu.draw(grid_data["gl_type"])

        # ── Borde del PiP (rectángulo amarillo) ────────────────────────────────
        GL.glScissor(
            pip_x - pip_border, pip_y - pip_border,
            pip_w + 2 * pip_border, pip_h + 2 * pip_border,
        )
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glClearColor(0.9, 0.8, 0.2, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_SCISSOR_TEST)

        # ── Fondo del PiP ──────────────────────────────────────────────────────
        GL.glScissor(pip_x, pip_y, pip_w, pip_h)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glClearColor(0.03, 0.03, 0.06, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_SCISSOR_TEST)

        # ── Contenido del PiP (vista externa) ─────────────────────────────────
        GL.glViewport(pip_x, pip_y, pip_w, pip_h)
        GL.glEnable(GL.GL_DEPTH_TEST)

        bunny_pipeline.use()
        set_uniforms(bunny_pipeline, tr.identity(), external_view, external_projection)
        bunny_gpu.draw(bunny_data["gl_type"])

        flat_pipeline.use()
        flat_pipeline["color"] = (0.40, 0.40, 0.50)
        GL.glLineWidth(1.0)
        set_uniforms(flat_pipeline, grid_transform, external_view, external_projection)
        grid_gpu.draw(grid_data["gl_type"])

        # Frustum de la cámara observada
        flat_pipeline["color"] = (1.0, 0.85, 0.0)
        GL.glLineWidth(2.0)
        set_uniforms(flat_pipeline, tr.identity(), external_view, external_projection)
        frustum_gpu.draw(GL.GL_LINES)

        # Marcador de posición de la cámara (cruz roja)
        flat_pipeline["color"] = (1.0, 0.3, 0.2)
        GL.glLineWidth(3.0)
        set_uniforms(flat_pipeline, tr.identity(), external_view, external_projection)
        marker_gpu.draw(GL.GL_LINES)

        # ── Restaurar viewport completo y dibujar HUD ─────────────────────────
        GL.glViewport(0, 0, width, height)

        with ui_overlay():
            # Parámetros y controles de la cámara observada
            pyglet.text.Label(
                f"FOV: {observed_fov:.0f}°  Far: {far_dist:.2f}",
                font_name="Fira Code", font_size=13,
                x=10, y=height - 26, color=(255, 255, 255, 255),
            ).draw()
            pyglet.text.Label(
                "+/-: FOV     ↑/↓: largo del frustum",
                font_name="Fira Code", font_size=11,
                x=10, y=height - 46, color=(190, 190, 190, 255),
            ).draw()

            # Matrices de vista y proyección de la cámara observada
            view_bottom = draw_matrix(observed_view,       "V =", x=10, y=height - 70)
            draw_matrix(observed_projection, "P =", x=10, y=view_bottom - 8)

            # Etiqueta del PiP
            pyglet.text.Label(
                "Vista externa",
                font_name="Fira Code", font_size=10,
                x=pip_x + 6, y=pip_y + pip_h - 18,
                color=(230, 200, 50, 255),
            ).draw()

    def update(dt, window):
        nonlocal total_time
        total_time += dt
        update_dynamic_geometry()

    pyglet.clock.schedule_interval(update, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
