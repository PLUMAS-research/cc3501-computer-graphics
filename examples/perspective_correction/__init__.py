"""Demostración de corrección por perspectiva en el muestreo de texturas.

Una superficie cuadrada formada por dos triángulos y texturada con un tablero
de ajedrez se muestra inclinada en perspectiva. La tecla SPACE alterna entre
la interpolación correcta (división por w en cada fragmento) y la versión
afín en pantalla (calificador `noperspective` en GLSL), que reproduce el
efecto característico de PSX que se ve en juegos como Ridge Racer.

Controles:
- SPACE: alterna corrección de perspectiva
- D: resalta la diagonal del cuadrado donde se juntan los dos triángulos
- flechas izquierda/derecha: rotan la superficie alrededor del eje vertical
- flechas arriba/abajo: cambian la inclinación del plano
- N/M: disminuye/aumenta el FOV (visualiza la dependencia con el campo de visión)
- R: reinicia parámetros
"""

import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from PIL import Image

import grafica.transformations as tr
from grafica.textures import texture_2D_setup
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline


def _checkerboard_image(size=512, cells=8):
    """Genera una imagen RGB con tablero de ajedrez de dos colores."""
    cell_size = size // cells
    indices = np.arange(size) // cell_size
    pattern = (indices[:, None] + indices[None, :]) % 2
    image_array = np.where(
        pattern[..., None],
        np.array([245, 245, 245], dtype=np.uint8),
        np.array([30, 30, 30], dtype=np.uint8),
    ).astype(np.uint8)
    # marca un par de celdas con colores distintos para reconocer la orientación
    image_array[:cell_size, :cell_size] = np.array([220, 70, 70], dtype=np.uint8)
    image_array[:cell_size, -cell_size:] = np.array([70, 130, 220], dtype=np.uint8)
    image_array[-cell_size:, :cell_size] = np.array([70, 200, 100], dtype=np.uint8)
    image_array[-cell_size:, -cell_size:] = np.array([220, 200, 70], dtype=np.uint8)
    return Image.fromarray(image_array, mode="RGB")


@click.command(
    "perspective_correction",
    short_help="Demuestra el efecto de la corrección por perspectiva en texturas",
)
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
def perspective_correction(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    # cuadrado en el plano XY, formado por dos triángulos:
    #   v3 ---- v2
    #    | \  T1 |
    #    |T0 \   |
    #   v0 ---- v1
    quad_positions = np.array(
        [
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
             1.0,  1.0, 0.0,
            -1.0,  1.0, 0.0,
        ],
        dtype=np.float32,
    )
    quad_texcoords = np.array(
        [
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ],
        dtype=np.float32,
    )
    quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    quad_gpu = pipeline.vertex_list_indexed(4, GL.GL_TRIANGLES, quad_indices)
    quad_gpu.position[:] = quad_positions
    quad_gpu.texcoords[:] = quad_texcoords

    checker_texture = texture_2D_setup(_checkerboard_image(), flip_top_bottom=False)

    initial_state = {
        "perspective_correct": True,
        "show_seam": False,
        "tilt_degrees": 70.0,
        "spin_degrees": 0.0,
        "fov_degrees": 60.0,
    }
    state = dict(initial_state)

    def _build_projection():
        return tr.perspective(
            state["fov_degrees"], width / height, 0.1, 50.0
        )

    view = tr.lookAt(
        np.array([0.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    @window.event
    def on_key_press(symbol, modifiers):
        keys = pyglet.window.key
        if symbol == keys.SPACE:
            state["perspective_correct"] = not state["perspective_correct"]
        elif symbol == keys.D:
            state["show_seam"] = not state["show_seam"]
        elif symbol == keys.LEFT:
            state["spin_degrees"] -= 5.0
        elif symbol == keys.RIGHT:
            state["spin_degrees"] += 5.0
        elif symbol == keys.UP:
            state["tilt_degrees"] = min(85.0, state["tilt_degrees"] + 2.0)
        elif symbol == keys.DOWN:
            state["tilt_degrees"] = max(0.0, state["tilt_degrees"] - 2.0)
        elif symbol == keys.N:
            state["fov_degrees"] = max(15.0, state["fov_degrees"] - 5.0)
        elif symbol == keys.M:
            state["fov_degrees"] = min(120.0, state["fov_degrees"] + 5.0)
        elif symbol == keys.R:
            state.update(initial_state)

    mode_label = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=14,
        x=10,
        y=height - 24,
        color=(245, 245, 245, 255),
    )
    params_label = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=11,
        x=10,
        y=height - 48,
        color=(200, 210, 220, 255),
    )
    hint_label = pyglet.text.Label(
        "SPACE: corrección  ·  D: marcar diagonal  ·  flechas: rotar/inclinar  ·  N/M: FOV  ·  R: reset",
        font_name="Fira Code",
        font_size=10,
        x=10,
        y=10,
        color=(200, 200, 210, 255),
    )

    @window.event
    def on_draw():
        window.clear()
        GL.glClearColor(0.08, 0.08, 0.12, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        transform = tr.matmul([
            tr.rotationY(np.radians(state["spin_degrees"])),
            tr.rotationX(np.radians(-state["tilt_degrees"])),
        ])
        projection = _build_projection()

        pipeline.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, checker_texture)
        pipeline["texture_sampler"] = 0
        pipeline["perspective_correct"] = bool(state["perspective_correct"])
        pipeline["show_seam"] = bool(state["show_seam"])
        pipeline["transform"] = transform.reshape(16, 1, order="F")
        pipeline["view"] = view.reshape(16, 1, order="F")
        pipeline["projection"] = projection.reshape(16, 1, order="F")
        quad_gpu.draw(GL.GL_TRIANGLES)
        pipeline.stop()

        mode_text = (
            "Corrección por perspectiva: ACTIVA  (interpolación correcta)"
            if state["perspective_correct"]
            else "Corrección por perspectiva: DESACTIVADA  (afín en pantalla, estilo PSX)"
        )
        mode_label.text = mode_text
        params_label.text = (
            f"inclinación: {state['tilt_degrees']:5.1f}°   "
            f"giro: {state['spin_degrees']:6.1f}°   "
            f"FOV: {state['fov_degrees']:5.1f}°   "
            f"diagonal: {'visible' if state['show_seam'] else 'oculta'}"
        )

        with ui_overlay():
            mode_label.draw()
            params_label.draw()
            hint_label.draw()

    pyglet.app.run()
