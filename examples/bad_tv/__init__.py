import os
from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL

from grafica.utils import load_pipeline


@click.command("bad_tv", short_help="Distorsión de señal de TV análoga")
@click.option("--width", type=int, default=800)
@click.option("--height", type=int, default=600)
def bad_tv(width, height):
    win = pyglet.window.Window(width, height)

    # generamos una grilla de vértices en lugar de un simple cuadrilátero.
    # necesitamos muchas filas para que la deformación sinusoidal
    # del vertex program sea visible (con solo 4 vértices,
    # la onda no se puede apreciar).
    rows = 80
    cols = 2

    vertices = []
    for i in range(rows):
        y = -1.0 + 2.0 * i / (rows - 1)
        for j in range(cols):
            x = -1.0 + 2.0 * j / (cols - 1)
            vertices.extend([x, y])

    vertices = np.array(vertices, dtype=np.float32)

    # cada par de filas adyacentes forma una franja horizontal
    # compuesta por dos triángulos
    indices = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            tl = i * cols + j
            tr = tl + 1
            bl = (i + 1) * cols + j
            br = bl + 1
            indices.extend([tl, bl, tr, tr, bl, br])

    indices = np.array(indices, dtype=np.uint32)

    n_vertices = rows * cols

    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    gpu_data = pipeline.vertex_list_indexed(n_vertices, GL.GL_TRIANGLES, indices)
    gpu_data.position[:] = vertices

    total_time = 0.0

    @win.event
    def on_draw():
        nonlocal total_time
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        win.clear()
        pipeline["time"] = total_time
        pipeline.use()
        gpu_data.draw(GL.GL_TRIANGLES)

    def update(dt, window):
        nonlocal total_time
        total_time += dt

    pyglet.clock.schedule_interval(update, 1 / 60.0, win)
    pyglet.app.run()
