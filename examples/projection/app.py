import os
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pyglet
import pyglet.gl as GL

import click

import grafica.transformations as tr
from grafica.utils import load_pipeline
from grafica.ui import ui_overlay
# esta vez pusimos todos nuestros elementos en un archivo extra
from .elementos import rectangulo, stanford_bunny, regular_grid

@click.command("projection_example", short_help='Ejemplo de proyección')
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
def projection_example(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(str(
        Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf"
    ))

    # primer elemento: el rectángulo de fondo
    bg_rectangle = rectangulo()

    # reusamos nuestros shaders
    bg_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / ".." / "hello_world" / "vertex_program.glsl", 
        Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl") 

    bg_gpu_data = bg_pipeline.vertex_list_indexed(bg_rectangle['n_vertices'], bg_rectangle['gl_type'], bg_rectangle['indices'])

    bg_gpu_data.position[:] = bg_rectangle['position']
    bg_gpu_data.color[:] = bg_rectangle['color']

    # segundo, el conejo
    bunny = stanford_bunny()

    # cargamos el shader que usaremos para graficar al conejo
    bunny_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "normal_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl")

    bunny_gpu = bunny_pipeline.vertex_list_indexed(
        bunny['n_vertices'], bunny['gl_type'], bunny['indices']
    )
    bunny_gpu.position[:] = bunny['position']
    bunny_gpu.normal[:] = bunny['normal']

    # el tercer elemento es una grilla que graficaremos con GL_LINES (líneas)
    # nuevamente reusamos el fragment program. solo debemos cargar el vertex program
    grid = regular_grid(resolution=20)
    
    grid_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "grid_vertex_program.glsl", 
        Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl")

    
    grid_gpu = grid_pipeline.vertex_list_indexed(
        grid['n_vertices'], grid['gl_type'], grid['indices']
    )
    
    grid_gpu.position[:] = grid['position']

    # agregamos la vista y la proyección a nuestro estado de programa
    total_time = 0.0
    fov = 60.0

    # Vistas y proyecciones
    perspective_view = tr.lookAt(
        np.array([-1.0, 0, 0.25]),
        np.array([0, 0, 0.25]),
        np.array([0.0, 0.0, 1.0]),
    )
    isometric_view = tr.lookAt(
        np.array([-0.7, -0.7, 0.7]),
        np.array([0, 0, 0.25]),
        np.array([0.0, 0.0, 1.0]),
    )
    orthographic_projection = tr.ortho(-0.5, 0.5, -0.5, 0.5, 0.001, 5.0)

    transformations = {
        "bunny": tr.identity(),
        "grid": tr.translate(-0.5, -0.5, 0),
        "view": perspective_view,
        "projection": tr.perspective(fov, width / height, 0.001, 5.0),
        "projection_type": "perspective",
    }

    def _draw_matrix(matrix, title, x, y, font_size=11):
        """Dibuja título + matriz 4×4 alineados a la derecha en (x, y).
        Devuelve la y del último elemento dibujado."""
        line_height = font_size + 5
        pyglet.text.Label(
            title,
            font_name="Fira Code", font_size=font_size,
            x=x, y=y, anchor_x="right",
            color=(200, 200, 200, 255),
        ).draw()
        for row_index in range(4):
            row = matrix[row_index]
            text = f"[{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f} {row[3]:7.3f}]"
            pyglet.text.Label(
                text,
                font_name="Fira Code", font_size=font_size,
                x=x, y=y - (row_index + 1) * line_height,
                anchor_x="right",
                color=(160, 210, 160, 255),
            ).draw()
        return y - 5 * line_height

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal fov
        if symbol == pyglet.window.key.P:
            if transformations["projection_type"] == "perspective":
                transformations["projection"] = orthographic_projection
                transformations["view"] = isometric_view
                transformations["projection_type"] = "isometric"
            else:
                transformations["projection"] = tr.perspective(fov, width / height, 0.001, 5.0)
                transformations["view"] = perspective_view
                transformations["projection_type"] = "perspective"
        elif transformations["projection_type"] == "perspective":
            if symbol in (pyglet.window.key.PLUS, pyglet.window.key.EQUAL):
                fov = min(120.0, fov + 5.0)
                transformations["projection"] = tr.perspective(fov, width / height, 0.001, 5.0)
            elif symbol == pyglet.window.key.MINUS:
                fov = max(10.0, fov - 5.0)
                transformations["projection"] = tr.perspective(fov, width / height, 0.001, 5.0)

    @window.event
    def on_draw():
        GL.glClearColor(0.06, 0.08, 0.25, 1.0)
        GL.glLineWidth(1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        window.clear()

        # desactivamos el test de profundidad porque el fondo es eso, un fondo
        GL.glDisable(GL.GL_DEPTH_TEST)
        bg_pipeline.use()
        bg_gpu_data.draw(bg_rectangle['gl_type'])

        # lo activamos a la hora de graficar nuestra escena
        GL.glEnable(GL.GL_DEPTH_TEST)

        # hora de dibujar al conejo! activamos su shader
        bunny_pipeline.use()

        bunny_pipeline["transform"] = transformations["bunny"].reshape(
            16, 1, order="F"
        )
        # le entregamos los nuevos parámetros al pipeline
        bunny_pipeline["view"] = transformations["view"].reshape(16, 1, order="F")
        bunny_pipeline["projection"] = transformations["projection"].reshape(
            16, 1, order="F"
        )
        bunny_gpu.draw(bunny['gl_type'])

        # ahora la grilla. activamos su shader y le pasamos los parámetros correspondientes
        grid_pipeline.use()
        grid_pipeline["transform"] = transformations["grid"].reshape(
            16, 1, order="F"
        )
        grid_pipeline["view"] = transformations["view"].reshape(16, 1, order="F")
        grid_pipeline["projection"] = transformations["projection"].reshape(
            16, 1, order="F"
        )
        # como dibujaremos líneas y no polígonos, debemos especificarlo en la llamada a draw
        grid_gpu.draw(grid['gl_type'])

        projection_type = transformations["projection_type"]
        if projection_type == "perspective":
            label_text = f"Perspectiva  |  FOV: {fov:.0f}°  |  +/- cambia FOV  |  P para isométrica"
        else:
            label_text = "Isométrica  |  P para perspectiva"
        with ui_overlay():
            pyglet.text.Label(
                label_text,
                font_name="Fira Code", font_size=13,
                x=10, y=14, color=(255, 255, 255, 255),
            ).draw()

            # Matrices de vista y proyección en la esquina superior derecha
            view_bottom = _draw_matrix(transformations["view"],       "V =", x=width - 10, y=height - 22)
            _draw_matrix(transformations["projection"], "P =", x=width - 10, y=view_bottom - 8)

    def update_world(dt, window):
        nonlocal total_time
        total_time += dt

        # actualizamos la transformación del conejo.
        # esta vez respecto al eje Z, es decir, en el "mundo del conejo"
        # y no en las coordenadas de OpenGL :)
        transformations["bunny"] = tr.rotationZ(total_time * 2.0)

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)