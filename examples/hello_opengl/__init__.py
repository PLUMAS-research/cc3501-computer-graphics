import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

from grafica.utils import load_pipeline


@click.command("hello_opengl", short_help="¡Hola, OpenGL!")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
def hola_opengl(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    # cargamos el conejo de Stanford (formato STL binario)
    bunny = tm.load("assets/Stanford_Bunny.stl")

    # lo centramos en el origen y lo normalizamos para que quepa en [-1, 1]^3
    bunny.apply_translation(-bunny.centroid)
    bunny.apply_scale(2.0 / bunny.scale)

    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    bunny_vertex_list = tm.rendering.mesh_to_vertexlist(bunny)
    bunny_gpu = pipeline.vertex_list_indexed(
        len(bunny_vertex_list[4][1]) // 3, GL.GL_TRIANGLES, bunny_vertex_list[3]
    )
    bunny_gpu.position[:] = bunny_vertex_list[4][1]

    # pipeline simple para dibujar el cuadro que representa el volumen normalizado.
    # sus vértices se especifican directamente en coordenadas NDC reales,
    # por eso no necesita uniforms.
    box_vert_src = """
#version 330
in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""
    box_frag_src = """
#version 330
out vec4 outColor;
void main() {
    outColor = vec4(0.9, 0.8, 0.2, 1.0);
}
"""
    box_vert = pyglet.graphics.shader.Shader(box_vert_src, "vertex")
    box_frag = pyglet.graphics.shader.Shader(box_frag_src, "fragment")
    box_pipeline = pyglet.graphics.shader.ShaderProgram(box_vert, box_frag)

    # el cuadro se dibuja en las coordenadas NDC reales donde termina el volumen
    # simulado, es decir, en ±VIEWPORT_SCALE
    VIEWPORT_SCALE = 0.85
    v = VIEWPORT_SCALE
    box_vertices = np.array(
        [
            -v,
            -v,
            0.0,
            v,
            -v,
            0.0,
            v,
            v,
            0.0,
            -v,
            v,
            0.0,
        ],
        dtype=np.float32,
    )
    box_gpu = box_pipeline.vertex_list(4, GL.GL_LINE_LOOP)
    box_gpu.position[:] = box_vertices

    # estado de la animación e interacción
    state = {
        "angle": 0.0,
        "object_scale": 1.0,
        "clip_enabled": True,
        "rotating": True,
    }

    label = pyglet.text.Label(
        "+/-: escala   C: clipping   ESPACIO: pausa   R: reset",
        font_name="Fira Code",
        font_size=13,
        x=width // 2,
        y=12,
        anchor_x="center",
        anchor_y="bottom",
        color=(200, 200, 200, 255),
    )

    @window.event
    def on_draw():
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glLineWidth(1.0)
        window.clear()

        # cuadro del volumen normalizado simulado
        box_pipeline.use()
        box_gpu.draw(GL.GL_LINE_LOOP)

        # conejo con rotación, escala y clipping simulado
        pipeline.use()
        pipeline["angle"] = state["angle"]
        pipeline["object_scale"] = state["object_scale"]
        pipeline["viewport_scale"] = VIEWPORT_SCALE
        pipeline["clip_enabled"] = 1 if state["clip_enabled"] else 0
        bunny_gpu.draw(GL.GL_TRIANGLES)

        # el label usa renderizado 2D de pyglet; restauramos el modo de polígonos
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        label.draw()

    def on_update(dt):
        if state["rotating"]:
            state["angle"] += dt * 0.8  # ~0.8 rad/s

    pyglet.clock.schedule_interval(on_update, 1 / 60)

    @window.event
    def on_key_press(symbol, modifiers):
        # aumentar escala: el conejo sale del cubo y se ve el clipping
        if symbol in (
            pyglet.window.key.PLUS,
            pyglet.window.key.EQUAL,
            pyglet.window.key.NUM_ADD,
        ):
            state["object_scale"] = min(state["object_scale"] + 0.1, 2.5)
        # reducir escala
        elif symbol in (pyglet.window.key.MINUS, pyglet.window.key.NUM_SUBTRACT):
            state["object_scale"] = max(state["object_scale"] - 0.1, 0.2)
        # activar / desactivar el clipping simulado
        elif symbol == pyglet.window.key.C:
            state["clip_enabled"] = not state["clip_enabled"]
        # pausar / reanudar la rotación
        elif symbol == pyglet.window.key.SPACE:
            state["rotating"] = not state["rotating"]
        # volver al estado inicial
        elif symbol == pyglet.window.key.R:
            state["object_scale"] = 1.0
            state["angle"] = 0.0

    pyglet.app.run()
