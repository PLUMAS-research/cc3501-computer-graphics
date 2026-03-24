import ctypes
import os
from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL

from grafica.utils import load_pipeline


@click.command("sr_jengibre", short_help="Señor Jengibre (acumulación en GPU)")
@click.option("--width", type=int, default=800)
@click.option("--height", type=int, default=800)
@click.option("--particles", type=int, default=20, help="Número de partículas")
@click.option("--steps", type=int, default=200, help="Pasos de iteración por frame")
def sr_jengibre(width, height, particles, steps):
    """Atractor Gingerbreadman con acumulación en GPU.

    La iteración del mapa ocurre en CPU (NumPy), pero los puntos visitados
    se acumulan en la GPU usando GL_POINTS con blending aditivo sobre un
    framebuffer con textura float. Un fragment shader de visualización
    aplica tone mapping y una paleta de color.
    """

    # Estado de las partículas
    px = np.random.uniform(-0.5, 0.5, particles)
    py = np.random.uniform(-0.5, 0.5, particles)

    # Rango del espacio de fase
    x_min, x_max = -5.0, 9.0
    y_min, y_max = -5.0, 9.0

    def iterate_and_collect(num_steps):
        """Itera el mapa de Gingerbreadman y retorna puntos en NDC."""
        nonlocal px, py

        all_x = np.empty(num_steps * particles)
        all_y = np.empty(num_steps * particles)
        count = 0

        for _ in range(num_steps):
            # Un paso del mapa: x' = 1 - y + |x|, y' = x
            new_x = 1 - py + np.abs(px)
            new_y = px.copy()
            px[:] = new_x
            py[:] = new_y

            # Reiniciar partículas que se escapan
            escaped = (np.abs(px) > 20) | (np.abs(py) > 20)
            px[escaped] = np.random.uniform(-0.5, 0.5, escaped.sum())
            py[escaped] = np.random.uniform(-0.5, 0.5, escaped.sum())

            # Filtrar las que están dentro del rango visible
            visible = ((px >= x_min) & (px <= x_max) &
                       (py >= y_min) & (py <= y_max))
            n = visible.sum()
            if n > 0:
                all_x[count:count + n] = px[visible]
                all_y[count:count + n] = py[visible]
                count += n

        if count == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Convertir a NDC [-1, 1]
        ndc_x = 2.0 * (all_x[:count] - x_min) / (x_max - x_min) - 1.0
        ndc_y = 2.0 * (all_y[:count] - y_min) / (y_max - y_min) - 1.0

        return np.column_stack([ndc_x, ndc_y]).astype(np.float32)

    # --- Ventana ---
    win = pyglet.window.Window(width, height, caption="Sr. Jengibre")

    # --- Framebuffer de acumulación con textura float ---
    accum_tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, accum_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_R16F,
        width, height, 0,
        GL.GL_RED, GL.GL_FLOAT, None
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D, accum_tex, 0
    )
    GL.glClearColor(0.0, 0.0, 0.0, 0.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # --- Pipeline de puntos ---
    pipeline_points = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "point_fragment.glsl",
    )

    # VAO y VBO para los puntos
    vao_points = GL.glGenVertexArrays(1)
    vbo_points = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao_points)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_points)
    pos_loc = GL.glGetAttribLocation(pipeline_points.id, "position")
    GL.glEnableVertexAttribArray(pos_loc)
    GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
    GL.glBindVertexArray(0)

    # --- Pipeline de visualización ---
    pipeline_vis = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "visualization.glsl",
    )

    # Cuadrilátero fullscreen
    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    uv = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    gpu_quad = pipeline_vis.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    gpu_quad.position[:] = vertices
    gpu_quad.uv[:] = uv

    # Estado
    paused = False
    current_steps = steps
    exposure = 0.02

    def tick(dt):
        if paused:
            return

        points = iterate_and_collect(current_steps)
        if len(points) == 0:
            return

        # Acumular en el framebuffer con blending aditivo
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

        pipeline_points.use()

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_points)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, points.nbytes, points, GL.GL_STREAM_DRAW
        )
        GL.glBindVertexArray(vao_points)
        GL.glDrawArrays(GL.GL_POINTS, 0, len(points))
        GL.glBindVertexArray(0)

        GL.glDisable(GL.GL_BLEND)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @win.event
    def on_key_press(symbol, modifiers):
        nonlocal paused, current_steps, exposure, px, py

        if symbol == pyglet.window.key.SPACE:
            paused = not paused
            print("Pausado" if paused else "Reanudado")
        elif symbol == pyglet.window.key.R:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            px[:] = np.random.uniform(-0.5, 0.5, particles)
            py[:] = np.random.uniform(-0.5, 0.5, particles)
            print("Reiniciado")
        elif symbol == pyglet.window.key.PLUS or symbol == pyglet.window.key.EQUAL:
            current_steps = min(current_steps * 2, 10000)
            print(f"Pasos por frame: {current_steps}")
        elif symbol == pyglet.window.key.MINUS:
            current_steps = max(current_steps // 2, 1)
            print(f"Pasos por frame: {current_steps}")
        elif symbol == pyglet.window.key.UP:
            exposure *= 2.0
            print(f"Exposición: {exposure:.6f}")
        elif symbol == pyglet.window.key.DOWN:
            exposure /= 2.0
            print(f"Exposición: {exposure:.6f}")

    @win.event
    def on_draw():
        win.clear()
        GL.glViewport(0, 0, width, height)

        pipeline_vis.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, accum_tex)

        sampler_loc = GL.glGetUniformLocation(pipeline_vis.id, "accum_tex")
        if sampler_loc != -1:
            GL.glUniform1i(sampler_loc, 0)

        exposure_loc = GL.glGetUniformLocation(pipeline_vis.id, "exposure")
        if exposure_loc != -1:
            GL.glUniform1f(exposure_loc, exposure)

        gpu_quad.draw(GL.GL_TRIANGLES)

    print("Sr. Jengibre (acumulación en GPU)")
    print("Controles:")
    print("  ESPACIO: pausar/reanudar")
    print("  R: reiniciar")
    print("  +/-: más/menos pasos por frame")
    print("  ARRIBA/ABAJO: ajustar exposición")
    print(f"  Partículas: {particles}, Pasos/frame: {steps}")

    pyglet.clock.schedule_interval(tick, 1 / 60.0)
    pyglet.app.run()
