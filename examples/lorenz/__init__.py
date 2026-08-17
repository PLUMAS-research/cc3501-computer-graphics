import ctypes
import os
from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL

from grafica.utils import load_pipeline

# Parámetros clásicos del sistema de Lorenz. Con estos valores el sistema es
# caótico y su atractor tiene la forma de alas de mariposa.
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0

# Rango que ocupa el atractor en cada plano de proyección. Lo que caiga fuera
# no se dibuja, igual que en sr_jengibre.
RANGOS = {
    "xz": (-25.0, 25.0, 0.0, 50.0),
    "xy": (-25.0, 25.0, -30.0, 30.0),
    "yz": (-30.0, 30.0, 0.0, 50.0),
}


@click.command("lorenz", short_help="Atractor de Lorenz acumulado en GPU")
@click.option("--width", type=int, default=800)
@click.option("--height", type=int, default=800)
@click.option("--particles", type=int, default=20, help="Número de trayectorias")
@click.option("--steps", type=int, default=200, help="Pasos de integración por frame")
@click.option("--dt", type=float, default=0.004, help="Paso de tiempo de Euler")
@click.option("--plano", type=click.Choice(["xz", "xy", "yz"]), default="xz",
              help="Plano sobre el que se proyecta la trayectoria 3D")
def lorenz(width, height, particles, steps, dt, plano):
    """Atractor de Lorenz integrado en CPU y acumulado en GPU.

    El sistema es continuo, así que hay que integrarlo: cada paso avanza un
    intervalo dt con el método de Euler. Las posiciones visitadas se proyectan
    sobre un plano y se acumulan en una textura, igual que en sr_jengibre.
    """

    # Estado de las trayectorias, todas cerca del mismo punto de partida.
    # La separación inicial es diminuta y aun así terminan en distinta ala.
    x = 0.0 + np.random.uniform(-1e-3, 1e-3, particles)
    y = 1.0 + np.random.uniform(-1e-3, 1e-3, particles)
    z = 1.05 + np.random.uniform(-1e-3, 1e-3, particles)

    x_min, x_max, y_min, y_max = RANGOS[plano]

    def integrate_and_collect(num_steps):
        """Avanza el sistema y retorna los puntos proyectados, en NDC."""
        nonlocal x, y, z

        horizontal = np.empty(num_steps * particles)
        vertical = np.empty(num_steps * particles)
        count = 0

        for _ in range(num_steps):
            # Las tres derivadas se calculan ANTES de actualizar el estado:
            # si se actualizara x primero, dy y dz usarían el valor nuevo y el
            # esquema de integración sería otro.
            dx = SIGMA * (y - x)
            dy = x * (RHO - z) - y
            dz = x * y - BETA * z

            x = x + dx * current_dt
            y = y + dy * current_dt
            z = z + dz * current_dt

            if plano == "xz":
                a, b = x, z
            elif plano == "xy":
                a, b = x, y
            else:
                a, b = y, z

            visible = ((a >= x_min) & (a <= x_max)
                       & (b >= y_min) & (b <= y_max))
            n = int(visible.sum())
            if n > 0:
                horizontal[count:count + n] = a[visible]
                vertical[count:count + n] = b[visible]
                count += n

        if count == 0:
            return np.empty((0, 2), dtype=np.float32)

        ndc_x = 2.0 * (horizontal[:count] - x_min) / (x_max - x_min) - 1.0
        ndc_y = 2.0 * (vertical[:count] - y_min) / (y_max - y_min) - 1.0
        return np.column_stack([ndc_x, ndc_y]).astype(np.float32)

    # `current_dt` lo cambia el teclado, y por eso lo lee integrate_and_collect
    # en cada llamada en vez de recibirlo como parámetro
    current_dt = dt

    win = pyglet.window.Window(width, height, caption=f"Lorenz ({plano})")

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

    vao_points = GL.glGenVertexArrays(1)
    vbo_points = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao_points)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_points)
    pos_loc = GL.glGetAttribLocation(pipeline_points.id, "position")
    GL.glEnableVertexAttribArray(pos_loc)
    GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0,
                             ctypes.c_void_p(0))
    GL.glBindVertexArray(0)

    # --- Pipeline de visualización ---
    pipeline_vis = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "visualization.glsl",
    )

    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    uv = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    gpu_quad = pipeline_vis.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    gpu_quad.position[:] = vertices
    gpu_quad.uv[:] = uv

    paused = False
    exposure = 0.02

    def tick(frame_time):
        if paused:
            return

        points = integrate_and_collect(steps)
        if len(points) == 0:
            return

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

        pipeline_points.use()

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_points)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, points.nbytes, points,
                        GL.GL_STREAM_DRAW)
        GL.glBindVertexArray(vao_points)
        GL.glDrawArrays(GL.GL_POINTS, 0, len(points))
        GL.glBindVertexArray(0)

        GL.glDisable(GL.GL_BLEND)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def limpiar():
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @win.event
    def on_key_press(symbol, modifiers):
        nonlocal paused, current_dt, exposure, x, y, z

        if symbol == pyglet.window.key.SPACE:
            paused = not paused
            print("Pausado" if paused else "Reanudado")
        elif symbol == pyglet.window.key.R:
            limpiar()
            x = 0.0 + np.random.uniform(-1e-3, 1e-3, particles)
            y = 1.0 + np.random.uniform(-1e-3, 1e-3, particles)
            z = 1.05 + np.random.uniform(-1e-3, 1e-3, particles)
            print("Reiniciado")
        elif symbol in (pyglet.window.key.PLUS, pyglet.window.key.EQUAL):
            # Con un dt grande el método de Euler deja de seguir la curva:
            # la trayectoria se abre hacia afuera y termina escapando.
            current_dt *= 2.0
            limpiar()
            print(f"dt: {current_dt:.5f}")
        elif symbol == pyglet.window.key.MINUS:
            current_dt /= 2.0
            limpiar()
            print(f"dt: {current_dt:.5f}")
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

    print(f"Atractor de Lorenz, proyección {plano}")
    print("Controles:")
    print("  ESPACIO: pausar/reanudar")
    print("  R: reiniciar")
    print("  +/-: duplicar o dividir el paso de integración dt")
    print("  ARRIBA/ABAJO: ajustar exposición")
    print(f"  Trayectorias: {particles}, pasos por frame: {steps}, dt: {dt}")

    pyglet.clock.schedule_interval(tick, 1 / 60.0)
    pyglet.app.run()
