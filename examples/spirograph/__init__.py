"""
Espirógrafo interactivo.

Muestra cómo la composición de transformaciones (rotación y traslación)
produce curvas complejas. El mecanismo tiene tres partes:
  1. Un círculo exterior fijo de radio R.
  2. Un círculo interior de radio r que rueda dentro del exterior.
  3. Un lápiz a distancia d del centro del círculo interior.

La posición del lápiz se calcula componiendo cuatro transformaciones en orden:
  rotación_externa @ traslación_al_centro_interior @ rotación_interna @ traslación_al_lápiz

Controles:
  ↑ / ↓    aumentar / disminuir r/R (radio relativo del círculo interior)
  ← / →    disminuir / aumentar d/r (distancia relativa del lápiz)
  Espacio  pausar / reanudar
  C        limpiar traza
  A        mostrar / ocultar el brazo
"""

import os
from collections import deque
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr

# -----------------------------------------------------------------------
# Geometría base
# -----------------------------------------------------------------------

def _circle_vertices(n_points: int = 128) -> np.ndarray:
    """Vértices de un círculo unitario (radio 1, centrado en el origen) en 3D."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    vertices = np.zeros((n_points, 3), dtype=np.float32)
    vertices[:, 0] = np.cos(angles)
    vertices[:, 1] = np.sin(angles)
    return vertices.flatten()


# -----------------------------------------------------------------------
# Comando principal
# -----------------------------------------------------------------------

@click.command("spirograph", short_help="Espirógrafo interactivo con composición de transformaciones")
@click.option("--width", type=int, default=800)
@click.option("--height", type=int, default=800)
def spirograph(width, height):
    window = pyglet.window.Window(width, height, "Espirógrafo")

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    # Radio del círculo exterior fijo, en coordenadas NDC.
    # Todo lo demás se mide como fracción de este radio.
    R_ORBIT = 0.8

    # Cantidad máxima de puntos en la traza.
    # A 60 fps, alcanza para ~66 segundos antes de comenzar a descartar puntos antiguos.
    MAX_TRACE_POINTS = 4000

    CIRCLE_N = 128   # subdivisiones del círculo

    # Estado mutable del programa
    state = {
        'angle': 0.0,      # ángulo acumulado del brazo exterior (radianes)
        'speed': 1.0,      # velocidad angular (rad/s)
        'paused': False,
        'r_ratio': 0.35,   # r / R: radio relativo del círculo interior
        'd_ratio': 0.85,   # d / r: distancia relativa del lápiz al centro del círculo interior
        'show_arm': True,
    }

    # Traza acumulada: descarta puntos antiguos cuando supera MAX_TRACE_POINTS
    trace_points = deque(maxlen=MAX_TRACE_POINTS)
    trace_valid_count = [0]   # cuántos puntos de la GPU son válidos

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    with open(Path(os.path.dirname(__file__)) / "vertex_program.glsl") as f:
        vertex_source = f.read()
    with open(Path(os.path.dirname(__file__)) / "fragment_program.glsl") as f:
        fragment_source = f.read()

    vert_shader = pyglet.graphics.shader.Shader(vertex_source, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source, "fragment")
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)

    # ------------------------------------------------------------------
    # Geometría en GPU
    # ------------------------------------------------------------------

    # Círculo unitario reutilizable para el exterior y el interior
    unit_circle_gpu = pipeline.vertex_list(CIRCLE_N, GL.GL_LINE_LOOP)
    unit_circle_gpu.position[:] = _circle_vertices(CIRCLE_N)

    # Brazo: dos segmentos = cuatro vértices (GL_LINES los interpreta en pares)
    arm_gpu = pipeline.vertex_list(4, GL.GL_LINES)
    arm_gpu.position[:] = np.zeros(12, dtype=np.float32)

    # Punto del lápiz
    pen_dot_gpu = pipeline.vertex_list(1, GL.GL_POINTS)
    pen_dot_gpu.position[:] = np.zeros(3, dtype=np.float32)

    # Buffer de traza preasignado.
    # Los vértices no usados se rellenan con el último punto válido para
    # evitar líneas espurias hacia el origen.
    trace_gpu = pipeline.vertex_list(MAX_TRACE_POINTS, GL.GL_LINE_STRIP)
    trace_gpu.position[:] = np.zeros(MAX_TRACE_POINTS * 3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Cálculo de transformaciones
    # ------------------------------------------------------------------

    def compute_transforms():
        """
        Calcula la posición del lápiz y del centro del círculo interior
        usando composición de transformaciones matriciales.

        El mecanismo se descompone en cuatro transformaciones sucesivas.
        Cada una representa un grado de libertad del espirógrafo:

          T_lápiz = R_ext(angle) @ T(R-r, 0, 0) @ R_int(rolling_angle) @ T(d, 0, 0)

        donde:
          R_ext: rotación del brazo exterior en torno al origen
          T(R-r): traslación al centro del círculo interior (distancia R-r)
          R_int: rotación del círculo interior sobre su propio eje
                 (condición de rodadura: rolling_angle = -angle * R / r,
                  que es el ángulo en el marco del brazo; en el marco fijo
                  el círculo gira a -(R-r)/r, pero el brazo ya gira a +1,
                  así que la velocidad relativa es -(R-r)/r - 1 = -R/r)
          T(d): traslación al lápiz dentro del círculo interior

        Retorna:
          (inner_center_xy, pen_xy, inner_center_transform)
        """
        R = R_ORBIT
        r = state['r_ratio'] * R
        d = state['d_ratio'] * r
        angle = state['angle']

        # Transformación 1: el brazo externo gira alrededor del origen
        outer_rotation = tr.rotationZ(angle)

        # Transformación 2: el centro del círculo interior está a R-r del origen
        arm_translation = tr.translate(R - r, 0, 0)

        # Transformación 3: el círculo interior rueda sin deslizar.
        # En el marco fijo gira a -(R-r)/r por unidad de angle.
        # En el marco del brazo (que ya gira a +1), la velocidad relativa es
        # -(R-r)/r - 1 = -R/r. La transformación actúa en el marco del brazo,
        # por eso se usa -R/r y no -(R-r)/r.
        rolling_angle = -angle * R / r
        inner_rotation = tr.rotationZ(rolling_angle)

        # Transformación 4: el lápiz está a distancia d del centro del círculo interior
        pen_offset = tr.translate(d, 0, 0)

        # Composición: primero se aplica pen_offset, luego inner_rotation, etc.
        # (la multiplicación matricial se lee de derecha a izquierda)
        inner_center_transform = outer_rotation @ arm_translation
        pen_transform = inner_center_transform @ inner_rotation @ pen_offset

        # Aplicamos la transformación al punto de origen en coordenadas homogéneas
        origin_h = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        inner_center_pos = inner_center_transform @ origin_h
        pen_pos = pen_transform @ origin_h

        return inner_center_pos[:2], pen_pos[:2], inner_center_transform

    # ------------------------------------------------------------------
    # Gestión de la traza
    # ------------------------------------------------------------------

    def upload_trace_to_gpu():
        """Sube los puntos de traza acumulados al buffer en GPU."""
        n = len(trace_points)
        if n == 0:
            return
        trace_array = np.array(trace_points, dtype=np.float32)  # (n, 2)
        buffer = np.zeros((MAX_TRACE_POINTS, 3), dtype=np.float32)
        buffer[:n, :2] = trace_array
        # Los vértices sin usar toman el valor del último punto válido.
        # Así GL_LINE_STRIP no genera segmentos hacia el origen.
        buffer[n:, :2] = trace_array[-1]
        trace_gpu.position[:] = buffer.flatten()
        trace_valid_count[0] = n

    def reset_trace():
        trace_points.clear()
        trace_valid_count[0] = 0
        trace_gpu.position[:] = np.zeros(MAX_TRACE_POINTS * 3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Etiqueta de estado
    # ------------------------------------------------------------------

    info_label = pyglet.text.Label(
        '',
        font_name='Fira Code',
        font_size=11,
        multiline=True,
        width=width - 20,
        color=(200, 200, 200, 255),
        x=10,
        y=height - 15,
        anchor_y='top',
    )

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    @window.event
    def on_draw():
        GL.glClearColor(0.10, 0.10, 0.15, 1.0)
        window.clear()

        R = R_ORBIT
        r = state['r_ratio'] * R

        inner_center, pen_pos, inner_center_transform = compute_transforms()

        pipeline.use()

        # Círculo exterior fijo: escalar el círculo unitario a radio R
        pipeline["transform"] = tr.uniformScale(R).reshape(16, 1, order="F")
        pipeline["color"] = (0.35, 0.35, 0.45)
        unit_circle_gpu.draw(GL.GL_LINE_LOOP)

        # Círculo interior rodante: trasladar al centro y escalar a radio r
        transform_inner_circle = inner_center_transform @ tr.scale(r, r, 1.0)
        pipeline["transform"] = transform_inner_circle.reshape(16, 1, order="F")
        pipeline["color"] = (0.25, 0.45, 0.65)
        unit_circle_gpu.draw(GL.GL_LINE_LOOP)

        # Brazo (origen -> centro interior -> lápiz)
        if state['show_arm']:
            arm_gpu.position[:] = np.array([
                0.0,              0.0,              0.0,
                inner_center[0],  inner_center[1],  0.0,
                inner_center[0],  inner_center[1],  0.0,
                pen_pos[0],       pen_pos[1],        0.0,
            ], dtype=np.float32)
            pipeline["transform"] = tr.identity().reshape(16, 1, order="F")
            pipeline["color"] = (0.75, 0.65, 0.20)
            arm_gpu.draw(GL.GL_LINES)

        # Traza acumulada
        if trace_valid_count[0] > 1:
            pipeline["transform"] = tr.identity().reshape(16, 1, order="F")
            pipeline["color"] = (0.90, 0.35, 0.15)
            trace_gpu.draw(GL.GL_LINE_STRIP)

        # Punto del lápiz
        pen_dot_gpu.position[:] = np.array(
            [pen_pos[0], pen_pos[1], 0.0], dtype=np.float32
        )
        pipeline["transform"] = tr.identity().reshape(16, 1, order="F")
        pipeline["color"] = (1.0, 0.60, 0.10)
        GL.glPointSize(7.0)
        pen_dot_gpu.draw(GL.GL_POINTS)

        # UI
        status = "pausado" if state['paused'] else "animando"
        info_label.text = (
            f"r/R = {state['r_ratio']:.2f}   d/r = {state['d_ratio']:.2f}   [{status}]\n"
            f"  ↑↓: r/R     ←→: d/r     Esp: pausa     C: limpiar     A: brazo"
        )
        info_label.draw()

    def update_world(dt, window):
        if state['paused']:
            return

        state['angle'] += state['speed'] * dt

        _, pen_pos, _ = compute_transforms()
        trace_points.append((float(pen_pos[0]), float(pen_pos[1])))
        upload_trace_to_gpu()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            state['paused'] = not state['paused']

        elif symbol == pyglet.window.key.C:
            reset_trace()

        elif symbol == pyglet.window.key.A:
            state['show_arm'] = not state['show_arm']

        elif symbol == pyglet.window.key.UP:
            state['r_ratio'] = min(0.95, round(state['r_ratio'] + 0.05, 2))
            reset_trace()

        elif symbol == pyglet.window.key.DOWN:
            state['r_ratio'] = max(0.05, round(state['r_ratio'] - 0.05, 2))
            reset_trace()

        elif symbol == pyglet.window.key.RIGHT:
            state['d_ratio'] = min(1.50, round(state['d_ratio'] + 0.05, 2))
            reset_trace()

        elif symbol == pyglet.window.key.LEFT:
            state['d_ratio'] = max(0.05, round(state['d_ratio'] - 0.05, 2))
            reset_trace()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
