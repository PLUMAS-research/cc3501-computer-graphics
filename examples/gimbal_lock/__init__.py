import os
from pathlib import Path

import numpy as np
import pyglet
import pyglet.gl as GL

import click

from grafica.utils import load_pipeline
import grafica.transformations as tr


N_RING_SEGMENTS = 128
ROTATION_SPEED = 1.5   # radianes por segundo al mantener una tecla


def _ring_xy(n: int, radius: float) -> np.ndarray:
    """Puntos de un círculo en el plano XY (perpendicular al eje Z = eje de yaw)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        np.cos(angles) * radius,
        np.sin(angles) * radius,
        np.zeros(n),
    ]).astype(np.float32)


def _ring_xz(n: int, radius: float) -> np.ndarray:
    """Puntos de un círculo en el plano XZ (perpendicular al eje Y = eje de pitch)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        np.cos(angles) * radius,
        np.zeros(n),
        np.sin(angles) * radius,
    ]).astype(np.float32)


def _ring_yz(n: int, radius: float) -> np.ndarray:
    """Puntos de un círculo en el plano YZ (perpendicular al eje X = eje de roll).

    Usamos Ry(90°): (x,y,z) → (z, y, −x), de modo que los puntos
    (cos θ, sin θ, 0) del plano XY pasan a (0, sin θ, −cos θ) en el plano YZ.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        np.zeros(n),
        np.sin(angles) * radius,
        -np.cos(angles) * radius,
    ]).astype(np.float32)


@click.command("gimbal_lock", short_help="Demostración interactiva de Gimbal Lock")
@click.option("--width",  type=int, default=900)
@click.option("--height", type=int, default=900)
def gimbal_lock(width, height):
    """
    Gimbal Lock: un sistema de tres anillos concéntricos (yaw/pitch/roll).

    Cada anillo gira alrededor de un eje; el eje de cada anillo interior es
    solidario al anillo anterior.  Al llevar pitch a ±90°, el eje de roll
    queda alineado con el eje de yaw: se pierde un grado de libertad.

    Controles:
        ←  →   : yaw   (anillo azul, eje Z)
        ↑  ↓   : pitch (anillo verde, eje Y')
        Q  E   : roll  (anillo rojo, eje X'')
        R      : reiniciar todos los ángulos
    """
    window = pyglet.window.Window(width, height, caption="Gimbal Lock")

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    # -----------------------------------------------------------------------
    # Radios de los tres anillos
    # -----------------------------------------------------------------------
    YAW_RADIUS   = 0.88   # anillo exterior — yaw,   eje Z  (azul)
    PITCH_RADIUS = 0.66   # anillo medio    — pitch, eje Y' (verde)
    ROLL_RADIUS  = 0.46   # anillo interior — roll,  eje X'' (rojo)
    AXIS_HALF    = 0.72   # semilongitud de las líneas que marcan el eje de cada anillo

    # -----------------------------------------------------------------------
    # Geometría de los anillos (GL_LINE_LOOP)
    # -----------------------------------------------------------------------
    ring_yaw_gpu = pipeline.vertex_list(N_RING_SEGMENTS, GL.GL_LINE_LOOP)
    ring_yaw_gpu.position[:] = _ring_xy(N_RING_SEGMENTS, YAW_RADIUS).flatten()

    ring_pitch_gpu = pipeline.vertex_list(N_RING_SEGMENTS, GL.GL_LINE_LOOP)
    ring_pitch_gpu.position[:] = _ring_xz(N_RING_SEGMENTS, PITCH_RADIUS).flatten()

    ring_roll_gpu = pipeline.vertex_list(N_RING_SEGMENTS, GL.GL_LINE_LOOP)
    ring_roll_gpu.position[:] = _ring_yz(N_RING_SEGMENTS, ROLL_RADIUS).flatten()

    # -----------------------------------------------------------------------
    # Líneas de eje de cada anillo (GL_LINES): una línea recta a lo largo del
    # eje de rotación del anillo, pasando por el centro.
    #   — eje yaw   : dirección Z en el marco mundo
    #   — eje pitch : dirección Y en el marco post-yaw
    #   — eje roll  : dirección X en el marco post-yaw-pitch
    # Cuando pitch → ±90°, el eje roll queda alineado con el eje yaw (ambos
    # apuntan en Z) y las dos líneas resultan paralelas: Gimbal Lock.
    # -----------------------------------------------------------------------
    axis_yaw_gpu = pipeline.vertex_list(2, GL.GL_LINES)
    axis_yaw_gpu.position[:] = np.array(
        [0, 0, -AXIS_HALF,  0, 0,  AXIS_HALF], dtype=np.float32
    )

    axis_pitch_gpu = pipeline.vertex_list(2, GL.GL_LINES)
    axis_pitch_gpu.position[:] = np.array(
        [0, -AXIS_HALF, 0,  0,  AXIS_HALF, 0], dtype=np.float32
    )

    axis_roll_gpu = pipeline.vertex_list(2, GL.GL_LINES)
    axis_roll_gpu.position[:] = np.array(
        [-AXIS_HALF, 0, 0,   AXIS_HALF, 0, 0], dtype=np.float32
    )

    # -----------------------------------------------------------------------
    # Cruz de coordenadas del objeto (transforma con el gimbal completo)
    # Indica la orientación actual del objeto; muestra visualmente que,
    # en Gimbal Lock, yaw y roll producen la misma rotación.
    # -----------------------------------------------------------------------
    CROSS = 0.30
    cross_x_gpu = pipeline.vertex_list(2, GL.GL_LINES)
    cross_x_gpu.position[:] = np.array([0, 0, 0,  CROSS, 0, 0], dtype=np.float32)

    cross_y_gpu = pipeline.vertex_list(2, GL.GL_LINES)
    cross_y_gpu.position[:] = np.array([0, 0, 0,  0, CROSS, 0], dtype=np.float32)

    cross_z_gpu = pipeline.vertex_list(2, GL.GL_LINES)
    cross_z_gpu.position[:] = np.array([0, 0, 0,  0, 0, CROSS], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Cámara: perspectiva desde arriba-derecha para ver los tres anillos
    # -----------------------------------------------------------------------
    view = tr.lookAt(
        np.array([1.8, 1.4, 2.2], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    projection = tr.perspective(40, width / height, 0.1, 20.0)
    view_projection = projection @ view

    # -----------------------------------------------------------------------
    # Estado
    # -----------------------------------------------------------------------
    state = {
        'yaw':   0.0,
        'pitch': 0.0,
        'roll':  0.0,
        'keys':  set(),
    }

    # -----------------------------------------------------------------------
    # Etiquetas
    # -----------------------------------------------------------------------
    def _label(text, x, y, color=(200, 200, 200, 255), anchor_x='left'):
        label = pyglet.text.Label(
            text, font_name='Fira Code', font_size=13,
            color=color, x=x, y=y, anchor_y='top', anchor_x=anchor_x,
        )
        return label

    label_yaw    = _label('', 10, height - 18,  color=(100, 160, 255, 255))
    label_pitch  = _label('', 10, height - 38,  color=(80,  220, 120, 255))
    label_roll   = _label('', 10, height - 58,  color=(255, 100,  80, 255))
    label_reset  = _label('R : reiniciar', 10, height - 78, color=(120, 120, 120, 255))
    label_lock        = _label('', width // 2, 38,
                               color=(255, 220, 60, 255), anchor_x='center')
    label_lock.anchor_y = 'bottom'
    label_lock_detail = _label('', width // 2, 18,
                               color=(200, 180, 80, 255), anchor_x='center')
    label_lock_detail.anchor_y = 'bottom'

    # -----------------------------------------------------------------------
    # Helper de dibujo
    # -----------------------------------------------------------------------
    def _draw(vertex_list, draw_mode, world_transform, color):
        pipeline["transform"] = (view_projection @ world_transform).reshape(16, 1, order="F")
        pipeline["color"] = color
        vertex_list.draw(draw_mode)

    # -----------------------------------------------------------------------
    # Colores base y colores de bloqueo
    # -----------------------------------------------------------------------
    COLOR_YAW_RING   = (0.25, 0.50, 1.00)
    COLOR_YAW_AXIS   = (0.12, 0.28, 0.65)
    COLOR_PITCH_RING = (0.20, 1.00, 0.45)
    COLOR_PITCH_AXIS = (0.10, 0.55, 0.25)
    COLOR_ROLL_RING  = (1.00, 0.30, 0.20)
    COLOR_ROLL_AXIS  = (0.60, 0.15, 0.10)
    COLOR_LOCK       = (1.00, 0.90, 0.30)   # amarillo cuando hay bloqueo

    @window.event
    def on_draw():
        GL.glClearColor(0.07, 0.07, 0.10, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        yaw   = state['yaw']
        pitch = state['pitch']
        roll  = state['roll']

        # ¿Cerca del Gimbal Lock? |sin(pitch)| → 1 cuando pitch → ±90°
        near_lock = abs(np.sin(pitch)) > 0.96   # dentro de ~16° de ±90°

        # Los ejes de yaw y roll se alinean → mismo color cuando hay bloqueo
        color_yaw_ring  = COLOR_LOCK if near_lock else COLOR_YAW_RING
        color_yaw_axis  = COLOR_LOCK if near_lock else COLOR_YAW_AXIS
        color_roll_ring = COLOR_LOCK if near_lock else COLOR_ROLL_RING
        color_roll_axis = COLOR_LOCK if near_lock else COLOR_ROLL_AXIS

        Rz = tr.rotationZ(yaw)
        Ry = tr.rotationY(pitch)
        Rx = tr.rotationX(roll)

        identity  = tr.identity()
        yaw_only  = Rz
        yaw_pitch = Rz @ Ry
        full      = Rz @ Ry @ Rx

        pipeline.use()

        # Anillo yaw (azul): fijo en el mundo, plano XY
        _draw(ring_yaw_gpu,   GL.GL_LINE_LOOP, identity,  color_yaw_ring)
        _draw(axis_yaw_gpu,   GL.GL_LINES,     identity,  color_yaw_axis)

        # Anillo pitch (verde): hereda el yaw, plano XZ en el marco de yaw
        _draw(ring_pitch_gpu, GL.GL_LINE_LOOP, yaw_only,  COLOR_PITCH_RING)
        _draw(axis_pitch_gpu, GL.GL_LINES,     yaw_only,  COLOR_PITCH_AXIS)

        # Anillo roll (rojo): hereda yaw+pitch, plano YZ en el marco de yaw+pitch
        _draw(ring_roll_gpu,  GL.GL_LINE_LOOP, yaw_pitch, color_roll_ring)
        _draw(axis_roll_gpu,  GL.GL_LINES,     yaw_pitch, color_roll_axis)

        # Cruz del objeto: transformación completa yaw → pitch → roll
        _draw(cross_x_gpu, GL.GL_LINES, full, (1.00, 0.25, 0.25))
        _draw(cross_y_gpu, GL.GL_LINES, full, (0.25, 1.00, 0.25))
        _draw(cross_z_gpu, GL.GL_LINES, full, (0.40, 0.65, 1.00))

        # ----------------------------------------------------------------
        # HUD
        # ----------------------------------------------------------------
        label_yaw.text   = f"Yaw   (← →) : {np.degrees(yaw):+7.1f}°   [eje Z]"
        label_pitch.text = f"Pitch (↑ ↓) : {np.degrees(pitch):+7.1f}°   [eje Y']"
        label_roll.text  = f"Roll  (Q E) : {np.degrees(roll):+7.1f}°   [eje X'']"

        if near_lock:
            label_lock.text        = "GIMBAL LOCK — el eje roll (rojo) se alinea con el eje yaw (azul)"
            label_lock_detail.text = "← → (yaw) y Q E (roll) controlan el mismo eje: se pierde un grado de libertad"
        else:
            label_lock.text        = ""
            label_lock_detail.text = ""

        label_yaw.draw()
        label_pitch.draw()
        label_roll.draw()
        label_reset.draw()
        label_lock.draw()
        label_lock_detail.draw()

    def update_world(dt, window):
        keys  = state['keys']
        delta = ROTATION_SPEED * dt

        if pyglet.window.key.LEFT  in keys:
            state['yaw']   += delta
        if pyglet.window.key.RIGHT in keys:
            state['yaw']   -= delta
        if pyglet.window.key.UP    in keys:
            state['pitch'] += delta
        if pyglet.window.key.DOWN  in keys:
            state['pitch'] -= delta
        if pyglet.window.key.Q     in keys:
            state['roll']  += delta
        if pyglet.window.key.E     in keys:
            state['roll']  -= delta

    @window.event
    def on_key_press(symbol, modifiers):
        state['keys'].add(symbol)
        if symbol == pyglet.window.key.R:
            state['yaw']   = 0.0
            state['pitch'] = 0.0
            state['roll']  = 0.0

    @window.event
    def on_key_release(symbol, modifiers):
        state['keys'].discard(symbol)

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
