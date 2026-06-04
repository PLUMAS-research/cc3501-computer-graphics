"""Ejemplo entretenido de SDF: una lampara de lava con metaballs.

Toma la union de solidos del CSG y la suaviza. En vez de la union dura
min(f_a, f_b), que deja un borde anguloso donde dos esferas se tocan, usa la
union suave (smooth-min): una mezcla continua que redondea el encuentro en una
franja de ancho k. Con k chico las esferas se tocan con borde marcado; con k
grande se "derriten" en una sola gota. Esa fusion continua es justo el efecto
de una lampara de lava.

Siete esferas suben y bajan con oscilaciones desfasadas (movimiento calculado
en la CPU) y se suben al shader como un arreglo de vec4 (centro + radio). El
fragment program hace raymarching del campo de distancia fusionado y tinta las
gotas con un gradiente calido segun su altura.

Es un ejemplo rico que extiende csg_raymarching: misma tecnica de sphere
tracing, pero cambiando el min duro por smooth_min y animando las primitivas.

Controles:
- , / .:     menos / mas fusion (k de la smooth-min).
- z / x:     mas lento / mas rapido el movimiento.
- espacio:   pausa / reanuda la animacion.
- arrastra:  orbita la camara.
- R:         reinicia k, velocidad y camara.
- ESC:       cierra la ventana.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from pyglet.gl import GLfloat

from grafica.ui import InfoPanel
from grafica.utils import load_pipeline

N_BLOBS = 7  # debe coincidir con el #define del fragment program.
DEFAULTS = {"k": 0.55, "velocidad": 1.0, "yaw": 0.5, "pitch": 0.1}


def _parametros_blobs():
    """Posiciones base, radios y fases de las gotas. Deterministas (semilla fija)."""
    rng = np.random.default_rng(7)
    base_xz = rng.uniform(-0.55, 0.55, size=(N_BLOBS, 2))
    radios = rng.uniform(0.55, 0.85, size=N_BLOBS)
    fases = rng.uniform(0.0, 2.0 * np.pi, size=N_BLOBS)
    velocidades = rng.uniform(0.6, 1.1, size=N_BLOBS)
    amplitudes = rng.uniform(1.0, 1.6, size=N_BLOBS)
    return base_xz, radios, fases, velocidades, amplitudes


@click.command("metaballs", short_help="Lampara de lava con metaballs (union suave de SDF)")
@click.option("--width", type=int, default=720)
@click.option("--height", type=int, default=820)
def metaballs(width, height):
    window = pyglet.window.Window(width, height, caption="lampara de lava (metaballs)")

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    quad = pipeline.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    quad.position[:] = vertices

    pipeline.use()
    pipeline["resolution"] = (float(width), float(height))

    # buffer ctypes para el uniform vec4 blobs[N_BLOBS], con una vista numpy
    # sobre el mismo bloque para reescribirlo cada frame sin reasignar.
    BlobBuffer = (GLfloat * 4) * N_BLOBS
    blob_buffer = BlobBuffer()
    blob_view = np.frombuffer(blob_buffer, dtype=np.float32).reshape(N_BLOBS, 4)

    base_xz, radios, fases, velocidades, amplitudes = _parametros_blobs()
    state = dict(DEFAULTS)
    state["tiempo"] = 0.0
    state["animar"] = True

    panel = (
        InfoPanel(x=14, y_top=height - 22, color=(235, 225, 215, 255),
                  background=(25, 12, 16), background_width=430)
        .add("titulo", size=14)
        .add("fusion")
        .add("velocidad")
        .footer(", . fusion   z x velocidad   espacio pausa   arrastra orbita   R reset")
    )

    def actualizar_blobs():
        """Mueve las gotas y vuelca centro + radio al buffer del shader."""
        t = state["tiempo"]
        blob_view[:, 0] = base_xz[:, 0] + 0.25 * np.sin(0.5 * velocidades * t + fases)
        blob_view[:, 1] = amplitudes * np.sin(velocidades * t + fases)
        blob_view[:, 2] = base_xz[:, 1] + 0.25 * np.cos(0.4 * velocidades * t + fases)
        blob_view[:, 3] = radios * (1.0 + 0.12 * np.sin(1.3 * velocidades * t + fases))
        pipeline.use()
        pipeline["blobs"] = blob_buffer

    def apply_state():
        pipeline.use()
        pipeline["smooth_k"] = state["k"]
        pipeline["camera_yaw"] = state["yaw"]
        pipeline["camera_pitch"] = state["pitch"]
        panel["titulo"] = "metaballs: union suave smin(a, b, k) de 7 esferas"
        panel["fusion"] = f"fusion k = {state['k']:.2f}   (k chico: borde marcado; k grande: se derriten)"
        panel["velocidad"] = f"velocidad: {state['velocidad']:.2f}x   {'(pausa)' if not state['animar'] else ''}"
        print(f"[metaballs] k={state['k']:.2f} vel={state['velocidad']:.2f} animar={state['animar']}")

    def update(dt):
        if state["animar"]:
            state["tiempo"] += dt * state["velocidad"]
        actualizar_blobs()

    pyglet.clock.schedule_interval(update, 1 / 60)

    @window.event
    def on_draw():
        window.clear()
        pipeline.use()
        quad.draw(GL.GL_TRIANGLES)
        panel.draw()

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        state["yaw"] += dx * 0.01
        state["pitch"] = float(np.clip(state["pitch"] - dy * 0.01, -1.2, 1.2))
        apply_state()

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.COMMA:
            state["k"] = max(0.05, state["k"] - 0.05)
        elif symbol == key.PERIOD:
            state["k"] = min(1.5, state["k"] + 0.05)
        elif symbol == key.Z:
            state["velocidad"] = max(0.0, state["velocidad"] - 0.1)
        elif symbol == key.X:
            state["velocidad"] = min(3.0, state["velocidad"] + 0.1)
        elif symbol == key.SPACE:
            state["animar"] = not state["animar"]
        elif symbol == key.R:
            state.update(DEFAULTS)
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    actualizar_blobs()
    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    metaballs()
