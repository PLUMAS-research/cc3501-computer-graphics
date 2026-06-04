"""Ejemplo entretenido de SDF: un cubo de gelatina acuoso y semitransparente.

Un solido implicito (caja redondeada por SDF) dibujado por raymarching con
apariencia de gelatina mojada. La transparencia se construye fisicamente, no
con alpha: el rayo entra al cubo, se refracta, cruza el interior acumulando
absorcion de color (Beer-Lambert: a mas espesor, mas verde) y sale por la cara
trasera hacia el ambiente. Encima se mezcla un reflejo segun el angulo (Fresnel:
los bordes reflejan, el centro deja ver a traves). Un piso a cuadros bajo el
cubo hace visible la distorsion de la refraccion.

El temblor de gelatina es una deformacion del dominio: antes de evaluar la caja
se desplaza el punto con una suma de senos dependiente del tiempo, asi la
superficie ondula sin cambiar la primitiva.

Es un ejemplo rico que extiende csg_raymarching y metaballs: mismo sphere
tracing de un SDF, pero el sombreado modela refraccion, absorcion y Fresnel.

Controles:
- , / .:     menos / mas temblor de gelatina.
- z / x:     baja / sube el indice de refraccion (1.0 invisible, ~1.5 vidrio).
- espacio:   pausa / reanuda el temblor.
- arrastra:  orbita la camara.
- R:         reinicia temblor, refraccion y camara.
- ESC:       cierra la ventana.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

from grafica.ui import InfoPanel
from grafica.utils import load_pipeline

DEFAULTS = {"wobble": 0.06, "ior": 1.33, "yaw": 0.6, "pitch": 0.35}


@click.command("cubo_gelatina", short_help="Cubo de gelatina acuoso (SDF con refraccion)")
@click.option("--width", type=int, default=820)
@click.option("--height", type=int, default=820)
def cubo_gelatina(width, height):
    window = pyglet.window.Window(width, height, caption="cubo de gelatina")

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

    state = dict(DEFAULTS)
    state["tiempo"] = 0.0
    state["animar"] = True

    panel = (
        InfoPanel(x=14, y_top=height - 22, color=(225, 235, 230, 255),
                  background=(12, 22, 22), background_width=440)
        .add("titulo", size=14)
        .add("temblor")
        .add("refraccion")
        .footer(", . temblor   z x refraccion   espacio pausa   arrastra orbita   R reset")
    )

    def apply_state():
        pipeline.use()
        pipeline["wobble"] = state["wobble"]
        pipeline["indice_refraccion"] = state["ior"]
        pipeline["camera_yaw"] = state["yaw"]
        pipeline["camera_pitch"] = state["pitch"]
        panel["titulo"] = "cubo de gelatina: SDF con refraccion + absorcion + Fresnel"
        panel["temblor"] = f"temblor de gelatina: {state['wobble']:.3f}   {'(pausa)' if not state['animar'] else ''}"
        panel["refraccion"] = f"indice de refraccion: {state['ior']:.2f}   (1.0 invisible, 1.33 agua, 1.5 vidrio)"
        print(f"[cubo_gelatina] wobble={state['wobble']:.3f} ior={state['ior']:.2f} animar={state['animar']}")

    def update(dt):
        if state["animar"]:
            state["tiempo"] += dt
        pipeline.use()
        pipeline["tiempo"] = state["tiempo"]

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
        state["pitch"] = float(np.clip(state["pitch"] - dy * 0.01, -1.3, 1.3))
        apply_state()

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.COMMA:
            state["wobble"] = max(0.0, state["wobble"] - 0.01)
        elif symbol == key.PERIOD:
            state["wobble"] = min(0.25, state["wobble"] + 0.01)
        elif symbol == key.Z:
            state["ior"] = max(1.0, state["ior"] - 0.02)
        elif symbol == key.X:
            state["ior"] = min(2.0, state["ior"] + 0.02)
        elif symbol == key.SPACE:
            state["animar"] = not state["animar"]
        elif symbol == key.R:
            state.update(DEFAULTS)
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    cubo_gelatina()
