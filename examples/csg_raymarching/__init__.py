"""Ejemplo atomico de CSG: operaciones booleanas sobre solidos con SDF.

Aisla el nucleo de la geometria constructiva de solidos: dos primitivas (una
caja y una esfera) combinadas por una operacion booleana. Las operaciones no
tocan mallas: actuan sobre las funciones de distancia con signo de cada
primitiva, segun las reglas del apunte:

  union        = min(f_a, f_b)
  interseccion = max(f_a, f_b)
  diferencia   = max(f_a, -f_b)

donde f(p) < 0 indica que el punto esta dentro del solido. El solido se dibuja
por raymarching en un fragment program (sphere tracing): un quad cubre la
pantalla y cada pixel lanza un rayo que avanza hasta tocar la superficie
f(p) ~ 0. No hay mallas ni vertices del solido en juego, solo la evaluacion de
la funcion.

La caja queda en el origen y la esfera se desliza en x con , y . para mostrar
como cambia el resultado de cada operacion al separar las primitivas. Cada
parte de la superficie se tinta segun la primitiva que la aporta.

Controles:
- arrastra:  orbita la camara alrededor del solido.
- 1 / 2 / 3: union / interseccion / diferencia.
- , / .:     acerca / aleja la esfera respecto de la caja.
- R:         reinicia camara y separacion.
- ESC:       cierra la ventana.
"""

import math
from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL

from grafica.ui import InfoPanel
from grafica.utils import load_pipeline

OPERACIONES = ["union", "interseccion", "diferencia"]
REGLA = {
    "union": "f(p) = min(f_caja, f_esfera)   (todo lo que esta dentro de alguna)",
    "interseccion": "f(p) = max(f_caja, f_esfera)   (solo lo que esta en ambas)",
    "diferencia": "f(p) = max(f_caja, -f_esfera)   (la caja menos la esfera)",
}
DEFAULTS = {"operacion": 0, "separation": 0.0, "yaw": 0.7, "pitch": 0.5}


@click.command("csg_raymarching", short_help="CSG: operaciones booleanas sobre SDF por raymarching")
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=720)
def csg_raymarching(width, height):
    window = pyglet.window.Window(width, height, caption="CSG por raymarching de SDF")

    state = dict(DEFAULTS)

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    # quad que cubre toda la pantalla en coordenadas NDC.
    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    quad = pipeline.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    quad.position[:] = vertices

    pipeline.use()
    pipeline["resolution"] = (float(width), float(height))

    panel = (
        InfoPanel(x=14, y_top=height - 22, color=(230, 230, 230, 255),
                  background=(20, 22, 30), background_width=470)
        .add("operacion", size=14)
        .add("regla")
        .add("separacion")
        .footer("arrastra orbita   1/2/3 operacion   , . separa esfera   R reset")
    )

    def apply_state():
        """Sube el estado a los uniforms del shader y al panel. Unico propagador."""
        pipeline.use()
        pipeline["operation"] = state["operacion"]
        pipeline["separation"] = state["separation"]
        pipeline["camera_yaw"] = state["yaw"]
        pipeline["camera_pitch"] = state["pitch"]

        nombre = OPERACIONES[state["operacion"]]
        panel["operacion"] = f"operacion: {nombre}   (caja naranja, esfera azul)"
        panel["regla"] = REGLA[nombre]
        panel["separacion"] = f"separacion esfera-caja: {state['separation']:+.2f}"
        print(f"[csg_raymarching] op={nombre} sep={state['separation']:+.2f}")

    @window.event
    def on_draw():
        window.clear()
        pipeline.use()
        quad.draw(GL.GL_TRIANGLES)
        panel.draw()

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        state["yaw"] += dx * 0.01
        # el pitch se limita para no cruzar los polos y voltear la camara.
        state["pitch"] = float(np.clip(state["pitch"] - dy * 0.01, -1.4, 1.4))
        apply_state()

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol in (key._1, key._2, key._3):
            state["operacion"] = {key._1: 0, key._2: 1, key._3: 2}[symbol]
        elif symbol == key.COMMA:
            state["separation"] = max(-1.6, state["separation"] - 0.05)
        elif symbol == key.PERIOD:
            state["separation"] = min(1.6, state["separation"] + 0.05)
        elif symbol == key.R:
            state.update(DEFAULTS)
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    csg_raymarching()
