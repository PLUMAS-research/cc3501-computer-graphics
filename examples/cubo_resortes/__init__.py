"""Cubo de gelatina hecho con resortes (enfoque Lagrangiano, dinamica real).

A diferencia de cubo_gelatina, que tiembla con un efecto procedural (domain
warping de un SDF), aqui la gelatina es una simulacion fisica: un reticulo 3D de
masas unidas por resortes que cae, choca con el piso, rebota y oscila. El jiggle
sale de la dinamica, no de una formula: si subes la rigidez de los resortes el
cubo se vuelve duro; si la bajas, se vuelve blando y se aplasta al caer.

La simulacion (Verlet + relajacion de restricciones, igual que cloth) vive en
jelly.py. Este archivo dibuja: la superficie verde translucida deja ver los
resortes internos, asi se aprecia la estructura que produce la deformacion.

Hace par con cubo_gelatina (mismo objeto, dos enfoques) y con masa_resorte /
cloth (misma fisica de resortes, en 1D y 2D respectivamente).

Controles:
- espacio:   lanza el cubo hacia arriba con un giro (para verlo rebotar).
- , / .:     baja / sube la rigidez de los resortes (blando / duro).
- W:         muestra u oculta los resortes internos.
- arrastra:  orbita la camara (arcball). scroll: zoom.
- R:         reinicia el cubo en reposo.
- ESC:       cierra la ventana.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

from .jelly import CuboGelatina

COLOR_GELATINA = (0.30, 0.85, 0.55)
COLOR_RESORTES = (0.20, 0.45, 0.35)
COLOR_MASAS = (0.85, 0.95, 0.70)
COLOR_PISO = (0.22, 0.24, 0.28)
SUBSTEPS = 5  # pasos de fisica por cuadro (dt fisica = (1/60)/SUBSTEPS).


def _piso_lineas(pipeline, piso_y, extension=4.0, divisiones=16):
    """Grilla de lineas en el piso, como referencia espacial."""
    coords = np.linspace(-extension, extension, divisiones + 1)
    vertices = []
    for c in coords:
        vertices += [-extension, piso_y, c, extension, piso_y, c]
        vertices += [c, piso_y, -extension, c, piso_y, extension]
    vertices = np.array(vertices, dtype=np.float32)
    cantidad = len(vertices) // 3
    lista = pipeline.vertex_list(cantidad, GL.GL_LINES)
    lista.position[:] = vertices
    lista.normal[:] = np.tile([0.0, 1.0, 0.0], cantidad).astype(np.float32)
    return lista


@click.command("cubo_resortes", short_help="Cubo de gelatina con masas y resortes (Verlet)")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=760)
@click.option("--n", type=int, default=7, help="masas por lado del reticulo")
def cubo_resortes(width, height, n):
    window = pyglet.window.Window(width, height, caption="cubo de gelatina con resortes")

    cubo = CuboGelatina(n=n, dt=(1.0 / 60.0) / SUBSTEPS)

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    # superficie (cascara): indices fijos, posiciones y normales por frame.
    indices_superficie = cubo.superficie.ravel()
    superficie = pipeline.vertex_list_indexed(len(cubo.posiciones), GL.GL_TRIANGLES,
                                              indices_superficie)
    # resortes como lineas.
    indices_resortes = np.empty(2 * len(cubo.indice_a), dtype=np.uint32)
    indices_resortes[0::2] = cubo.indice_a
    indices_resortes[1::2] = cubo.indice_b
    resortes = pipeline.vertex_list_indexed(len(cubo.posiciones), GL.GL_LINES, indices_resortes)
    resortes.normal[:] = np.tile([0.0, 1.0, 0.0], len(cubo.posiciones)).astype(np.float32)
    # masas como puntos.
    masas = pipeline.vertex_list(len(cubo.posiciones), GL.GL_POINTS)
    masas.normal[:] = np.tile([0.0, 1.0, 0.0], len(cubo.posiciones)).astype(np.float32)

    piso = _piso_lineas(pipeline, cubo.piso_y)

    state = {"mostrar_resortes": True}

    projection = tr.perspective(45, width / height, 0.1, 30.0)
    view = tr.lookAt(np.array([0, 1.5, 6.5]), np.array([0, -0.3, 0]), np.array([0, 1, 0]))
    arcball = Arcball(np.linalg.inv(view), np.array((width, height), dtype=float), 2.0,
                      np.array([0.0, 0.0, 0.0]))

    panel = (
        InfoPanel(x=14, y_top=height - 22, color=(225, 235, 225, 255),
                  background=(14, 22, 18), background_width=460)
        .add("titulo", size=14)
        .add("rigidez")
        .add("info")
        .footer("espacio lanza   , . rigidez   W resortes   arrastra orbita   R reset")
    )

    def apply_state():
        panel["titulo"] = "gelatina con resortes: Verlet + relajacion de restricciones"
        panel["rigidez"] = f"rigidez de resortes: {cubo.rigidez:.2f}   (bajo: blando; alto: duro)"
        panel["info"] = f"{len(cubo.posiciones)} masas   {len(cubo.indice_a)} resortes"
        print(f"[cubo_resortes] rigidez={cubo.rigidez:.2f}")

    def update(dt):
        for _ in range(SUBSTEPS):
            cubo.paso(cubo.dt)

    pyglet.clock.schedule_interval(update, 1 / 60)

    @window.event
    def on_draw():
        GL.glClearColor(0.10, 0.11, 0.13, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        posiciones = cubo.posiciones.astype(np.float32).ravel()
        superficie.position[:] = posiciones
        superficie.normal[:] = cubo.normales_superficie().ravel()
        resortes.position[:] = posiciones
        masas.position[:] = posiciones

        view_actual = np.linalg.inv(arcball.pose)
        pipeline.use()
        pipeline["projection"] = projection.reshape(16, 1, order="F")
        pipeline["view"] = view_actual.reshape(16, 1, order="F")
        pipeline["transform"] = tr.identity().reshape(16, 1, order="F")

        # piso opaco.
        pipeline["opacidad"] = 1.0
        pipeline["color_base"] = COLOR_PISO
        piso.draw(GL.GL_LINES)

        # resortes y masas opacos, con profundidad, para verlos dentro del cubo.
        if state["mostrar_resortes"]:
            pipeline["color_base"] = COLOR_RESORTES
            resortes.draw(GL.GL_LINES)
            GL.glPointSize(6.0)
            pipeline["color_base"] = COLOR_MASAS
            masas.draw(GL.GL_POINTS)

        # superficie translucida encima: lee profundidad pero no la escribe, asi
        # los resortes internos se ven a traves de la gelatina.
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        pipeline["opacidad"] = 0.45
        pipeline["color_base"] = COLOR_GELATINA
        superficie.draw(GL.GL_TRIANGLES)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        with ui_overlay():
            panel.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        elif button == pyglet.window.mouse.MIDDLE:
            arcball.set_state(Arcball.STATE_ZOOM)
        arcball.down((x, y))

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.SPACE:
            cubo.lanzar()
        elif symbol == key.COMMA:
            cubo.rigidez = max(0.05, cubo.rigidez - 0.05)
        elif symbol == key.PERIOD:
            cubo.rigidez = min(1.0, cubo.rigidez + 0.05)
        elif symbol == key.W:
            state["mostrar_resortes"] = not state["mostrar_resortes"]
        elif symbol == key.R:
            cubo.reiniciar()
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    cubo_resortes()
