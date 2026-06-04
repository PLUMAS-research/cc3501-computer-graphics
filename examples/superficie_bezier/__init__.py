"""Ejemplo rico de superficies de Bezier: un patch bicubico de 4x4 puntos.

Extiende la curva de Bezier cubica a una superficie con dos parametros. En vez
de cuatro puntos de control en una secuencia, ahora hay una grilla de 4x4 que
define el patch:

  S(u,v) = sum_i sum_j P_ij B_i(u) B_j(v),   (u,v) en [0,1] x [0,1]

donde B_i son los polinomios de Bernstein cubicos. La superficie pasa por las
cuatro esquinas de la grilla y queda contenida en la envoltura convexa de los
16 puntos. El alumno levanta o baja un punto de control y ve deformarse la
superficie: el punto atrae la superficie sin que esta lo toque (salvo en las
esquinas).

La superficie se evalua en una grilla fina y se sube como malla con normales
calculadas de las derivadas parciales analiticas dS/du x dS/dv. El wireframe
muestra las curvas isoparametricas (u constante y v constante), que son las
curvas de Bezier de la figura del apunte.

Controles:
- flechas:   mueven la seleccion entre los 16 puntos de control.
- + / -:     suben / bajan el punto de control seleccionado.
- W:         muestra u oculta el wireframe (curvas isoparametricas).
- N:         muestra u oculta la grilla de control.
- arrastrar: rota la superficie (arcball). scroll: zoom.
- R:         reinicia los puntos de control y la camara.
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

COLOR_SUPERFICIE = (0.55, 0.78, 0.95)  # celeste.
COLOR_WIRE = (0.10, 0.16, 0.22)
COLOR_NET = (0.55, 0.55, 0.60)         # grilla de control.
COLOR_PUNTOS = (0.95, 0.80, 0.35)      # puntos de control.
COLOR_SELECCION = (0.95, 0.45, 0.30)   # punto seleccionado.
RESOLUCION = 24  # muestras por lado en la evaluacion de la superficie.


def _puntos_iniciales():
    """Grilla 4x4 plana en el plano xz con un punto central levantado."""
    pasos = np.linspace(-1.2, 1.2, 4)
    puntos = np.zeros((4, 4, 3), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            puntos[i, j] = (pasos[i], 0.0, pasos[j])
    # se levantan los cuatro puntos interiores para formar una cupula inicial.
    puntos[1:3, 1:3, 1] = 1.3
    return puntos


def _bernstein(t):
    """Polinomios de Bernstein cubicos B_i(t) evaluados en cada t. t: (R,)."""
    un_menos = 1.0 - t
    return np.stack([
        un_menos**3,
        3.0 * t * un_menos**2,
        3.0 * t**2 * un_menos,
        t**3,
    ], axis=1)  # (R, 4)


def _bernstein_derivada(t):
    """Derivadas dB_i/dt de los polinomios de Bernstein cubicos."""
    un_menos = 1.0 - t
    return np.stack([
        -3.0 * un_menos**2,
        3.0 * un_menos**2 - 6.0 * t * un_menos,
        6.0 * t * un_menos - 3.0 * t**2,
        3.0 * t**2,
    ], axis=1)  # (R, 4)


def _evaluar_superficie(puntos, resolucion):
    """Evalua S(u,v) y sus normales en una grilla resolucion x resolucion.

    Devuelve (posiciones, normales) con forma (R, R, 3). Las normales salen del
    producto cruz de las derivadas parciales dS/du x dS/dv.
    """
    t = np.linspace(0.0, 1.0, resolucion)
    base_u, base_v = _bernstein(t), _bernstein(t)
    derivada_u, derivada_v = _bernstein_derivada(t), _bernstein_derivada(t)

    posiciones = np.einsum("ai,bj,ijc->abc", base_u, base_v, puntos)
    parcial_u = np.einsum("ai,bj,ijc->abc", derivada_u, base_v, puntos)
    parcial_v = np.einsum("ai,bj,ijc->abc", base_u, derivada_v, puntos)

    normales = np.cross(parcial_u, parcial_v)
    largo = np.linalg.norm(normales, axis=2, keepdims=True)
    largo[largo == 0] = 1.0
    return posiciones, normales / largo


def _indices_grilla(resolucion):
    """Indices de triangulos de una grilla regular resolucion x resolucion."""
    indice = lambda a, b: a * resolucion + b
    caras = []
    for a in range(resolucion - 1):
        for b in range(resolucion - 1):
            caras.append((indice(a, b), indice(a + 1, b), indice(a + 1, b + 1)))
            caras.append((indice(a, b), indice(a + 1, b + 1), indice(a, b + 1)))
    return np.array(caras, dtype=np.uint32)


@click.command("superficie_bezier", short_help="Patch bicubico de Bezier (4x4 puntos de control)")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
def superficie_bezier(width, height):
    window = pyglet.window.Window(width, height, caption="superficie de Bezier")

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    indices = _indices_grilla(RESOLUCION).ravel()
    superficie = pipeline.vertex_list_indexed(RESOLUCION * RESOLUCION, GL.GL_TRIANGLES, indices)

    # grilla de control: lineas entre puntos vecinos en filas y columnas.
    aristas_net = []
    for i in range(4):
        for j in range(4):
            if i < 3:
                aristas_net.append((i * 4 + j, (i + 1) * 4 + j))
            if j < 3:
                aristas_net.append((i * 4 + j, i * 4 + (j + 1)))
    indices_net = np.array(aristas_net, dtype=np.uint32).ravel()
    net = pipeline.vertex_list_indexed(16, GL.GL_LINES, indices_net)
    net.normal[:] = np.tile([0.0, 1.0, 0.0], 16).astype(np.float32)
    puntos_gpu = pipeline.vertex_list(16, GL.GL_POINTS)
    puntos_gpu.normal[:] = np.tile([0.0, 1.0, 0.0], 16).astype(np.float32)

    state = {"puntos": _puntos_iniciales(), "sel_i": 1, "sel_j": 1,
             "wireframe": True, "net": True}

    projection = tr.perspective(45, width / height, 0.1, 10.0)
    view = tr.lookAt(np.array([0, 2.2, 4.2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    arcball = Arcball(np.linalg.inv(view), np.array((width, height), dtype=float), 2.0,
                      np.array([0.0, 0.0, 0.0]))

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(20, 22, 28), background_width=470)
        .add("titulo", size=14)
        .add("seleccion")
        .add("propiedad")
        .footer("flechas seleccion   +/- altura   W wire   N grilla   arrastrar rota   R reset")
    )

    def rebuild():
        """Re-evalua la superficie y reescribe los buffers tras mover un punto."""
        puntos = state["puntos"]
        posiciones, normales = _evaluar_superficie(puntos, RESOLUCION)
        superficie.position[:] = posiciones.reshape(-1).astype(np.float32)
        superficie.normal[:] = normales.reshape(-1).astype(np.float32)
        net.position[:] = puntos.reshape(-1).astype(np.float32)
        puntos_gpu.position[:] = puntos.reshape(-1).astype(np.float32)

    def apply_state():
        i, j = state["sel_i"], state["sel_j"]
        altura = state["puntos"][i, j, 1]
        panel["titulo"] = "patch de Bezier 4x4   S(u,v) = sum P_ij B_i(u) B_j(v)"
        panel["seleccion"] = f"punto de control seleccionado: P[{i}][{j}]   altura y = {altura:+.2f}"
        esquina = (i in (0, 3)) and (j in (0, 3))
        panel["propiedad"] = (
            "esquina: la superficie pasa exactamente por este punto"
            if esquina else
            "interior: atrae la superficie pero esta no lo toca (envoltura convexa)"
        )
        print(f"[superficie_bezier] P[{i}][{j}] altura={altura:+.2f}")

    @window.event
    def on_draw():
        GL.glClearColor(0.12, 0.12, 0.15, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        view_actual = np.linalg.inv(arcball.pose)
        pipeline.use()
        pipeline["projection"] = projection.reshape(16, 1, order="F")
        pipeline["view"] = view_actual.reshape(16, 1, order="F")
        pipeline["transform"] = tr.identity().reshape(16, 1, order="F")

        # relleno de la superficie, empujado al fondo si hay wireframe encima.
        if state["wireframe"]:
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1.0, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        pipeline["color_base"] = COLOR_SUPERFICIE
        superficie.draw(GL.GL_TRIANGLES)

        if state["wireframe"]:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            pipeline["color_base"] = COLOR_WIRE
            superficie.draw(GL.GL_TRIANGLES)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # grilla de control y puntos, dibujados sin depth test para verlos siempre.
        if state["net"]:
            GL.glDisable(GL.GL_DEPTH_TEST)
            pipeline["color_base"] = COLOR_NET
            net.draw(GL.GL_LINES)
            GL.glPointSize(9.0)
            pipeline["color_base"] = COLOR_PUNTOS
            puntos_gpu.draw(GL.GL_POINTS)
            # el punto seleccionado, mas grande y en otro color.
            seleccion = pipeline.vertex_list(1, GL.GL_POINTS)
            seleccion.position[:] = state["puntos"][state["sel_i"], state["sel_j"]].astype(np.float32)
            seleccion.normal[:] = [0.0, 1.0, 0.0]
            GL.glPointSize(15.0)
            pipeline["color_base"] = COLOR_SELECCION
            seleccion.draw(GL.GL_POINTS)
            seleccion.delete()
            GL.glEnable(GL.GL_DEPTH_TEST)

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
        if symbol == key.UP:
            state["sel_i"] = min(3, state["sel_i"] + 1)
        elif symbol == key.DOWN:
            state["sel_i"] = max(0, state["sel_i"] - 1)
        elif symbol == key.RIGHT:
            state["sel_j"] = min(3, state["sel_j"] + 1)
        elif symbol == key.LEFT:
            state["sel_j"] = max(0, state["sel_j"] - 1)
        elif symbol in (key.EQUAL, key.PLUS, key.NUM_ADD):
            state["puntos"][state["sel_i"], state["sel_j"], 1] += 0.15
            rebuild()
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            state["puntos"][state["sel_i"], state["sel_j"], 1] -= 0.15
            rebuild()
        elif symbol == key.W:
            state["wireframe"] = not state["wireframe"]
        elif symbol == key.N:
            state["net"] = not state["net"]
        elif symbol == key.R:
            state["puntos"] = _puntos_iniciales()
            state["sel_i"], state["sel_j"] = 1, 1
            arcball.reset()
            rebuild()
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    rebuild()
    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    superficie_bezier()
