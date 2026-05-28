"""Ejemplo rico de mallas: simplificacion por error cuadratico (QEM).

Construye sobre la operacion collapse de la estructura half-edge. El alumno baja
el presupuesto de triangulos de un modelo 3D y ve la malla simplificarse: el
algoritmo colapsa primero las aristas mas baratas, las que menos alteran la
forma. Es el problema de los niveles de detalle (LOD) en videojuegos: el mismo
modelo en varias resoluciones segun la distancia a la camara.

Los niveles se precalculan al iniciar (una sola pasada de colapsos que captura
la malla al cruzar cada presupuesto) y luego se recorren al instante con el
teclado. El algoritmo QEM vive en qem.py; la estructura y el collapse en
grafica/half_edge.py.

Controles:
- , / .:    nivel mas fino / mas grueso de simplificacion.
- W:        muestra u oculta el wireframe (las aristas de los triangulos).
- arrastrar: rota el modelo (arcball). scroll: zoom.
- R:        reinicia la camara.
- ESC:      cierra la ventana.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.half_edge import HalfEdgeMesh
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

from .qem import simplificar_a_niveles

COLOR_MALLA = (0.85, 0.78, 0.95)  # lila pastel.
COLOR_WIRE = (0.08, 0.08, 0.10)
FRACCIONES = [1.0, 0.6, 0.4, 0.25, 0.15, 0.08]  # presupuestos como fraccion del original.


def _cargar_normalizada(filename):
    """Carga el modelo y lo centra y escala para caber en el cubo [-1, 1]."""
    tri = tm.load(filename, force="mesh")
    vertices = np.asarray(tri.vertices, dtype=np.float64)
    centro = (vertices.min(0) + vertices.max(0)) / 2
    escala = 2.0 / (vertices.max(0) - vertices.min(0)).max()
    vertices = (vertices - centro) * escala
    return vertices, np.asarray(tri.faces, dtype=np.int64)


def _construir_vertex_list(pipeline, positions, faces):
    """Arma un vertex_list plano con una normal por cara (flat shading).

    Repetimos la normal de cada cara en sus tres vertices, asi el sombreado es
    constante por triangulo y se ve la malla poligonal sin suavizado. Esto deja
    a la vista la reduccion de triangulos.
    """
    triangulos = positions[faces]                       # (F, 3, 3)
    borde1 = triangulos[:, 1] - triangulos[:, 0]
    borde2 = triangulos[:, 2] - triangulos[:, 0]
    normales = np.cross(borde1, borde2)
    largo = np.linalg.norm(normales, axis=1, keepdims=True)
    largo[largo == 0] = 1.0
    normales /= largo

    vertices_planos = triangulos.reshape(-1, 3).astype(np.float32)
    normales_planas = np.repeat(normales, 3, axis=0).astype(np.float32)
    vertex_list = pipeline.vertex_list(len(vertices_planos), GL.GL_TRIANGLES)
    vertex_list.position[:] = vertices_planos.ravel()
    vertex_list.normal[:] = normales_planas.ravel()
    return vertex_list


@click.command("simplificacion_qem", short_help="Simplificacion de mallas por error cuadratico (QEM)")
@click.argument("filename", type=str, default="assets/bunny.obj")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
def simplificacion_qem(filename, width, height):
    window = pyglet.window.Window(width, height, caption="simplificacion QEM")

    positions, faces = _cargar_normalizada(filename)
    malla = HalfEdgeMesh.from_faces(positions, faces)
    caras_originales = malla.n_faces()
    objetivos = [int(caras_originales * fraccion) for fraccion in FRACCIONES]
    print(f"[qem] {filename}: {caras_originales} caras. preparando niveles {objetivos} ...")
    snapshots = simplificar_a_niveles(
        malla, objetivos,
        reporte=lambda n: print(f"[qem]   colapsando... {n} caras", flush=True),
    )
    print(f"[qem] listo: {len(snapshots)} niveles")

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )
    vertex_lists = [_construir_vertex_list(pipeline, pos, f) for pos, f in snapshots]

    projection = tr.perspective(45, width / height, 0.1, 10.0)
    view = tr.lookAt(np.array([0, 0, 3.2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    arcball = Arcball(np.linalg.inv(view), np.array((width, height), dtype=float), 2.0,
                      np.array([0.0, 0.0, 0.0]))

    state = {"nivel": 0, "wireframe": True}

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(20, 20, 25), background_width=440)
        .add("nivel", size=14)
        .add("caras")
        .add("reduccion")
        .footer(", . nivel   W wireframe   arrastrar rota   scroll zoom   R reset")
    )

    def apply_state():
        pos, caras = snapshots[state["nivel"]]
        n_caras = len(caras)
        panel["nivel"] = f"nivel {state['nivel'] + 1}/{len(snapshots)}"
        panel["caras"] = f"caras: {n_caras}   vertices: {len(pos)}"
        panel["reduccion"] = (
            f"reduccion: {100 * (1 - n_caras / caras_originales):.0f}% "
            f"(original: {caras_originales} caras)"
        )
        print(f"[qem] nivel {state['nivel'] + 1}/{len(snapshots)}  caras={n_caras}")

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

        vertex_list = vertex_lists[state["nivel"]]

        # paso de relleno. Con wireframe activo, empujamos el relleno hacia atras
        # con polygon offset para que las lineas no peleen con el en el z-buffer.
        if state["wireframe"]:
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1.0, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        pipeline["color_base"] = COLOR_MALLA
        vertex_list.draw(GL.GL_TRIANGLES)

        if state["wireframe"]:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            pipeline["color_base"] = COLOR_WIRE
            vertex_list.draw(GL.GL_TRIANGLES)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

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
        if symbol == key.PERIOD:
            state["nivel"] = min(len(snapshots) - 1, state["nivel"] + 1)
            apply_state()
        elif symbol == key.COMMA:
            state["nivel"] = max(0, state["nivel"] - 1)
            apply_state()
        elif symbol == key.W:
            state["wireframe"] = not state["wireframe"]
        elif symbol == key.R:
            arcball.reset()
        elif symbol == key.ESCAPE:
            window.close()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    simplificacion_qem()
