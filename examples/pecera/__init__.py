from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline

from .world import Pecera


COLOR_VIDRIO = np.array([0.55, 0.85, 0.95], dtype=np.float32)
ALPHA_VIDRIO = 0.18

COLOR_AGUA = np.array([0.30, 0.65, 0.85], dtype=np.float32)
ALPHA_AGUA = 0.30

COLOR_ARENA = np.array([0.85, 0.74, 0.55], dtype=np.float32)


def subir_quad(pipeline, esquinas, color):
    p0, p1, p2, p3 = esquinas
    posiciones = np.array(
        [p0, p1, p2, p0, p2, p3], dtype=np.float32
    ).flatten()
    colores = np.tile(color, 6).astype(np.float32)
    vl = pipeline.vertex_list(6, GL.GL_TRIANGLES, position="f", color="f")
    vl.position[:] = posiciones
    vl.color[:] = colores
    return vl


def construir_paredes(pipeline, tank_min, tank_max):
    xmin, ymin, zmin = tank_min
    xmax, ymax, zmax = tank_max

    frente = [
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]
    atras = [
        (xmax, ymin, zmin),
        (xmin, ymin, zmin),
        (xmin, ymax, zmin),
        (xmax, ymax, zmin),
    ]
    izquierda = [
        (xmin, ymin, zmin),
        (xmin, ymin, zmax),
        (xmin, ymax, zmax),
        (xmin, ymax, zmin),
    ]
    derecha = [
        (xmax, ymin, zmax),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmax, ymax, zmax),
    ]
    superficie = [
        (xmin, ymax, zmin),
        (xmax, ymax, zmin),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]

    caras = [
        ("frente", frente, COLOR_VIDRIO, ALPHA_VIDRIO),
        ("atras", atras, COLOR_VIDRIO, ALPHA_VIDRIO),
        ("izquierda", izquierda, COLOR_VIDRIO, ALPHA_VIDRIO),
        ("derecha", derecha, COLOR_VIDRIO, ALPHA_VIDRIO),
        ("agua", superficie, COLOR_AGUA, ALPHA_AGUA),
    ]

    paredes = []
    for nombre, esquinas, color, alpha in caras:
        vl = subir_quad(pipeline, esquinas, color)
        centro = np.mean(np.asarray(esquinas, dtype=np.float32), axis=0)
        paredes.append({"nombre": nombre, "vl": vl, "centro": centro, "alpha": alpha})

    return paredes


def construir_arena(pipeline, tank_min, tank_max):
    xmin, ymin, zmin = tank_min
    xmax, _, zmax = tank_max
    esquinas = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymin, zmax),
        (xmin, ymin, zmax),
    ]
    return subir_quad(pipeline, esquinas, COLOR_ARENA)


def ordenar_paredes(paredes, view_matrix):
    """Ordena las paredes por profundidad en espacio de cámara, de más lejano a
    más cercano."""
    profundidades = []
    for pared in paredes:
        pos_mundo = np.array([*pared["centro"], 1.0], dtype=np.float32)
        pos_camara = view_matrix @ pos_mundo
        profundidades.append(-pos_camara[2])
    orden = np.argsort(profundidades)[::-1]
    return [paredes[i] for i in orden]


@click.command(
    "pecera",
    short_help="Pecera con boids 3D para mostrar profundidad y transparencia",
)
@click.option("--n_peces", type=int, default=40)
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=768)
def pecera(n_peces, width, height):
    window = pyglet.window.Window(width, height)
    window.set_caption("Pecera: profundidad, blending y orden de transparencia")

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    tank_min = np.array([-1.5, -0.9, -1.0], dtype=np.float32)
    tank_max = np.array([1.5, 0.9, 1.0], dtype=np.float32)

    sim = Pecera(n_peces, tank_min, tank_max)

    arena_vl = construir_arena(pipeline, tank_min, tank_max)
    paredes = construir_paredes(pipeline, tank_min, tank_max)

    near_plane = 0.1
    far_plane = 30.0
    projection = tr.perspective(
        45.0, float(width) / float(height), near_plane, far_plane
    )
    view_inicial = tr.lookAt(
        np.array([3.5, 1.8, 4.5]),
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
    )

    arcball = Arcball(
        np.linalg.inv(view_inicial),
        np.array((width, height), dtype=float),
        2.0,
        np.zeros(3),
    )

    estado = {
        "pausado": False,
        "depth_test": True,
        "blend": True,
        "orden_correcto": True,
    }

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
    def on_mouse_release(x, y, button, modifiers):
        arcball.set_state(Arcball.STATE_ROTATE)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.D:
            estado["depth_test"] = not estado["depth_test"]
        elif symbol == pyglet.window.key.B:
            estado["blend"] = not estado["blend"]
        elif symbol == pyglet.window.key.O:
            estado["orden_correcto"] = not estado["orden_correcto"]
        elif symbol == pyglet.window.key.P:
            estado["pausado"] = not estado["pausado"]
        elif symbol == pyglet.window.key.R:
            arcball.pose = np.linalg.inv(view_inicial)

    def actualizar_simulacion(dt):
        if not estado["pausado"]:
            sim.step()

    hud_label = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=11,
        x=10,
        y=height - 10,
        anchor_x="left",
        anchor_y="top",
        color=(230, 230, 230, 255),
        multiline=True,
        width=width - 20,
    )

    def actualizar_hud():
        if estado["orden_correcto"]:
            descripcion_orden = "ON  (opacos -> transparentes ordenados, sin escritura)"
        else:
            descripcion_orden = "OFF (vidrios primero con escritura de profundidad)"
        hud_label.text = (
            f"Peces: {sim.n}  Pausa: {'si' if estado['pausado'] else 'no'}\n"
            f"[D] Depth test:    {'ON' if estado['depth_test'] else 'OFF'}\n"
            f"[B] Blending:      {'ON' if estado['blend'] else 'OFF'}\n"
            f"[O] Orden correcto: {descripcion_orden}\n"
            f"[P] Pausa  [R] Reset camara  Mouse: izq rota, der mueve, rueda zoom"
        )

    def dibujar_arena():
        pipeline["alpha"] = 1.0
        arena_vl.draw(GL.GL_TRIANGLES)

    def dibujar_peces():
        posiciones, colores = sim.fish_triangles(size=0.07)
        n_vertices_peces = len(posiciones) // 3
        peces_vl = pipeline.vertex_list(
            n_vertices_peces, GL.GL_TRIANGLES, position="f", color="f"
        )
        peces_vl.position[:] = posiciones
        peces_vl.color[:] = colores
        peces_vl.draw(GL.GL_TRIANGLES)
        peces_vl.delete()

    def dibujar_paredes(paredes_a_dibujar):
        for pared in paredes_a_dibujar:
            pipeline["alpha"] = pared["alpha"]
            pared["vl"].draw(GL.GL_TRIANGLES)

    @window.event
    def on_draw():
        GL.glClearColor(0.06, 0.06, 0.10, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if estado["depth_test"]:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

        if estado["blend"]:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        else:
            GL.glDisable(GL.GL_BLEND)

        view_matrix = np.linalg.inv(arcball.pose)
        identidad = tr.identity()

        pipeline.use()
        pipeline["projection"] = projection.reshape(16, 1, order="F")
        pipeline["view"] = view_matrix.reshape(16, 1, order="F")
        pipeline["model"] = identidad.reshape(16, 1, order="F")

        if estado["orden_correcto"]:
            # Fase 1: opacos con profundidad activa.
            GL.glDepthMask(GL.GL_TRUE)
            dibujar_arena()
            dibujar_peces()

            # Fase 2: transparentes ordenados back-to-front, sin escribir profundidad.
            GL.glDepthMask(GL.GL_FALSE)
            dibujar_paredes(ordenar_paredes(paredes, view_matrix))
        else:
            # Sin separar fases: vidrios primero con escritura de profundidad activa.
            # Las paredes dejan su profundidad en el buffer y los peces que queden
            # detras pierden el depth test, asi que desaparecen.
            GL.glDepthMask(GL.GL_TRUE)
            dibujar_paredes(paredes)
            dibujar_arena()
            dibujar_peces()

        # Restaurar estado por si pyglet dibuja algo más.
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        actualizar_hud()
        with ui_overlay():
            hud_label.draw()

    pyglet.clock.schedule_interval(actualizar_simulacion, 1 / 60.0)
    pyglet.app.run()


if __name__ == "__main__":
    pecera()
