import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_nodes import _node_from_mesh
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

from examples.raytracing_basico.trazado import Camara
from examples.raytracing_basico import render_progresivo
from .cajas import CajasAABB
from .fondo import FondoCielo
from .raytracing import recolectar_geometria, trazar_grafo


# La misma camara alimenta la rasterizacion (lookAt + perspective) y el
# trazado de rayos (generacion de rayos primarios). Asi ambas vistas coinciden.
POSICION_CAMARA = np.array([0.0, 0.35, -1.0])
OBJETIVO_CAMARA = np.array([0.0, -0.1, 2.5])
ARRIBA_MUNDO = np.array([0.0, 1.0, 0.0])
FOV_GRADOS = 60.0

LUZ = {
    "posicion": np.array([5.0, 6.0, -4.0]),
    "color": np.array([1.0, 1.0, 1.0]),
}

# Coeficientes de iluminacion compartidos por la rasterizacion (shader Phong)
# y el trazador por CPU, para que ambas vistas sean comparables.
AMBIENTE = 0.12
DIFUSO = 0.9
ESPECULAR = 0.5
BRILLO = 40.0
ESCALA_TABLERO = 1.2
COLOR_PISO_A = np.array([0.95, 0.95, 0.95])
COLOR_PISO_B = np.array([0.20, 0.20, 0.20])

# Reflectividad por objeto (solo afecta al ray tracing; la rasterizacion no
# tiene reflejos). El piso espeja un poco a los Pokemon; cada Pokemon refleja
# su entorno de forma sutil.
REFLECTIVIDADES = {
    "charmander": 0.25,
    "squirtle": 0.25,
    "bulbasaur": 0.25,
    "piso": 0.35,
}

# Colores del cielo (gradiente segun la altura), compartidos por el fondo
# rasterizado y el fondo del trazador para que ambos modos coincidan.
COLOR_CIELO_ARRIBA = np.array([0.55, 0.70, 0.95])
COLOR_CIELO_ABAJO = np.array([0.85, 0.90, 0.98])

# Anchos disponibles para el render por CPU (las teclas , . recorren la lista).
# A mayor resolucion la comparacion con la rasterizacion es mas justa, pero el
# trazado por CPU es lento, por eso se calcula de forma progresiva
# (RenderProgresivo: el trazado se reparte en franjas a lo largo de los cuadros).
RESOLUCIONES = [120, 200, 300, 450, 600, 900]

POKEMON = [
    ("charmander", "assets/Charmander.STL", tr.translate(1.35, -0.5, 1.6), np.array([1.0, 0.6, 0.0])),
    ("squirtle", "assets/Squirtle.STL", tr.translate(0.25, -0.5, 2.25), np.array([0.2, 0.8, 1.0])),
    ("bulbasaur", "assets/Bulbasaur.STL", tr.translate(-1.5, -0.5, 3.5), np.array([0.3, 1.0, 0.2])),
]


def cargar_pokemon(ruta, escala_objetivo=1.5):
    """Carga una malla STL, la escala y la orienta para que quede de pie."""
    mesh = tm.load(ruta, force="mesh")
    mesh.apply_scale(escala_objetivo / mesh.scale)
    mesh.apply_transform(tr.rotationZ(np.pi) @ tr.rotationX(np.pi / 2.0))
    return mesh


def crear_piso(semilado=8.0):
    """Cuadrado horizontal grande en el plano XZ (dos triangulos)."""
    vertices = np.array(
        [[-semilado, 0.0, -semilado], [semilado, 0.0, -semilado],
         [semilado, 0.0, semilado], [-semilado, 0.0, semilado]],
        dtype=float,
    )
    # winding para que la normal de cara apunte hacia +y (arriba)
    caras = np.array([[0, 2, 1], [0, 3, 2]])
    return tm.Trimesh(vertices=vertices, faces=caras, process=False)


@click.command("raytracing_cpu", short_help="Rasterizacion vs ray tracing del mismo grafo")
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=700)
@click.option("--resolucion", type=int, default=160, help="Ancho en pixeles del render por CPU")
def raytracing_cpu(width, height, resolucion):
    window = pyglet.window.Window(width, height, caption="Rasterizacion vs ray tracing")

    aspecto = width / height

    phong_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "phong_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "phong_fragment_program.glsl",
    )

    graph = Scenegraph("root")
    graph.register_pipeline("phong", phong_pipeline)

    for nombre, ruta, transform, color in POKEMON:
        graph.register_mesh(nombre, _node_from_mesh(cargar_pokemon(ruta), fix_normals=True))
        graph.add_object(
            nombre, nombre, "phong", parent="root",
            transform=transform, color=color, es_piso=0.0,
        )

    # sin fix_normals: el winding de crear_piso ya orienta la normal hacia arriba
    graph.register_mesh("piso", _node_from_mesh(crear_piso(), fix_normals=False))
    graph.add_object(
        "piso", "piso", "phong", parent="root",
        transform=tr.translate(0.0, -0.5, 0.0), color=np.array([0.5, 0.5, 0.5]), es_piso=1.0,
    )

    view = tr.lookAt(POSICION_CAMARA, OBJETIVO_CAMARA, ARRIBA_MUNDO)
    projection = tr.perspective(FOV_GRADOS, aspecto, 0.1, 100.0)
    graph.register_view_transform(view)
    graph.set_global_attributes(
        projection=projection,
        light_position=LUZ["posicion"].astype(np.float32),
        light_color=LUZ["color"].astype(np.float32),
        view_position=POSICION_CAMARA.astype(np.float32),
        ambient_strength=AMBIENTE,
        diffuse_strength=DIFUSO,
        specular_strength=ESPECULAR,
        shininess=BRILLO,
        escala_tablero=ESCALA_TABLERO,
        color_piso_a=COLOR_PISO_A.astype(np.float32),
        color_piso_b=COLOR_PISO_B.astype(np.float32),
    )

    camara = Camara(POSICION_CAMARA, OBJETIVO_CAMARA, ARRIBA_MUNDO, FOV_GRADOS, aspecto)
    fondo = FondoCielo(camara, COLOR_CIELO_ARRIBA, COLOR_CIELO_ABAJO)

    # la escena es estatica, asi que recolectamos la geometria de mundo una sola
    # vez (triangulos + normales + AABB) y la reusamos en cada render y para las cajas
    geometrias = recolectar_geometria(graph, reflectividades=REFLECTIVIDADES, nodos_piso={"piso"})
    cajas = CajasAABB(geometrias)

    state = {"modo_rt": False, "sombras": True, "rebotes": 1, "cajas": False}

    def trazar_banda(ancho, alto, fila_inicio, fila_fin):
        return trazar_grafo(
            geometrias, camara, LUZ, ancho, alto,
            ambiente=AMBIENTE, difuso=DIFUSO, especular=ESPECULAR, brillo=BRILLO,
            escala_tablero=ESCALA_TABLERO,
            color_piso_a=COLOR_PISO_A, color_piso_b=COLOR_PISO_B,
            color_cielo_arriba=COLOR_CIELO_ARRIBA, color_cielo_abajo=COLOR_CIELO_ABAJO,
            usar_sombras=state["sombras"], rebotes=state["rebotes"],
            fila_inicio=fila_inicio, fila_fin=fila_fin,
        )

    estado_render = render_progresivo.nuevo_estado(
        trazar_banda, RESOLUCIONES, aspecto, resolucion_inicial=resolucion
    )

    def perillas():
        return (state["sombras"], state["rebotes"])

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(15, 15, 20))
        .add("modo")
        .add("resolucion")
        .add("sombras")
        .add("rebotes")
        .add("tiempo")
        .footer("T raster/RT   S sombras   R rebotes   B cajas   , . resolucion   espacio recalcular")
    )

    def actualizar_panel():
        ancho_render, alto_render = render_progresivo.resolucion_actual(estado_render)
        panel["modo"] = f"modo: {'ray tracing (CPU)' if state['modo_rt'] else 'rasterizacion (OpenGL)'}"
        panel["resolucion"] = f"resolucion RT: {ancho_render} x {alto_render}"
        panel["sombras"] = f"sombras: {'ON' if state['sombras'] else 'off'}"
        panel["rebotes"] = f"rebotes de reflexion: {state['rebotes']}"
        if estado_render["activo"]:
            panel["tiempo"] = f"calculando... {render_progresivo.porcentaje(estado_render)}%"
        else:
            panel["tiempo"] = f"tiempo RT: {estado_render['segundos']:.2f} s"

    def avanzar_render(dt):
        if not (state["modo_rt"] and estado_render["activo"]):
            return
        termino = render_progresivo.avanzar(estado_render, dt)
        actualizar_panel()
        if termino:
            print(
                f"[raytracing_cpu] {estado_render['ancho']}x{estado_render['alto']} "
                f"sombras={state['sombras']} rebotes={state['rebotes']} "
                f"tiempo={estado_render['segundos']:.2f}s"
            )

    actualizar_panel()

    @window.event
    def on_draw():
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        window.clear()

        if state["modo_rt"] and estado_render["textura"] is not None:
            estado_render["textura"].blit(0, 0, width=width, height=height)
        else:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            fondo.draw()
            GL.glEnable(GL.GL_DEPTH_TEST)
            graph.render(recalculate_transforms=False)

        # las cajas AABB que usa el trazador como fase ancha, sobre cualquiera de las vistas
        if state["cajas"]:
            cajas.draw(view, projection)

        with ui_overlay():
            panel.draw()

    def cambiar_perilla():
        # al tocar una perilla en modo RT, mostramos la imagen cacheada o la calculamos
        if state["modo_rt"]:
            render_progresivo.mostrar(estado_render, perillas())
        actualizar_panel()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.T:
            state["modo_rt"] = not state["modo_rt"]
            if state["modo_rt"]:
                render_progresivo.mostrar(estado_render, perillas())
            actualizar_panel()
        elif symbol == pyglet.window.key.S:
            state["sombras"] = not state["sombras"]
            cambiar_perilla()
        elif symbol == pyglet.window.key.R:
            state["rebotes"] = (state["rebotes"] + 1) % 4
            cambiar_perilla()
        elif symbol == pyglet.window.key.B:
            state["cajas"] = not state["cajas"]
            actualizar_panel()
        elif symbol == pyglet.window.key.COMMA:
            render_progresivo.menos_resolucion(estado_render)
            cambiar_perilla()
        elif symbol == pyglet.window.key.PERIOD:
            render_progresivo.mas_resolucion(estado_render)
            cambiar_perilla()
        elif symbol == pyglet.window.key.SPACE:
            # espacio fuerza un recalculo aunque la imagen este cacheada
            if state["modo_rt"]:
                render_progresivo.iniciar(estado_render, perillas())

    graph.calculate_global_transforms()
    pyglet.clock.schedule_interval(avanzar_render, 1 / 60.0)
    pyglet.app.run()
