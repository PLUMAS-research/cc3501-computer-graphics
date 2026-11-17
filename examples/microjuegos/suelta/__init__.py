"""Microjuego de ejemplo: soltar el bloque sobre la torre.

El bloque oscila sobre la torre y cae al presionar espacio. La escena es 3D y
se dibuja con el grafo de escena del curso (unidades 08, 09 y 12). La caída y el
choque los simula pymunk en el plano xy (unidad 15): un bloque que aterriza con
su centro de masa fuera de la torre se vuelca y cae al suelo. El anfitrión crea
el espacio de pymunk y lo avanza; como la escena es 3D, el microjuego define su
propio dibujar() en vez de uniformes().
"""

import math
import random
from pathlib import Path

import numpy as np
import pyglet.gl as GL
import pymunk
import trimesh as tm

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_nodes import _node_from_mesh

ROL = "microjuego"
INSTRUCCION = "¡Suelta!"
GANA_SI_SOBREVIVE = False

CARPETA = Path(__file__).parent
GRAVEDAD = -12.0
ALTURA_INICIAL = 2.5
ANCHO_BLOQUE, ALTO_BLOQUE = 1.0, 0.5
ALTO_TORRE = 3.0
AMPLITUD = 2.2
# segundos que el bloque tiene que quedarse quieto sobre la torre para ganar
QUIETO_PARA_GANAR = 0.3

# frecuencia de la oscilación (rad/s), ancho de la torre y si la torre se mueve.
# El bloque se mantiene si su centro de masa cae sobre la torre, así que la
# mitad del ancho es la tolerancia al soltarlo.
NIVELES = {
    1: {"frecuencia": 2.0, "ancho_torre": 1.0, "torre_movil": False},
    2: {"frecuencia": 3.0, "ancho_torre": 0.7, "torre_movil": False},
    3: {"frecuencia": 3.5, "ancho_torre": 0.6, "torre_movil": True},
}


def caja(color):
    """Caja unitaria con vértices separados por cara, para que el sombreado sea plano."""
    malla = tm.creation.box(extents=(1.0, 1.0, 1.0))
    malla.unmerge_vertices()
    malla.visual.vertex_colors = color
    return _node_from_mesh(malla, fix_normals=False)


def preparar(ventana):
    grafo = Scenegraph("mundo")
    grafo.load_and_register_pipeline("difuso", CARPETA / "vertex_program.glsl", CARPETA / "fragment_program.glsl")
    grafo.register_mesh("caja", caja([255, 255, 255, 255]))

    grafo.add_object("suelo", "caja", "difuso", parent="mundo",
                     transform=tr.matmul([tr.translate(0, -ALTO_TORRE - 0.05, 0), tr.scale(20, 0.1, 20)]),
                     color=np.array([0.45, 0.6, 0.4], dtype=np.float32))
    grafo.add_object("torre", "caja", "difuso", parent="mundo", color=np.array([0.04, 0.05, 0.31], dtype=np.float32))
    grafo.add_object("cable", "caja", "difuso", parent="mundo", color=np.array([0.2, 0.2, 0.2], dtype=np.float32))
    grafo.add_object("bloque", "caja", "difuso", parent="mundo", color=np.array([0.81, 0.22, 0.54], dtype=np.float32))

    grafo.register_view_transform(tr.lookAt(np.array([0.0, 1.5, 8.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])))
    grafo.set_global_attributes(projection=tr.perspective(45, ventana.width / ventana.height, 0.1, 50.0))
    return {"grafo": grafo}


def iniciar(dificultad, espacio):
    nivel = NIVELES[dificultad]
    # la escena está en metros, así que la gravedad del anfitrión (en píxeles) no sirve
    espacio.gravity = (0.0, GRAVEDAD)

    suelo = pymunk.Segment(espacio.static_body, (-20, -ALTO_TORRE), (20, -ALTO_TORRE), 0.0)
    suelo.friction = 0.8
    # la torre es cinemática: los choques no la mueven, pero arrastra al bloque
    # con su velocidad cuando se desplaza
    torre = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    torre.position = (0.0, -ALTO_TORRE / 2)
    forma_torre = pymunk.Poly.create_box(torre, (nivel["ancho_torre"], ALTO_TORRE))
    forma_torre.friction = 0.8
    espacio.add(suelo, torre, forma_torre)

    return {
        "nivel": nivel,
        "tiempo": 0.0,
        "fase": random.uniform(0, 2 * math.pi),
        "espacio": espacio,
        "torre": torre,
        "bloque": None,  # el cuerpo del bloque entra al espacio recién al soltarlo
        "x": 0.0,
        "quieto": 0.0,
        "resultado": None,
    }


def soltar(estado):
    masa = 1.0
    bloque = pymunk.Body(masa, pymunk.moment_for_box(masa, (ANCHO_BLOQUE, ALTO_BLOQUE)))
    bloque.position = (estado["x"], ALTURA_INICIAL)
    forma = pymunk.Poly.create_box(bloque, (ANCHO_BLOQUE, ALTO_BLOQUE))
    forma.friction = 0.8
    estado["espacio"].add(bloque, forma)
    estado["bloque"] = bloque


def actualizar(estado, dt, entrada):
    nivel = estado["nivel"]
    if estado["bloque"] is None:
        estado["x"] = AMPLITUD * math.sin(nivel["frecuencia"] * estado["tiempo"] + estado["fase"])
        if entrada["espacio_presionado"]:
            soltar(estado)

    estado["tiempo"] += dt
    if nivel["torre_movil"]:
        # x = 0.8 sin(1.3 t); pymunk integra la posición a partir de la velocidad
        estado["torre"].velocity = (0.8 * 1.3 * math.cos(1.3 * estado["tiempo"]), 0.0)

    bloque = estado["bloque"]
    if bloque is None or estado["resultado"] is not None:
        return
    if bloque.position.y < -ALTO_BLOQUE:
        estado["resultado"] = False  # se cayó por el costado de la torre
        return
    # quieto respecto de la torre, que en el nivel 3 se mueve
    relativa = bloque.velocity - estado["torre"].velocity
    if bloque.position.y > 0 and relativa.length < 0.1 and abs(bloque.angular_velocity) < 0.1:
        estado["quieto"] += dt
        if estado["quieto"] >= QUIETO_PARA_GANAR:
            estado["resultado"] = True
    else:
        estado["quieto"] = 0.0


def dibujar(recursos, estado):
    grafo = recursos["grafo"]
    ancho_torre = estado["nivel"]["ancho_torre"]
    grafo.nodes["torre"]["transform"] = tr.matmul([
        tr.translate(estado["torre"].position.x, -ALTO_TORRE / 2, 0), tr.scale(ancho_torre, ALTO_TORRE, ancho_torre)])

    bloque = estado["bloque"]
    if bloque is None:
        x, y, angulo = estado["x"], ALTURA_INICIAL, 0.0
    else:
        x, y, angulo = bloque.position.x, bloque.position.y, bloque.angle
    grafo.nodes["bloque"]["transform"] = tr.matmul([
        tr.translate(x, y, 0), tr.rotationZ(angulo), tr.scale(ANCHO_BLOQUE, ALTO_BLOQUE, ANCHO_BLOQUE)])
    # el cable cuelga mientras el bloque no se suelte
    largo_cable = 4.0 if bloque is None else 0.0
    grafo.nodes["cable"]["transform"] = tr.matmul([
        tr.translate(x, y + ALTO_BLOQUE / 2 + largo_cable / 2, 0), tr.scale(0.04, largo_cable, 0.04)])

    GL.glClearColor(0.85, 0.9, 0.95, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    grafo.render()
