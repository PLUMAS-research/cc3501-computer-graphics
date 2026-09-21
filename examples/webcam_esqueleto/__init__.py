"""Un esqueleto de grafo de escena movido por la pose de una persona.

El estimador entrega una posicion global por articulacion y ninguna jerarquia.
El ejemplo arma con eso un grafo de huesos, donde cada nodo guarda una
transformacion local respecto de su padre. La conversion es

    M_n = inv(G_padre) @ G_n

y esta en `esqueleto.transformaciones_locales`.

La tecla J alterna entre usar los largos de hueso calibrados una vez (el
esqueleto queda rigido y la pose es una aproximacion) y usar los largos que
salen de los datos en cada cuadro (las articulaciones caen justo sobre las
mediciones y los huesos se estiran).

Sin instalar nada corre con `--fuente demo`. Con la camara necesita el extra:

    uv sync --extra webcam
"""

import os
from collections import deque
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

from . import fuentes
from .esqueleto import (
    HUESOS,
    grupo,
    largos_medidos,
    nombre_de_hueso,
    padre_de_cada_hueso,
    transformaciones_globales,
    transformaciones_locales,
)

GROSOR_HUESO = 0.07
CUADROS_DE_CALIBRACION = 45
SUAVIZADO = 0.35            # peso del cuadro nuevo en el promedio exponencial

COLOR_DE_GRUPO = {
    "centro": np.array([0.82, 0.84, 0.90], dtype=np.float32),
    "izq": np.array([0.81, 0.22, 0.54], dtype=np.float32),
    "der": np.array([0.25, 0.72, 0.78], dtype=np.float32),
}

POSICION_CAMARA = np.array([0.0, 0.10, 3.1], dtype=np.float32)
PUNTO_MIRADO = np.array([0.0, 0.05, 0.0], dtype=np.float32)

ANCHO_PREVISUALIZACION = 320
ALTO_PREVISUALIZACION = 260
VERDE = (150, 240, 150, 255)
AMARILLO = (245, 225, 130, 255)
ROJO = (255, 140, 140, 255)


def geometria_del_hueso(largo, grosor):
    """Lleva la primitiva a un hueso de ese largo, apoyado en su origen.

    Las mallas se cargan sin `rezero` ni `normalize`, asi que van de -1 a 1 en
    cada eje y `scale` recibe medias dimensiones. El hueso tiene que terminar
    exactamente donde empieza el siguiente, asi que aqui las cuentas tienen que
    dar justo.
    """
    return tr.translate(0, largo / 2, 0) @ tr.scale(grosor / 2, largo / 2, grosor / 2)


def armar_grafo(graph):
    """Un nodo de transformacion por hueso, colgando del hueso anterior."""
    padres = padre_de_cada_hueso()

    for hueso in HUESOS:
        nombre = nombre_de_hueso(*hueso)
        padre = padres[hueso]
        graph.add_transform(nombre, tr.identity())
        graph.add_edge("raiz" if padre is None else nombre_de_hueso(*padre), nombre)

        malla = "esfera" if hueso[1] == "cabeza" else "cubo"
        graph.add_object(
            f"{nombre}_geom", malla, "solido", parent=nombre,
            transform=geometria_del_hueso(0.1, GROSOR_HUESO),
            color_instancia=COLOR_DE_GRUPO[grupo(hueso)],
        )


def suavizar(anteriores, nuevas, peso):
    if anteriores is None:
        return {nombre: np.array(punto, dtype=np.float64) for nombre, punto in nuevas.items()}
    return {
        nombre: peso * nuevas[nombre] + (1.0 - peso) * anteriores[nombre]
        for nombre in nuevas
    }


@click.command("webcam_esqueleto",
               short_help="Esqueleto en un grafo de escena movido por la pose de una persona")
@click.option("--width", type=int, default=1180)
@click.option("--height", type=int, default=760)
@click.option("--fuente", type=click.Choice(["camara", "demo"]), default="camara",
              help="camara usa mediapipe; demo sintetiza una caminata sin instalar nada")
@click.option("--video", type=click.Path(exists=True), default=None,
              help="archivo de video en vez de la webcam")
@click.option("--dispositivo", type=int, default=0, help="indice de la camara")
def webcam_esqueleto(width, height, fuente, video, dispositivo):
    if video:
        fuente = "camara"

    origen = (fuentes.nueva_fuente_demo() if fuente == "demo"
              else fuentes.nueva_fuente_camara(dispositivo, video))

    window = pyglet.window.Window(width, height)
    carpeta = Path(os.path.dirname(__file__))

    graph = Scenegraph("raiz")
    # sin rezero ni normalize las primitivas quedan tal cual vienen, de -1 a 1
    graph.load_and_register_mesh("cubo", "assets/cube.off", rezero=False, normalize=False)
    graph.load_and_register_mesh("esfera", "assets/sphere.off", rezero=False, normalize=False)
    graph.register_pipeline(
        "solido",
        load_pipeline(carpeta / "vertex_program.glsl", carpeta / "fragment_program.glsl"),
    )
    armar_grafo(graph)

    graph.register_view_transform(
        tr.lookAt(POSICION_CAMARA, PUNTO_MIRADO, np.array([0.0, 1.0, 0.0]))
    )
    graph.set_global_attributes(projection=tr.perspective(45, width / height, 0.05, 30.0))

    state = {
        "tiempo": 0.0,
        "posiciones": None,
        "largos": None,
        "calibrando": deque(maxlen=CUADROS_DE_CALIBRACION),
        "jerarquia": True,
        "suavizado": True,
        "girar": False,
        "angulo_camara": 0.0,
        "detecta": fuente == "demo",
        "desvio": 0.0,
        "imagen": None,
        "calco": None,
    }

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(18, 18, 24))
        .add("fuente")
        .add("modo")
        .add("largos")
        .add("suavizado")
        .add("estado")
        .footer("J largos fijos o crudos   S suavizado   C recalibrar   G girar camara   R reset")
    )

    def apply_state():
        cual = "demo sintetica" if fuente == "demo" else ("video" if video else "camara")
        panel["fuente"] = f"fuente: {cual}"
        panel["modo"] = (
            "modo: jerarquia con largos de hueso fijos"
            if state["jerarquia"]
            else "modo: articulaciones crudas, los huesos se estiran"
        )
        panel.color("modo", VERDE if state["jerarquia"] else ROJO)
        desvio = state["desvio"]
        if state["jerarquia"]:
            panel["largos"] = "largos de hueso: fijos, 0,0 % de variacion"
            panel.color("largos", VERDE)
        else:
            panel["largos"] = ("largos de hueso: del dato, hasta "
                              f"{desvio:.1f} % de variacion".replace(".", ","))
            panel.color("largos", ROJO)
        panel["suavizado"] = f"suavizado temporal: {'si' if state['suavizado'] else 'no'}"
        if state["largos"] is None:
            panel["estado"] = (
                f"calibrando largos de hueso ({len(state['calibrando'])}"
                f"/{CUADROS_DE_CALIBRACION})"
            )
            panel.color("estado", AMARILLO)
        elif state["detecta"]:
            panel["estado"] = "persona detectada"
            panel.color("estado", VERDE)
        else:
            panel["estado"] = "sin deteccion: se mantiene la ultima pose"
            panel.color("estado", ROJO)

    apply_state()

    def dibujar_previsualizacion():
        """La imagen de la camara con las articulaciones calcadas encima."""
        imagen = state["imagen"]
        if imagen is None:
            return

        alto_imagen, ancho_imagen = imagen.shape[:2]
        escala = min(ANCHO_PREVISUALIZACION / ancho_imagen,
                     ALTO_PREVISUALIZACION / alto_imagen)
        ancho = int(ancho_imagen * escala)
        alto = int(alto_imagen * escala)
        x0, y0 = 14, 46   # sobre la linea de teclas del pie

        pyglet.image.ImageData(
            ancho_imagen, alto_imagen, "RGB", imagen.tobytes(), pitch=-ancho_imagen * 3
        ).blit(x0, y0, width=ancho, height=alto)

        if not state["calco"]:
            return

        batch = pyglet.graphics.Batch()
        puntos = {
            nombre: (x0 + u * ancho, y0 + (1.0 - v) * alto)
            for nombre, (u, v) in state["calco"].items()
        }
        figuras = []
        for padre, hijo in HUESOS:
            if padre in puntos and hijo in puntos:
                figuras.append(pyglet.shapes.Line(
                    *puntos[padre], *puntos[hijo], thickness=2,
                    color=(207, 56, 137), batch=batch))
        for posicion in puntos.values():
            figuras.append(pyglet.shapes.Circle(
                *posicion, 3.5, color=(245, 245, 250), batch=batch))
        batch.draw()

    @window.event
    def on_draw():
        GL.glClearColor(0.07, 0.08, 0.12, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()
        graph.render()

        with ui_overlay():
            dibujar_previsualizacion()
            panel.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.J:
            state["jerarquia"] = not state["jerarquia"]
        elif symbol == pyglet.window.key.S:
            state["suavizado"] = not state["suavizado"]
        elif symbol == pyglet.window.key.C:
            state["largos"] = None
            state["calibrando"].clear()
        elif symbol == pyglet.window.key.G:
            state["girar"] = not state["girar"]
        elif symbol == pyglet.window.key.R:
            state.update(jerarquia=True, suavizado=True, girar=False, angulo_camara=0.0,
                         largos=None)
            state["calibrando"].clear()
        apply_state()

    @window.event
    def on_close():
        fuentes.cerrar(origen)

    def update_world(dt, _):
        state["tiempo"] += dt

        medidas, imagen, calco = fuentes.leer(origen, state["tiempo"])
        if imagen is not None:
            state["imagen"] = imagen
        state["detecta"] = medidas is not None
        if calco is not None:
            state["calco"] = calco

        if medidas is not None:
            state["posiciones"] = suavizar(
                state["posiciones"], medidas, SUAVIZADO if state["suavizado"] else 1.0
            )

        posiciones = state["posiciones"]
        if posiciones is None:
            apply_state()
            return

        medidos = largos_medidos(posiciones)

        # la calibracion promedia unos cuadros para no quedarse con una medicion
        # mala, y despues los largos no cambian mas
        if state["largos"] is None:
            state["calibrando"].append(medidos)
            if len(state["calibrando"]) == CUADROS_DE_CALIBRACION:
                state["largos"] = {
                    hueso: float(np.median([m[hueso] for m in state["calibrando"]]))
                    for hueso in HUESOS
                }
            apply_state()

        # cuanto se apartan los largos de este cuadro de los calibrados: es la
        # cifra que separa una bolsa de puntos de un esqueleto
        if state["largos"] is not None:
            state["desvio"] = 100.0 * max(
                abs(medidos[hueso] - state["largos"][hueso]) / state["largos"][hueso]
                for hueso in HUESOS
            )

        largos = medidos if (not state["jerarquia"] or state["largos"] is None) else state["largos"]

        globales, _ = transformaciones_globales(posiciones, largos)
        locales = transformaciones_locales(globales)

        for hueso in HUESOS:
            nombre = nombre_de_hueso(*hueso)
            largo = largos[hueso]
            es_cabeza = hueso[1] == "cabeza"
            graph.nodes[nombre]["transform"] = locales[hueso]
            graph.nodes[f"{nombre}_geom"]["transform"] = geometria_del_hueso(
                largo, largo * 0.85 if es_cabeza else GROSOR_HUESO
            )

        apply_state()

        if state["girar"]:
            state["angulo_camara"] += dt * 0.5
        giro = tr.rotationY(state["angulo_camara"])
        ojo = (giro @ np.append(POSICION_CAMARA, 1.0))[:3]
        graph.views[graph.current_view] = tr.lookAt(
            ojo.astype(np.float32), PUNTO_MIRADO, np.array([0.0, 1.0, 0.0])
        )

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
