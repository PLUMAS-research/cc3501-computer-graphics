"""Un carrusel armado con un grafo de escena.

La plataforma gira y cada caballito, colgado de ella, sube y baja mientras
mantiene su propia orientacion. La transformacion local de un caballito compone
tres factores y el orden entre ellos decide si el resultado es un carrusel o un
desastre: la tecla O permuta ese orden en vivo.

Los dos primeros caballitos llevan una cupula de vidrio translucida. Se dibujan en una
segunda pasada, despues de todo lo opaco, para poder ordenarlas por distancia a
la camara y apagar la escritura al depth buffer.
"""

import colorsys
import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

N_CABALLITOS_POR_OMISION = 8
RADIO_PLATAFORMA = 3.2
RADIO_CABALLITOS = 2.3
ALTURA_MONTURA = 0.95
ALTURA_TECHO = 2.45

VELOCIDAD_CARRUSEL = 0.55        # radianes por segundo
VELOCIDAD_OSCILACION = 2.0
AMPLITUD_OSCILACION = 0.26
AMPLITUD_GALOPE = 0.26           # radianes de inclinacion del caballito

POSICION_CAMARA = np.array([0.0, 3.05, 9.6], dtype=np.float32)
PUNTO_MIRADO = np.array([0.0, 1.35, 0.0], dtype=np.float32)

# los tres ordenes que puede tomar la transformacion local de un caballito.
# el primero es el correcto; los otros dos son los errores tipicos.
ORDENES = ("correcto", "pose primero", "oscilacion al final")

FORMULA_ORDEN = {
    "correcto": "colocacion @ oscilacion @ pose",
    "pose primero": "pose @ colocacion @ oscilacion",
    "oscilacion al final": "colocacion @ pose @ oscilacion",
}

COLOR_PLATAFORMA = np.array([0.82, 0.30, 0.28], dtype=np.float32)
COLOR_TECHO = np.array([0.94, 0.76, 0.26], dtype=np.float32)
COLOR_METAL = np.array([0.80, 0.80, 0.86], dtype=np.float32)
COLOR_AMPOLLETA = np.array([1.00, 0.96, 0.80], dtype=np.float32)
COLOR_PISO = np.array([0.16, 0.20, 0.18], dtype=np.float32)
COLOR_VIDRIO = np.array([0.55, 0.85, 1.00], dtype=np.float32)

VERDE = (150, 240, 150, 255)
ROJO = (255, 140, 140, 255)


PIEZAS = []


def agregar_solido(graph, nombre, malla, parent, transform, color, emisivo=0.0):
    """Agrega una pieza opaca al grafo y la anota para el contador del panel."""
    graph.add_object(
        nombre, malla, "solido", parent=parent, transform=transform,
        color_instancia=np.asarray(color, dtype=np.float32),
        emisivo=float(emisivo),
    )
    PIEZAS.append(nombre)


def color_caballito(indice, total):
    """Un tono distinto por caballito, recorriendo el circulo de matices."""
    matiz = (indice / total + 0.08) % 1.0
    return np.array(colorsys.hsv_to_rgb(matiz, 0.55, 0.95), dtype=np.float32)


def matriz_caballito(indice, total, tiempo, orden):
    """Compone los tres factores locales de un caballito en el orden pedido.

    colocacion -- lo lleva a su lugar sobre la plataforma (angulo y radio)
    oscilacion -- lo sube y lo baja sobre el poste
    pose       -- lo orienta hacia donde avanza y lo inclina al galopar
    """
    angulo = 2 * np.pi * indice / total
    fase = VELOCIDAD_OSCILACION * tiempo + angulo

    colocacion = tr.rotationY(angulo) @ tr.translate(RADIO_CABALLITOS, 0, 0)
    oscilacion = tr.translate(0, ALTURA_MONTURA + AMPLITUD_OSCILACION * np.sin(fase), 0)
    # el caballito mira hacia donde avanza: la colocacion deja su eje X
    # apuntando hacia afuera del carrusel, y este cuarto de giro lo pone
    # tangente, en el sentido en que gira la plataforma
    pose = tr.rotationY(np.pi / 2) @ tr.rotationZ(AMPLITUD_GALOPE * np.cos(fase))

    if orden == "correcto":
        return colocacion @ oscilacion @ pose
    if orden == "pose primero":
        return pose @ colocacion @ oscilacion
    return colocacion @ pose @ oscilacion


def agregar_caballito(graph, indice, total):
    """Arma el subgrafo de un caballito: cuerpo, cuello, cabeza, cola y patas.

    El caballito no es una malla: es un subgrafo de siete piezas que cuelgan de
    un nodo de transformacion. Ese subgrafo se construye una vez por caballito,
    aunque las mallas se suban a la GPU una sola vez.
    """
    nodo = f"caballito_{indice}"
    color = color_caballito(indice, total)
    color_oscuro = color * 0.72

    graph.add_transform(nodo, tr.identity())
    graph.add_edge("carrusel", nodo)

    piezas = [
        ("cuerpo", "cubo", tr.scale(0.38, 0.19, 0.18), color),
        ("cuello", "cubo",
         tr.translate(0.34, 0.23, 0) @ tr.rotationZ(-0.5) @ tr.scale(0.08, 0.20, 0.11),
         color),
        ("cabeza", "cubo",
         tr.translate(0.50, 0.42, 0) @ tr.scale(0.15, 0.08, 0.09), color),
        ("cola", "cubo",
         tr.translate(-0.42, 0.19, 0) @ tr.rotationZ(0.8) @ tr.scale(0.14, 0.045, 0.045),
         color_oscuro),
    ]
    for sufijo, malla, transform, color_pieza in piezas:
        agregar_solido(graph, f"{nodo}_{sufijo}", malla, nodo, transform, color_pieza)

    for numero_pata, (x, z) in enumerate([(0.24, 0.11), (0.24, -0.11),
                                          (-0.24, 0.11), (-0.24, -0.11)]):
        agregar_solido(
            graph, f"{nodo}_pata_{numero_pata}", "cilindro", nodo,
            tr.translate(x, -0.26, z) @ tr.scale(0.042, 0.14, 0.042),
            color_oscuro,
        )


@click.command("carrusel", short_help="Grafo de escena: orden de composicion y cupulas translucidas")
@click.option("--width", type=int, default=1100)
@click.option("--height", type=int, default=760)
@click.option("--caballitos", type=int, default=N_CABALLITOS_POR_OMISION,
              help="cuantos caballitos van sobre la plataforma")
def carrusel(width, height, caballitos):
    window = pyglet.window.Window(width, height)
    carpeta = Path(os.path.dirname(__file__))
    PIEZAS.clear()

    graph = Scenegraph("mundo")

    # sin rezero ni normalize las primitivas quedan de -1 a 1 en cada eje, asi
    # que los argumentos de tr.scale son medias dimensiones y las constantes de
    # arriba miden lo que dicen. Con los valores por omision, `_node_from_file`
    # escala por la diagonal de la caja y todo saldria a 1/raiz(3) del tamano.
    for nombre, archivo in [("cubo", "cube.off"), ("cilindro", "cylinder.off"),
                            ("cono", "cone.off"), ("esfera", "sphere.off")]:
        graph.load_and_register_mesh(nombre, f"assets/{archivo}",
                                     rezero=False, normalize=False)

    graph.register_pipeline(
        "solido",
        load_pipeline(carpeta / "vertex_program.glsl", carpeta / "fragment_program.glsl"),
    )
    # las dos cupulas tienen su propio pipeline aunque compartan los shaders.
    # el grafo dibuja los nodos en el orden en que se insertaron, asi que dar a
    # cada cupula un pipeline distinto es lo que permite elegir por cuadro cual
    # se dibuja primero. Un motor de verdad no hace esto: mantiene una cola de
    # objetos transparentes aparte del grafo, y la ordena.
    for nombre_vidrio in ("vidrio_a", "vidrio_b"):
        graph.register_pipeline(
            nombre_vidrio,
            load_pipeline(carpeta / "vertex_program.glsl",
                          carpeta / "vidrio_fragment_program.glsl"),
        )

    agregar_solido(graph, "piso", "cilindro", "mundo",
                   tr.translate(0, -0.03, 0) @ tr.scale(6.0, 0.03, 6.0), COLOR_PISO)

    # todo lo que cuelga de este nodo gira con el carrusel
    graph.add_transform("carrusel", tr.identity())
    graph.add_edge("mundo", "carrusel")

    agregar_solido(graph, "plataforma", "cilindro", "carrusel",
                   tr.translate(0, 0.06, 0) @ tr.scale(RADIO_PLATAFORMA, 0.06, RADIO_PLATAFORMA),
                   COLOR_PLATAFORMA)
    agregar_solido(graph, "eje", "cilindro", "carrusel",
                   tr.translate(0, ALTURA_TECHO / 2, 0) @ tr.scale(0.09, ALTURA_TECHO / 2, 0.09),
                   COLOR_METAL)

    # el techo es un nodo de transformacion sin geometria propia: el cono y la
    # ampolleta cuelgan de el. Si el techo llevara la escala del cono, la
    # ampolleta la heredaria y saldria aplastada.
    graph.add_transform("techo", tr.translate(0, ALTURA_TECHO, 0))
    graph.add_edge("carrusel", "techo")
    agregar_solido(graph, "techo_cono", "cono", "techo",
                   tr.translate(0, 0.26, 0) @ tr.scale(RADIO_PLATAFORMA, 0.42, RADIO_PLATAFORMA),
                   COLOR_TECHO)
    agregar_solido(graph, "ampolleta", "esfera", "techo",
                   tr.translate(0, -0.24, 0) @ tr.uniformScale(0.11),
                   COLOR_AMPOLLETA, emisivo=1.0)

    largo_poste = ALTURA_TECHO / 2
    for indice in range(caballitos):
        angulo = 2 * np.pi * indice / caballitos
        agregar_solido(
            graph, f"poste_{indice}", "cilindro", "carrusel",
            (tr.rotationY(angulo)
             @ tr.translate(RADIO_CABALLITOS, largo_poste, 0)
             @ tr.scale(0.022, largo_poste, 0.022)),
            COLOR_METAL,
        )
        agregar_caballito(graph, indice, caballitos)

    # las cupulas van sobre dos caballitos vecinos, asi se solapan en pantalla
    # dos veces por vuelta
    for indice, nombre_vidrio in list(zip(range(caballitos), ("vidrio_a", "vidrio_b"))):
        graph.add_object(
            f"cupula_{indice}", "esfera", nombre_vidrio,
            parent=f"caballito_{indice}",
            transform=tr.translate(0, 0.06, 0) @ tr.uniformScale(0.70),
            color_instancia=COLOR_VIDRIO,
        )

    n_cupulas = min(caballitos, 2)

    graph.register_view_transform(
        tr.lookAt(POSICION_CAMARA, PUNTO_MIRADO, np.array([0.0, 1.0, 0.0]))
    )
    graph.set_global_attributes(
        projection=tr.perspective(45, width / height, 0.1, 40.0),
        posicion_camara=POSICION_CAMARA,
    )

    state = {
        "tiempo": 0.0,
        "orden": ORDENES[0],
        "ordenar_cupulas": True,
        "escribir_profundidad": False,
        "velocidad": VELOCIDAD_CARRUSEL,
    }

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(18, 18, 24))
        .add("orden")
        .add("formula")
        .add("cupulas")
        .add("profundidad")
        .add("buffers")
        .footer("O orden de composicion   T ordenar cupulas   M escribir profundidad   , . velocidad   R reset")
    )

    def apply_state():
        panel["orden"] = f"orden de composicion: {state['orden']}"
        panel["formula"] = f"  transformacion local: {FORMULA_ORDEN[state['orden']]}"
        panel.color("formula", VERDE if state["orden"] == "correcto" else ROJO)
        panel["cupulas"] = (
            "cupulas: ordenadas de atras hacia adelante"
            if state["ordenar_cupulas"]
            else "cupulas: en el orden en que estan en el grafo"
        )
        panel.color("cupulas", VERDE if state["ordenar_cupulas"] else ROJO)
        panel["profundidad"] = (
            "escritura al depth buffer durante las cupulas: activada"
            if state["escribir_profundidad"]
            else "escritura al depth buffer durante las cupulas: apagada"
        )
        panel.color("profundidad", ROJO if state["escribir_profundidad"] else VERDE)
        panel["buffers"] = (
            f"4 mallas, {len(PIEZAS) + n_cupulas} piezas, "
            f"{graph.unique_gpu_buffers()} buffers en GPU"
        )

    def orden_de_las_cupulas():
        """Devuelve los pipelines de vidrio, del mas lejano al mas cercano."""
        pipelines = ["vidrio_a", "vidrio_b"][:n_cupulas]
        if not state["ordenar_cupulas"] or len(pipelines) < 2:
            return pipelines

        distancias = {
            nombre: np.linalg.norm(
                graph.get_global_position(f"cupula_{indice}") - POSICION_CAMARA
            )
            for indice, nombre in enumerate(pipelines)
        }
        return sorted(distancias, key=distancias.get, reverse=True)

    apply_state()

    @window.event
    def on_draw():
        GL.glClearColor(0.07, 0.08, 0.12, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        window.clear()

        graph.calculate_global_transforms()
        graph.set_global_attributes(
            posicion_ampolleta=graph.get_global_position("ampolleta").astype(np.float32)
        )

        # primera pasada: todo lo opaco, con el depth buffer haciendo su trabajo
        graph.render(recalculate_transforms=False, only_pipelines={"solido"})

        # segunda pasada: las cupulas, de atras hacia adelante y sin escribir
        # profundidad, para que cada una deje ver lo que quedo detras
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_TRUE if state["escribir_profundidad"] else GL.GL_FALSE)

        for nombre_vidrio in orden_de_las_cupulas():
            graph.render(recalculate_transforms=False, only_pipelines={nombre_vidrio})

        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        with ui_overlay():
            panel.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.O:
            siguiente = (ORDENES.index(state["orden"]) + 1) % len(ORDENES)
            state["orden"] = ORDENES[siguiente]
        elif symbol == pyglet.window.key.T:
            state["ordenar_cupulas"] = not state["ordenar_cupulas"]
        elif symbol == pyglet.window.key.M:
            state["escribir_profundidad"] = not state["escribir_profundidad"]
        elif symbol == pyglet.window.key.COMMA:
            state["velocidad"] = max(0.0, state["velocidad"] - 0.15)
        elif symbol == pyglet.window.key.PERIOD:
            state["velocidad"] = min(2.5, state["velocidad"] + 0.15)
        elif symbol == pyglet.window.key.R:
            state.update(tiempo=0.0, orden=ORDENES[0], ordenar_cupulas=True,
                         escribir_profundidad=False, velocidad=VELOCIDAD_CARRUSEL)
        apply_state()

    def update_world(dt, _):
        state["tiempo"] += dt

        # el giro del conjunto vive en un solo nodo: los caballitos, los postes,
        # el techo y la ampolleta lo heredan sin que nadie los toque
        graph.nodes["carrusel"]["transform"] = tr.rotationY(
            state["velocidad"] * state["tiempo"]
        )

        for indice in range(caballitos):
            graph.nodes[f"caballito_{indice}"]["transform"] = matriz_caballito(
                indice, caballitos, state["tiempo"], state["orden"]
            )

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
