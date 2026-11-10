"""Ejemplo atomico de cinematica inversa: una cadena 2D que alcanza un objetivo.

La cinematica directa va de los angulos de las articulaciones a la posicion de
la punta. La inversa resuelve el problema al reves: dada la posicion que debe
alcanzar la punta (el objetivo), busca los angulos. Con mas de dos huesos no hay
una formula cerrada y en general hay muchas soluciones, asi que se itera desde
la pose actual. El ejemplo compara dos metodos sobre la misma cadena:

- CCD (Cyclic Coordinate Descent): recorre las articulaciones de la punta a la
  raiz y gira cada una para que la punta quede sobre la recta que va de esa
  articulacion al objetivo. Cambia un angulo por vez.
- FABRIK (Forward And Backward Reaching Inverse Kinematics): trabaja con las
  posiciones. Pone la punta en el objetivo y arrastra la cadena hacia la raiz
  conservando el largo de cada hueso; despues devuelve la raiz a su lugar y
  arrastra la cadena hacia la punta.

Las dos funciones de iteracion operan sobre las posiciones de las articulaciones.
En CCD, girar la articulacion j es rotar en torno a ella todas las que vienen
despues, que es lo mismo que cambiar su angulo en la cinematica directa.

Es 2D y se dibuja con pyglet.shapes, sin shaders, para dejar a la vista solo la
cadena. El circulo gris es el alcance: la suma de los largos de los huesos. Un
objetivo fuera de el no se puede alcanzar, y la cadena queda estirada hacia el.

Controles:
- clic o arrastre: mueve el objetivo.
- M:        cambia entre CCD y FABRIK, con la cadena recta otra vez.
- espacio:  alterna entre resolver en cada cuadro y avanzar paso a paso.
- N:        una iteracion (en paso a paso). La pose anterior queda en gris.
- R:        reinicia la cadena.
- ESC:      cierra la ventana.
"""

import click
import numpy as np
import pyglet

from grafica.ui import InfoPanel

LARGOS = np.array([130.0, 110.0, 90.0, 70.0, 50.0])  # largo de cada hueso, en pixeles.
TOLERANCIA = 0.5            # distancia (en pixeles) a la que se da por alcanzado el objetivo.
ITERACIONES_POR_CUADRO = 30  # tope de iteraciones al resolver en cada cuadro.

COLOR_HUESO = (90, 200, 235)
COLOR_ARTICULACION = (235, 235, 240)
COLOR_RAIZ = (235, 160, 60)
COLOR_OBJETIVO = (235, 90, 170)
COLOR_ANTERIOR = (95, 100, 120)
COLOR_ALCANCE = (60, 64, 80)

METODOS = ["CCD", "FABRIK"]


def cadena_estirada(raiz):
    """Las posiciones de las articulaciones con la cadena recta hacia arriba."""
    alturas = np.concatenate([[0.0], np.cumsum(LARGOS)])
    return np.array([[raiz[0], raiz[1] + h] for h in alturas])


def unitario(vector):
    return vector / np.linalg.norm(vector)


def rotar(vectores, angulo):
    c, s = np.cos(angulo), np.sin(angulo)
    return vectores @ np.array([[c, s], [-s, c]])


def iteracion_ccd(posiciones, objetivo):
    """Una pasada de CCD, de la ultima articulacion a la raiz."""
    p = posiciones.copy()
    for j in range(len(p) - 2, -1, -1):
        a = p[-1] - p[j]      # de la articulacion j a la punta
        b = objetivo - p[j]   # de la articulacion j al objetivo
        # angulo con signo de a hasta b: atan2 de su producto cruz y su producto punto
        giro = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)
        # girar la articulacion j mueve a todas las que vienen despues
        p[j + 1:] = p[j] + rotar(p[j + 1:] - p[j], giro)
    return p


def iteracion_fabrik(posiciones, objetivo):
    """Una pasada de FABRIK: de la punta a la raiz y de vuelta."""
    p = posiciones.copy()
    raiz = p[0].copy()
    p[-1] = objetivo
    for i in range(len(p) - 2, -1, -1):   # de la punta a la raiz
        p[i] = p[i + 1] + LARGOS[i] * unitario(p[i] - p[i + 1])
    p[0] = raiz
    for i in range(len(p) - 1):           # de la raiz a la punta
        p[i + 1] = p[i] + LARGOS[i] * unitario(p[i + 1] - p[i])
    return p


ITERACION = {"CCD": iteracion_ccd, "FABRIK": iteracion_fabrik}


@click.command("cinematica_inversa", short_help="Cinematica inversa 2D: CCD y FABRIK")
@click.option("--width", type=int, default=1000)
@click.option("--height", type=int, default=720)
def cinematica_inversa(width, height):
    window = pyglet.window.Window(width, height, caption="Cinematica inversa: CCD y FABRIK")
    raiz = np.array([width / 2, 110.0])

    state = {
        "metodo": 0,
        "paso_a_paso": False,
        "posiciones": cadena_estirada(raiz),
        "anterior": None,
        "objetivo": raiz + np.array([260.0, 240.0]),
        "iteraciones": 0,
    }

    panel = (
        InfoPanel(x=20, y_top=height - 25, color=(220, 220, 220, 255), background=(20, 22, 30),
                  background_width=430)
        .add("metodo", size=14)
        .add("modo")
        .add("iteraciones")
        .add("distancia")
        .add("alcance")
        .footer("clic mueve el objetivo   M metodo   espacio paso a paso   N iteracion   R reinicia")
    )

    def distancia():
        return float(np.linalg.norm(state["posiciones"][-1] - state["objetivo"]))

    def iterar():
        """Una iteracion del metodo actual; guarda la pose anterior para dibujarla."""
        state["anterior"] = state["posiciones"]
        iteracion = ITERACION[METODOS[state["metodo"]]]
        state["posiciones"] = iteracion(state["posiciones"], state["objetivo"])
        state["iteraciones"] += 1

    def resolver():
        """Itera hasta alcanzar el objetivo, hasta el tope o hasta que la punta deja de acercarse."""
        state["anterior"] = None
        for _ in range(ITERACIONES_POR_CUADRO):
            if distancia() < TOLERANCIA:
                break
            antes = distancia()
            iterar()
            if antes - distancia() < 1e-3:
                break
        state["anterior"] = None

    def apply_state():
        alcance = LARGOS.sum()
        lejos = np.linalg.norm(state["objetivo"] - raiz)
        panel["metodo"] = f"metodo: {METODOS[state['metodo']]}"
        panel["modo"] = "modo: " + ("paso a paso (N)" if state["paso_a_paso"] else "resuelve en cada cuadro")
        panel["iteraciones"] = f"iteraciones desde el ultimo objetivo: {state['iteraciones']}"
        panel["distancia"] = f"distancia de la punta al objetivo: {distancia():.1f} px"
        panel["alcance"] = (f"alcance de la cadena: {alcance:.0f} px; objetivo a {lejos:.0f} px"
                            + ("" if lejos <= alcance else " (fuera de alcance)"))

    def mover_objetivo(x, y):
        state["objetivo"] = np.array([float(x), float(y)])
        state["iteraciones"] = 0
        state["anterior"] = None
        if not state["paso_a_paso"]:
            resolver()
        apply_state()

    def dibujar_cadena(batch, refs, posiciones, color, grosor):
        # pyglet.shapes espera floats de Python, no de numpy
        puntos = posiciones.tolist()
        for a, b in zip(puntos[:-1], puntos[1:]):
            refs.append(pyglet.shapes.Line(*a, *b, thickness=grosor, color=color, batch=batch))
        for p in puntos[1:]:
            refs.append(pyglet.shapes.Circle(*p, grosor * 0.9, color=color, batch=batch))

    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        refs = []
        x_raiz, y_raiz = raiz.tolist()
        refs.append(pyglet.shapes.Arc(x_raiz, y_raiz, float(LARGOS.sum()), segments=180,
                                      thickness=2, color=COLOR_ALCANCE, batch=batch))
        if state["anterior"] is not None:
            dibujar_cadena(batch, refs, state["anterior"], COLOR_ANTERIOR, 6)
        dibujar_cadena(batch, refs, state["posiciones"], COLOR_HUESO, 8)
        for p in state["posiciones"][1:-1].tolist():
            refs.append(pyglet.shapes.Circle(*p, 5, color=COLOR_ARTICULACION, batch=batch))
        refs.append(pyglet.shapes.Circle(x_raiz, y_raiz, 11, color=COLOR_RAIZ, batch=batch))
        objetivo = state["objetivo"].tolist()
        refs.append(pyglet.shapes.Circle(*objetivo, 10, color=COLOR_OBJETIVO, batch=batch))
        refs.append(pyglet.shapes.Circle(*objetivo, 5, color=(20, 22, 30), batch=batch))
        batch.draw()
        panel.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        mover_objetivo(x, y)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        mover_objetivo(x, y)

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.M:
            # al cambiar de metodo la cadena vuelve a estar recta, para que las
            # iteraciones de los dos metodos se cuenten desde la misma pose
            state["metodo"] = (state["metodo"] + 1) % len(METODOS)
            state["posiciones"] = cadena_estirada(raiz)
            state["anterior"] = None
            state["iteraciones"] = 0
            if not state["paso_a_paso"]:
                resolver()
        elif symbol == key.SPACE:
            state["paso_a_paso"] = not state["paso_a_paso"]
            state["anterior"] = None
            if not state["paso_a_paso"]:
                resolver()
        elif symbol == key.N and state["paso_a_paso"]:
            iterar()
        elif symbol == key.R:
            state["posiciones"] = cadena_estirada(raiz)
            state["anterior"] = None
            state["iteraciones"] = 0
            if not state["paso_a_paso"]:
                resolver()
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    resolver()
    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    cinematica_inversa()
