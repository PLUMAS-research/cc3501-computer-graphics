"""Ejemplo atomico de mallas: las operaciones elementales sobre half-edge.

Aisla las tres operaciones que modifican una malla localmente: flip (dar vuelta
una arista), split (partir una arista) y collapse (fundir una arista en un
punto). Son los bloques con que se construyen la subdivision y la
simplificacion.

La malla es una grilla triangulada 2D, plana, dibujada con pyglet.shapes (sin
shaders) para dejar a la vista solo la topologia. El alumno hace clic en una
arista para seleccionar una semi-arista (se ve su direccion y la de su gemela)
y aplica una operacion con el teclado. El panel muestra los conteos de
vertices, aristas y caras, la caracteristica de Euler V - E + F (invariante
ante las tres operaciones) y si la malla sigue siendo manifold.

La estructura half-edge y las tres operaciones viven en grafica/half_edge.py;
este archivo solo selecciona, dibuja y reporta.

Controles:
- clic:     selecciona la arista mas cercana al cursor.
- TAB:      alterna entre las dos semi-aristas de la arista (su gemela).
- F:        flip de la arista seleccionada (solo interior).
- S:        split: inserta un vertice en el punto medio.
- C:        collapse: funde la arista en un punto (solo si preserva manifold).
- R:        reinicia la grilla.
- ESC:      cierra la ventana.
"""

import click
import numpy as np
import pyglet

from grafica.half_edge import BORDE, HalfEdgeMesh
from grafica.ui import InfoPanel

FONDO_CARA = (60, 70, 95)        # relleno translucido de las caras.
COLOR_ARISTA = (150, 160, 180)   # aristas interiores.
COLOR_BORDE = (230, 170, 60)     # aristas y vertices de borde.
COLOR_VERTICE = (170, 200, 235)
COLOR_SELECCION = (90, 230, 235)  # semi-arista seleccionada.
COLOR_GEMELA = (45, 120, 130)     # su gemela, mas tenue.


def _grilla_en_pixeles(columnas, filas, separacion, origen):
    """Grilla triangulada en coordenadas de pantalla. Devuelve (posiciones, caras)."""
    posiciones = []
    for fila in range(filas):
        for columna in range(columnas):
            posiciones.append((origen[0] + columna * separacion,
                               origen[1] + fila * separacion))
    indice = lambda f, c: f * columnas + c
    caras = []
    for f in range(filas - 1):
        for c in range(columnas - 1):
            caras.append((indice(f, c), indice(f, c + 1), indice(f + 1, c)))
            caras.append((indice(f, c + 1), indice(f + 1, c + 1), indice(f + 1, c)))
    return np.array(posiciones, dtype=np.float64), np.array(caras, dtype=np.int64)


def _distancia_punto_segmento(punto, extremo_a, extremo_b):
    """Distancia del punto al segmento [a, b], todo en 2D."""
    segmento = extremo_b - extremo_a
    largo_cuadrado = float(segmento @ segmento)
    if largo_cuadrado == 0.0:
        return float(np.linalg.norm(punto - extremo_a))
    t = np.clip(float((punto - extremo_a) @ segmento) / largo_cuadrado, 0.0, 1.0)
    proyeccion = extremo_a + t * segmento
    return float(np.linalg.norm(punto - proyeccion))


@click.command("operaciones_malla", short_help="Operaciones elementales half-edge: flip/split/collapse")
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=700)
@click.option("--columnas", type=int, default=6, help="columnas de vertices de la grilla")
@click.option("--filas", type=int, default=5, help="filas de vertices de la grilla")
def operaciones_malla(width, height, columnas, filas):
    window = pyglet.window.Window(width, height, caption="operaciones elementales en half-edge")

    separacion = min((width - 260) / (columnas - 1), (height - 160) / (filas - 1))
    origen = (130, 110)

    def construir_malla():
        return HalfEdgeMesh.from_faces(*_grilla_en_pixeles(columnas, filas, separacion, origen))

    malla = construir_malla()
    state = {"semi_arista": 0, "ultima_accion": "selecciona una arista y aplica F, S o C"}

    panel = (
        InfoPanel(x=20, y_top=height - 25, color=(220, 220, 220, 255), background=(25, 28, 38),
                  background_width=420)
        .add("conteos", size=14)
        .add("euler")
        .add("manifold")
        .add("seleccion")
        .add("accion")
        .footer("clic arista   TAB gemela   F flip   S split   C collapse   R reset")
    )

    def primera_semi_arista_viva():
        return next(h for h in range(len(malla.he_to)) if malla.he_alive[h])

    def punto(vertice):
        return malla.positions[vertice][:2]

    def apply_state():
        """Vuelca el estado de la malla a las etiquetas del panel.

        Unico lugar que convierte topologia en texto: cada operacion del
        teclado modifica la malla y llama aqui, asi el panel nunca miente.
        """
        v, e, f = malla.n_vertices(), malla.n_edges(), malla.n_faces()
        panel["conteos"] = f"vertices V={v}   aristas E={e}   caras F={f}"
        panel["euler"] = f"caracteristica de Euler  V - E + F = {v - e + f}"
        panel["manifold"] = f"manifold: {'si' if malla.is_manifold() else 'NO'}"

        h = state["semi_arista"]
        interior = not malla.is_boundary_edge(h)
        puede_flip = interior
        puede_collapse = malla.can_collapse(h)
        panel["seleccion"] = (
            f"semi-arista {malla.tail(h)} -> {malla.head(h)}   "
            f"flip: {'si' if puede_flip else 'no'}   "
            f"collapse: {'si' if puede_collapse else 'no'}"
        )
        panel["accion"] = state["ultima_accion"]
        print(f"[operaciones_malla] V={v} E={e} F={f} euler={v - e + f}  {state['ultima_accion']}")

    def seleccionar_cercana(x, y):
        cursor = np.array([x, y], dtype=np.float64)
        mejor, mejor_distancia = None, float("inf")
        for h in range(len(malla.he_to)):
            if not malla.he_alive[h]:
                continue
            distancia = _distancia_punto_segmento(cursor, punto(malla.tail(h)), punto(malla.head(h)))
            # ante empate (las dos gemelas comparten segmento) gana el indice menor;
            # TAB permite saltar a la otra semi-arista.
            if distancia < mejor_distancia - 1e-9:
                mejor, mejor_distancia = h, distancia
        if mejor is not None:
            state["semi_arista"] = mejor

    def _flecha(extremo_a, extremo_b, desplazamiento, color, grosor, batch, refs):
        """Dibuja una semi-arista como flecha desplazada hacia su izquierda."""
        direccion = extremo_b - extremo_a
        largo = float(np.linalg.norm(direccion))
        if largo < 1e-6:
            return
        direccion /= largo
        izquierda = np.array([-direccion[1], direccion[0]])
        a = extremo_a + izquierda * desplazamiento
        b = extremo_b + izquierda * desplazamiento
        refs.append(pyglet.shapes.Line(a[0], a[1], b[0], b[1], thickness=grosor, color=color, batch=batch))
        base = b - direccion * 14
        for signo in (1, -1):
            punta = base + izquierda * signo * 7
            refs.append(pyglet.shapes.Line(b[0], b[1], punta[0], punta[1], thickness=grosor,
                                           color=color, batch=batch))

    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        refs = []

        # caras: relleno translucido para ver la superficie y sus cambios.
        for f in range(len(malla.f_he)):
            if not malla.f_alive[f]:
                continue
            h = malla.f_he[f]
            p0, p1, p2 = punto(malla.tail(h)), punto(malla.head(h)), punto(malla.head(malla.he_next[h]))
            triangulo = pyglet.shapes.Triangle(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1],
                                               color=FONDO_CARA, batch=batch)
            triangulo.opacity = 90
            refs.append(triangulo)

        # aristas: una linea por arista (la semi-arista de indice menor la dibuja).
        for h in range(len(malla.he_to)):
            if not malla.he_alive[h]:
                continue
            gemela = malla.he_twin[h]
            if gemela != BORDE and gemela < h:
                continue
            color = COLOR_BORDE if gemela == BORDE else COLOR_ARISTA
            a, b = punto(malla.tail(h)), punto(malla.head(h))
            refs.append(pyglet.shapes.Line(a[0], a[1], b[0], b[1], thickness=1.5, color=color, batch=batch))

        # semi-arista seleccionada (y su gemela, mas tenue), como flechas opuestas.
        h = state["semi_arista"]
        if malla.he_alive[h]:
            self_a, self_b = punto(malla.tail(h)), punto(malla.head(h))
            gemela = malla.he_twin[h]
            if gemela != BORDE:
                _flecha(punto(malla.tail(gemela)), punto(malla.head(gemela)), 6,
                        COLOR_GEMELA, 2.0, batch, refs)
            _flecha(self_a, self_b, 6, COLOR_SELECCION, 3.0, batch, refs)

        # vertices.
        for v in range(len(malla.positions)):
            if not malla.v_alive[v]:
                continue
            color = COLOR_VERTICE if malla.is_interior_vertex(v) else COLOR_BORDE
            p = punto(v)
            refs.append(pyglet.shapes.Circle(p[0], p[1], 5, color=color, batch=batch))

        batch.draw()
        panel.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        seleccionar_cercana(x, y)
        apply_state()

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal malla
        key = pyglet.window.key
        h = state["semi_arista"]
        if symbol == key.TAB:
            gemela = malla.he_twin[h]
            if gemela != BORDE:
                state["semi_arista"] = gemela
                state["ultima_accion"] = "saltaste a la semi-arista gemela"
        elif symbol == key.F:
            if malla.flip(h):
                # h sigue viva: ahora es la diagonal en su nueva posicion.
                state["ultima_accion"] = "flip aplicado: la diagonal cambio de lugar"
            else:
                state["ultima_accion"] = "flip rechazado: la arista esta en el borde"
        elif symbol == key.S:
            malla.split(h)
            state["semi_arista"] = primera_semi_arista_viva()
            state["ultima_accion"] = "split aplicado: nuevo vertice en el punto medio"
        elif symbol == key.C:
            if malla.collapse(h):
                state["semi_arista"] = primera_semi_arista_viva()
                state["ultima_accion"] = "collapse aplicado: la arista se fundio en un punto"
            else:
                state["ultima_accion"] = "collapse rechazado: rompe manifold o toca el borde"
        elif symbol == key.R:
            malla = construir_malla()
            state["semi_arista"] = 0
            state["ultima_accion"] = "grilla reiniciada"
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    operaciones_malla()
