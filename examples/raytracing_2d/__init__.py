"""Trazado de rayos en 2D: los rayos de cada pixel, sus rebotes y sus sombras.

La escena es una vista de costado de `raytracing_basico`: tres esferas (aqui
circulos) sobre un piso a cuadros, una luz y una camara cuyo plano de la imagen
es una columna de pixeles. La imagen resultante es esa columna, ampliada a la
derecha de la ventana.

El trazador es el mismo de `raytracing_basico`, con vectores de dos
componentes: `Esfera`, `Plano`, `interseccion_mas_cercana` y `color_rayo` se
importan sin cambios, y el color de cada pixel sale de `color_rayo`. Lo que
agrega este ejemplo es `recorrer_rayo`, que sigue la misma logica pero en vez
de devolver un color guarda los tramos que recorre el rayo, para dibujarlos.

Colores de los tramos:
- blanco: el rayo de la camara; magenta: los rayos reflejados.
- amarillo: un rayo de sombra que llega a la luz.
- gris: un rayo de sombra que choca con un objeto antes de llegar a la luz.
- tenue: un rayo que no toca nada y sale al cielo.

Controles:
- arrastra:      mueve el ojo, el objetivo de la camara, la luz o una esfera.
- arriba/abajo:  cambia el pixel seleccionado.
- A:             muestra los rayos de todos los pixeles o solo los del seleccionado.
- R:             rebotes de reflexion (0 a 3).
- S:             rayos de sombra.
- , / .:         cantidad de pixeles.
- ESC:           cierra la ventana.
"""

import click
import numpy as np
import pyglet

from grafica.ui import InfoPanel, load_firacode
from examples.raytracing_basico.trazado import (
    Esfera,
    Plano,
    color_rayo,
    interseccion_mas_cercana,
    normalizar,
)

# el mundo se dibuja a ESCALA pixeles por unidad, con su origen en ORIGEN_PANTALLA.
ESCALA = 100.0
ORIGEN_PANTALLA = np.array([70.0, 70.0])
LIMITE_MUNDO = (np.array([-0.6, -0.2]), np.array([10.6, 6.2]))

# los mismos coeficientes que `trazar` en raytracing_basico.
AMBIENTE, DIFUSO, ESPECULAR, BRILLO = 0.10, 0.9, 0.6, 50.0
COLOR_CIELO_ARRIBA = np.array([0.55, 0.70, 0.95])
COLOR_CIELO_ABAJO = np.array([0.85, 0.90, 0.98])

FOV_GRADOS = 50.0
PIXELES = [4, 6, 8, 12, 16, 24, 32]

COLOR_FONDO = (18, 20, 28)
COLOR_RAYO_CAMARA = (235, 235, 245)
COLOR_RAYO_REFLEJADO = (207, 56, 137)
COLOR_SOMBRA_LIBRE = (240, 200, 80)
COLOR_SOMBRA_BLOQUEADA = (120, 120, 140)
COLOR_SELECCION = (207, 56, 137)

# capas de dibujo: la escena atras, los rayos encima y el plano de la imagen al frente.
CAPA_ESCENA = pyglet.graphics.Group(order=0)
CAPA_RAYOS = pyglet.graphics.Group(order=1)
CAPA_FRENTE = pyglet.graphics.Group(order=2)


class Piso(Plano):
    """El `Plano` de raytracing_basico, con el tablero a lo largo de x: en 2D no hay z."""

    def color(self, punto):
        celda = int(np.floor(punto[0] * self.escala_tablero))
        return self.color_a if celda % 2 == 0 else self.color_b


def construir_escena():
    """La escena de raytracing_basico vista de costado, con los mismos colores."""
    return [
        Esfera([2.9, 0.42], 0.42, [0.20, 0.45, 0.90], reflectividad=0.35),
        Esfera([5.0, 0.5], 0.5, [0.30, 0.80, 0.35], reflectividad=0.5),
        Esfera([7.6, 1.15], 1.15, [0.85, 0.20, 0.20], reflectividad=0.35),
        Piso([0.0, 0.0], [0.0, 1.0], color_a=[0.95, 0.95, 0.95], color_b=[0.15, 0.15, 0.15],
             reflectividad=0.15, escala_tablero=0.6),
    ]


def color_fondo(direccion):
    """El degradado del cielo de raytracing_basico, segun la altura del rayo."""
    t = 0.5 * (direccion[1] + 1.0)
    return (1.0 - t) * COLOR_CIELO_ABAJO + t * COLOR_CIELO_ARRIBA


def rayos_de_la_camara(ojo, objetivo, cantidad_pixeles):
    """Origen y direccion del rayo de cada pixel, de la fila de arriba a la de abajo.

    Es `Camara.rayo` en 2D: el plano de la imagen esta a distancia 1 del ojo y
    tiene una sola columna, asi que solo queda la coordenada v.
    """
    adelante = normalizar(objetivo - ojo)
    arriba = np.array([-adelante[1], adelante[0]])
    semialto = np.tan(np.radians(FOV_GRADOS) / 2.0)
    rayos = []
    for fila in range(cantidad_pixeles):
        v = 1.0 - 2.0 * (fila + 0.5) / cantidad_pixeles
        punto_plano = ojo + adelante + v * semialto * arriba
        rayos.append((ojo, normalizar(punto_plano - ojo)))
    return rayos


def hasta_el_borde(origen, direccion):
    """Distancia hasta que el rayo sale del rectangulo del mundo, para dibujar el cielo."""
    minimo, maximo = LIMITE_MUNDO
    with np.errstate(divide="ignore", invalid="ignore"):
        salidas = np.where(direccion > 0, (maximo - origen) / direccion,
                           np.where(direccion < 0, (minimo - origen) / direccion, np.inf))
    return float(max(np.min(salidas), 0.0))


def recorrer_rayo(escena, origen, direccion, luz, usar_sombras, rebotes_restantes, tramos, nivel=0):
    """Agrega a `tramos` los segmentos que recorre un rayo, con la logica de `color_rayo`.

    Cada tramo es (tipo, inicio, fin, nivel). Los tipos son "rayo" (hasta un
    impacto), "cielo" (no toca nada), "sombra" (llega a la luz) y "bloqueado"
    (algo tapa la luz). El nivel es 0 para el rayo de la camara y aumenta en uno
    con cada rebote.
    """
    objeto, distancia = interseccion_mas_cercana(escena, origen, direccion)
    if objeto is None:
        fin = origen + hasta_el_borde(origen, direccion) * direccion
        tramos.append(("cielo", origen, fin, nivel))
        return

    punto = origen + direccion * distancia
    tramos.append(("rayo", origen, punto, nivel))
    normal = objeto.normal(punto)
    punto_desplazado = punto + normal * 1e-4

    if usar_sombras:
        hacia_luz = normalizar(luz["posicion"] - punto)
        distancia_luz = np.linalg.norm(luz["posicion"] - punto)
        # para decidir la sombra basta el primer objeto que se interpone (en_sombra);
        # para dibujar donde se corta el rayo de sombra se busca el mas cercano.
        _, distancia_bloqueo = interseccion_mas_cercana(escena, punto_desplazado, hacia_luz)
        if distancia_bloqueo < distancia_luz:
            tramos.append(("bloqueado", punto, punto_desplazado + distancia_bloqueo * hacia_luz, nivel))
        else:
            tramos.append(("sombra", punto, luz["posicion"], nivel))

    if rebotes_restantes > 0 and objeto.reflectividad > 0.0:
        direccion_reflejada = direccion - 2.0 * np.dot(direccion, normal) * normal
        recorrer_rayo(escena, punto_desplazado, normalizar(direccion_reflejada), luz,
                      usar_sombras, rebotes_restantes - 1, tramos, nivel + 1)


def a_pantalla(punto):
    return ORIGEN_PANTALLA + ESCALA * np.asarray(punto, dtype=float)


def a_mundo(x, y):
    return (np.array([x, y], dtype=float) - ORIGEN_PANTALLA) / ESCALA


def a_rgb(color):
    return tuple(int(255 * c) for c in np.clip(color, 0.0, 1.0))


@click.command("raytracing_2d", short_help="Rayos de cada pixel, rebotes y sombras, en 2D")
@click.option("--width", type=int, default=1280)
@click.option("--height", type=int, default=720)
def raytracing_2d(width, height):
    window = pyglet.window.Window(width, height, caption="Trazado de rayos en 2D")
    load_firacode()

    escena = construir_escena()
    luz = {"posicion": np.array([4.6, 4.3]), "color": np.array([1.0, 1.0, 1.0])}
    state = {
        "ojo": np.array([0.0, 1.6]),
        "objetivo": np.array([4.0, 0.95]),
        "rebotes": 2,
        "sombras": True,
        "indice_pixeles": PIXELES.index(12),
        "pixel": 5,
        "todos": True,
        "arrastrando": None,
        # resultado del trazado: un color y una lista de tramos por pixel.
        "colores": [],
        "tramos": [],
    }

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=COLOR_FONDO, background_width=620)
        .add("rebotes")
        .add("sombras")
        .add("pixeles")
        .add("seleccionado")
        .add("total")
        .footer("arrastra ojo, objetivo, luz o esferas   arriba/abajo pixel   "
                "A todos/uno   R rebotes   S sombras   , . pixeles")
    )

    def trazar():
        """Traza cada pixel: el color con color_rayo y los tramos con recorrer_rayo."""
        cantidad = PIXELES[state["indice_pixeles"]]
        state["pixel"] = min(state["pixel"], cantidad - 1)
        state["colores"], state["tramos"] = [], []
        for origen, direccion in rayos_de_la_camara(state["ojo"], state["objetivo"], cantidad):
            color = color_rayo(escena, origen, direccion, luz, state["ojo"], AMBIENTE, DIFUSO,
                               ESPECULAR, BRILLO, color_fondo, state["sombras"], state["rebotes"])
            tramos = []
            recorrer_rayo(escena, origen, direccion, luz, state["sombras"], state["rebotes"], tramos)
            state["colores"].append(np.clip(color, 0.0, 1.0))
            state["tramos"].append(tramos)

    def contar(tramos):
        visibilidad = sum(1 for tipo, *_ in tramos if tipo in ("rayo", "cielo"))
        sombra = sum(1 for tipo, *_ in tramos if tipo in ("sombra", "bloqueado"))
        return visibilidad, sombra

    def apply_state(imprimir=True):
        trazar()
        cantidad = len(state["colores"])
        visibilidad, sombra = contar(state["tramos"][state["pixel"]])
        total = sum(sum(contar(tramos)) for tramos in state["tramos"])
        panel["rebotes"] = f"rebotes de reflexion: {state['rebotes']}"
        panel["sombras"] = f"rayos de sombra: {'ON' if state['sombras'] else 'off'}"
        panel["pixeles"] = f"pixeles: {cantidad}"
        panel["seleccionado"] = (f"pixel {state['pixel']}: {visibilidad} rayos de camara y "
                                 f"reflejados, {sombra} de sombra")
        panel["total"] = f"todos los pixeles: {total} rayos ({total / cantidad:.2f} por pixel)"
        if imprimir:
            print(f"[raytracing_2d] rebotes={state['rebotes']} sombras={state['sombras']} "
                  f"pixeles={cantidad} rayos={total}")

    def arrastrables():
        """Puntos que se pueden mover con el mouse, con su radio de captura en el mundo."""
        puntos = [("ojo", state["ojo"], 0.2), ("objetivo", state["objetivo"], 0.2),
                  ("luz", luz["posicion"], 0.25)]
        for objeto in escena:
            if isinstance(objeto, Esfera):
                puntos.append((objeto, objeto.centro, objeto.radio))
        return puntos

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        cursor = a_mundo(x, y)
        state["arrastrando"] = None
        for clave, posicion, radio in arrastrables():
            if np.linalg.norm(cursor - posicion) < radio:
                state["arrastrando"] = clave
                break

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        clave = state["arrastrando"]
        if clave is None:
            return
        cursor = a_mundo(x, y)
        if clave == "luz":
            luz["posicion"] = cursor
        elif isinstance(clave, Esfera):
            clave.centro = cursor
        else:
            state[clave] = cursor
        apply_state(imprimir=False)

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        state["arrastrando"] = None

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.R:
            state["rebotes"] = (state["rebotes"] + 1) % 4
        elif symbol == key.S:
            state["sombras"] = not state["sombras"]
        elif symbol == key.A:
            state["todos"] = not state["todos"]
        elif symbol == key.UP:
            state["pixel"] = max(0, state["pixel"] - 1)
        elif symbol == key.DOWN:
            state["pixel"] = min(PIXELES[state["indice_pixeles"]] - 1, state["pixel"] + 1)
        elif symbol == key.COMMA:
            state["indice_pixeles"] = max(0, state["indice_pixeles"] - 1)
        elif symbol == key.PERIOD:
            state["indice_pixeles"] = min(len(PIXELES) - 1, state["indice_pixeles"] + 1)
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    def dibujar_tramos(tramos, destacado, batch, refs):
        grosor = 3.0 if destacado else 1.5
        opacidad = 255 if destacado else 150
        for tipo, inicio, fin, nivel in tramos:
            if tipo == "sombra":
                color = COLOR_SOMBRA_LIBRE
            elif tipo == "bloqueado":
                color = COLOR_SOMBRA_BLOQUEADA
            else:
                color = COLOR_RAYO_CAMARA if nivel == 0 else COLOR_RAYO_REFLEJADO
            a, b = a_pantalla(inicio), a_pantalla(fin)
            ancho = grosor * (0.6 if tipo in ("sombra", "bloqueado") else 1.0)
            alfa = opacidad // 2 if tipo == "cielo" else opacidad
            refs.append(pyglet.shapes.Line(a[0], a[1], b[0], b[1], thickness=ancho,
                                           color=(*color, alfa), batch=batch, group=CAPA_RAYOS))
            if tipo in ("rayo", "bloqueado"):
                refs.append(pyglet.shapes.Circle(b[0], b[1], 5 if destacado else 3,
                                                 color=(*color, opacidad), batch=batch,
                                                 group=CAPA_RAYOS))

    def dibujar_escena(batch, refs):
        # el piso, con las celdas del tablero que usa Piso.color
        minimo, maximo = LIMITE_MUNDO
        celda = 1.0 / escena[-1].escala_tablero
        x = np.floor(minimo[0] / celda) * celda
        while x < maximo[0]:
            inicio, fin = max(x, minimo[0]), min(x + celda, maximo[0])
            color = escena[-1].color(np.array([(inicio + fin) / 2, 0.0]))
            a = a_pantalla([inicio, -0.25])
            refs.append(pyglet.shapes.Rectangle(a[0], a[1], (fin - inicio) * ESCALA, 0.25 * ESCALA,
                                                color=a_rgb(color), batch=batch, group=CAPA_ESCENA))
            x += celda
        for objeto in escena[:-1]:
            centro = a_pantalla(objeto.centro)
            refs.append(pyglet.shapes.Circle(centro[0], centro[1], objeto.radio * ESCALA,
                                             color=a_rgb(objeto.color_base), batch=batch,
                                             group=CAPA_ESCENA))
        posicion_luz = a_pantalla(luz["posicion"])
        refs.append(pyglet.shapes.Star(posicion_luz[0], posicion_luz[1], 16, 7, 6,
                                       color=COLOR_SOMBRA_LIBRE, batch=batch, group=CAPA_FRENTE))
        ojo, objetivo = a_pantalla(state["ojo"]), a_pantalla(state["objetivo"])
        refs.append(pyglet.shapes.Circle(ojo[0], ojo[1], 8, color=COLOR_RAYO_CAMARA, batch=batch,
                                         group=CAPA_FRENTE))
        refs.append(pyglet.shapes.Circle(objetivo[0], objetivo[1], 6, color=(150, 150, 170),
                                         batch=batch, group=CAPA_FRENTE))
        for texto, posicion in (("ojo", ojo + (-14, 16)), ("objetivo", objetivo + (-30, 12)),
                                ("luz", posicion_luz + (20, 0))):
            refs.append(pyglet.text.Label(texto, font_name="Fira Code", font_size=11,
                                          x=posicion[0], y=posicion[1], color=(200, 200, 210, 255),
                                          batch=batch, group=CAPA_FRENTE))

    def dibujar_plano_y_columna(batch, refs):
        """El plano de la imagen junto al ojo y la misma columna, ampliada, a la derecha."""
        cantidad = len(state["colores"])
        adelante = normalizar(state["objetivo"] - state["ojo"])
        arriba = np.array([-adelante[1], adelante[0]])
        semialto = np.tan(np.radians(FOV_GRADOS) / 2.0)
        centro = state["ojo"] + adelante
        grosor = 0.08
        alto_columna = 0.62 * height
        lado = alto_columna / cantidad
        x_columna, y_columna = width - 110, (height - alto_columna) / 2
        for fila, color in enumerate(state["colores"]):
            v_alto = 1.0 - 2.0 * fila / cantidad
            v_bajo = 1.0 - 2.0 * (fila + 1) / cantidad
            esquinas = [centro + v * semialto * arriba + g * adelante
                        for v, g in ((v_bajo, -grosor), (v_bajo, grosor), (v_alto, grosor),
                                     (v_alto, -grosor))]
            refs.append(pyglet.shapes.Polygon(*[tuple(a_pantalla(p)) for p in esquinas],
                                              color=a_rgb(color), batch=batch, group=CAPA_FRENTE))
            # bordes redondeados al pixel de la ventana, para que las celdas no dejen rendijas
            y = round(y_columna + alto_columna - (fila + 1) * lado)
            alto_celda = round(y_columna + alto_columna - fila * lado) - y
            refs.append(pyglet.shapes.Rectangle(x_columna, y, 60, alto_celda, color=a_rgb(color),
                                                batch=batch, group=CAPA_ESCENA))
            if fila == state["pixel"]:
                refs.append(pyglet.shapes.Box(x_columna - 3, y - 3, 66, alto_celda + 6, thickness=3,
                                              color=COLOR_SELECCION, batch=batch,
                                              group=CAPA_FRENTE))
        refs.append(pyglet.text.Label("imagen", font_name="Fira Code", font_size=11,
                                      x=x_columna + 30, y=y_columna + alto_columna + 14,
                                      anchor_x="center", color=(200, 200, 210, 255), batch=batch,
                                      group=CAPA_FRENTE))

    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        refs = []
        dibujar_escena(batch, refs)
        for fila, tramos in enumerate(state["tramos"]):
            destacado = fila == state["pixel"]
            if state["todos"] or destacado:
                dibujar_tramos(tramos, destacado, batch, refs)
        dibujar_plano_y_columna(batch, refs)
        batch.draw()
        panel.draw()

    pyglet.gl.glClearColor(*(c / 255 for c in COLOR_FONDO), 1.0)
    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    raytracing_2d()
