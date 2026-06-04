"""Ejemplo atomico de curvas parametricas: Hermite, Catmull-Rom y Bezier.

Aisla el nucleo de la unidad: una curva cubica definida por cuatro puntos de
control. Los mismos cuatro puntos se interpretan de tres maneras distintas, asi
el alumno compara las tres familias sobre la misma configuracion:

- Bezier: los cuatro puntos son los puntos de control. La curva pasa por P0 y
  P3, queda contenida en la envoltura convexa, y se evalua con De Casteljau
  (interpolaciones lineales sucesivas, visibles al mover el parametro t).
- Catmull-Rom: el segmento interpola P1 y P2; P0 y P3 son los vecinos con que
  se calculan las tangentes t1 = (P2 - P0) / 2 y t2 = (P3 - P1) / 2.
- Hermite: P0 y P3 son los extremos; las tangentes salen de los handles P1 y
  P2 como t0 = P1 - P0 y t1 = P3 - P2. La magnitud del handle cambia la curva.

Es 2D, plano, dibujado con pyglet.shapes (sin shaders) para dejar a la vista
solo la geometria de la curva. La matematica de cada familia esta en este mismo
archivo porque es justamente el concepto que el ejemplo ensena.

Controles:
- arrastra:  mueve el punto de control bajo el cursor.
- 1 / 2 / 3: Bezier / Catmull-Rom / Hermite.
- , / .:     mueve el parametro t (construccion de De Casteljau en Bezier).
- espacio:   anima / detiene el barrido del parametro t.
- H:         muestra / oculta envoltura convexa y poligono de control.
- R:         reinicia los puntos de control.
- ESC:       cierra la ventana.
"""

import click
import numpy as np
import pyglet

from grafica.ui import InfoPanel

COLOR_POLIGONO = (90, 100, 125)      # poligono de control (lineas entre puntos).
COLOR_CURVA = (90, 200, 235)         # la curva resultante.
COLOR_PUNTO = (235, 210, 120)        # puntos de control.
COLOR_PUNTO_ACTIVO = (235, 130, 90)  # punto bajo el cursor.
COLOR_TANGENTE = (110, 220, 130)     # vectores tangentes.
COLOR_HULL = (120, 90, 160)          # envoltura convexa (solo Bezier).
COLOR_CASTELJAU = (235, 160, 60)     # construccion de De Casteljau.

MODO_BEZIER, MODO_CATMULL, MODO_HERMITE = 0, 1, 2
NOMBRE_MODO = {MODO_BEZIER: "Bezier", MODO_CATMULL: "Catmull-Rom", MODO_HERMITE: "Hermite"}

# Matriz de Catmull-Rom: p(t) = 0.5 * [t^3 t^2 t 1] @ M_CR @ [P0;P1;P2;P3].
M_CATMULL = 0.5 * np.array([
    [-1.0,  3.0, -3.0,  1.0],
    [ 2.0, -5.0,  4.0, -1.0],
    [-1.0,  0.0,  1.0,  0.0],
    [ 0.0,  2.0,  0.0,  0.0],
])

# Matriz de Hermite: p(t) = [t^3 t^2 t 1] @ M_H @ [P0;P1;t0;t1].
M_HERMITE = np.array([
    [ 2.0, -2.0,  1.0,  1.0],
    [-3.0,  3.0, -2.0, -1.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 1.0,  0.0,  0.0,  0.0],
])

N_MUESTRAS = 80  # segmentos con que se dibuja la curva.


def _puntos_iniciales(width, height):
    """Cuatro puntos de control repartidos en el ancho de la ventana."""
    margen_x, centro_y = width * 0.18, height * 0.5
    ancho = width - 2 * margen_x
    xs = margen_x + ancho * np.array([0.0, 0.33, 0.66, 1.0])
    ys = centro_y + np.array([-0.18, 0.22, -0.22, 0.18]) * height
    return np.stack([xs, ys], axis=1)


def _bezier_de_casteljau(puntos, t):
    """Punto de la curva de Bezier en t, junto con las etapas intermedias.

    Devuelve (punto_final, etapas), donde etapas es la lista de niveles de
    interpolacion: [nivel1 (3 puntos), nivel2 (2 puntos)]. El nivel0 son los
    puntos de control originales y el resultado final es el unico punto del
    ultimo nivel.
    """
    nivel = np.array(puntos, dtype=np.float64)
    etapas = []
    while len(nivel) > 1:
        nivel = (1.0 - t) * nivel[:-1] + t * nivel[1:]
        etapas.append(nivel)
    return nivel[0], etapas


def _curva_bezier(puntos):
    ts = np.linspace(0.0, 1.0, N_MUESTRAS)
    return np.array([_bezier_de_casteljau(puntos, t)[0] for t in ts])


def _evaluar_matriz(matriz, geometria, t):
    """Evalua p(t) = [t^3 t^2 t 1] @ matriz @ geometria para una cubica."""
    base = np.array([t**3, t**2, t, 1.0])
    return base @ matriz @ geometria


def _curva_catmull(puntos):
    ts = np.linspace(0.0, 1.0, N_MUESTRAS)
    return np.array([_evaluar_matriz(M_CATMULL, puntos, t) for t in ts])


def _tangentes_hermite(puntos):
    """Extremos y tangentes de la interpretacion Hermite de los cuatro puntos."""
    p0, p3 = puntos[0], puntos[3]
    t0 = puntos[1] - puntos[0]
    t1 = puntos[3] - puntos[2]
    return p0, p3, t0, t1


def _curva_hermite(puntos):
    p0, p3, t0, t1 = _tangentes_hermite(puntos)
    geometria = np.stack([p0, p3, t0, t1])
    ts = np.linspace(0.0, 1.0, N_MUESTRAS)
    return np.array([_evaluar_matriz(M_HERMITE, geometria, t) for t in ts])


def _envoltura_convexa(puntos):
    """Envoltura convexa 2D por el algoritmo de la cadena monotona de Andrew."""
    orden = sorted(map(tuple, puntos))
    def media(secuencia):
        cadena = []
        for p in secuencia:
            while len(cadena) >= 2:
                (ax, ay), (bx, by) = cadena[-2], cadena[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                cadena.pop()
            cadena.append(p)
        return cadena[:-1]
    return np.array(media(orden) + media(reversed(orden)))


@click.command("curvas_parametricas", short_help="Curvas cubicas: Hermite, Catmull-Rom, Bezier")
@click.option("--width", type=int, default=1000)
@click.option("--height", type=int, default=720)
def curvas_parametricas(width, height):
    window = pyglet.window.Window(width, height, caption="curvas parametricas cubicas")

    state = {
        "puntos": _puntos_iniciales(width, height),
        "modo": MODO_BEZIER,
        "t": 0.4,
        "animar": False,
        "ayudas": True,
        "arrastrando": None,
    }

    panel = (
        InfoPanel(x=20, y_top=height - 25, color=(220, 220, 220, 255),
                  background=(25, 28, 38), background_width=520)
        .add("modo", size=14)
        .add("regla")
        .add("paso")
        .add("parametro")
        .footer("arrastra puntos   1/2/3 modo   , . mueve t   espacio anima   H ayudas   R reset")
    )

    def curva_actual():
        puntos = state["puntos"]
        if state["modo"] == MODO_BEZIER:
            return _curva_bezier(puntos)
        if state["modo"] == MODO_CATMULL:
            return _curva_catmull(puntos)
        return _curva_hermite(puntos)

    def apply_state():
        """Vuelca el modo y sus reglas al panel. Unico lugar que describe la curva."""
        modo = state["modo"]
        panel["modo"] = f"modo: {NOMBRE_MODO[modo]}   (cubica con 4 puntos de control)"
        if modo == MODO_BEZIER:
            panel["regla"] = "la curva pasa por P0 y P3, contenida en la envoltura convexa"
            panel["paso"] = "evaluacion: De Casteljau (interpolaciones lineales sucesivas)"
            panel["parametro"] = f"t = {state['t']:.2f}   (, . para mover, espacio para animar)"
        elif modo == MODO_CATMULL:
            panel["regla"] = "interpola P1 y P2; tangentes t1=(P2-P0)/2, t2=(P3-P1)/2"
            panel["paso"] = "P0 y P3 solo aportan las tangentes en los extremos del segmento"
            panel["parametro"] = "C1 automatica: encadenando segmentos las tangentes calzan"
        else:
            panel["regla"] = "extremos P0 y P3; tangentes desde los handles P1 y P2"
            panel["paso"] = "t0 = P1 - P0,  t1 = P3 - P2 (la magnitud del handle curva mas)"
            panel["parametro"] = "con t0=3(P1-P0), t1=3(P3-P2) se recupera la curva de Bezier"
        print(f"[curvas_parametricas] modo={NOMBRE_MODO[modo]} t={state['t']:.2f}")

    def _linea(a, b, color, grosor, batch, refs):
        refs.append(pyglet.shapes.Line(a[0], a[1], b[0], b[1], thickness=grosor, color=color, batch=batch))

    def _polilinea(puntos, color, grosor, batch, refs):
        for a, b in zip(puntos[:-1], puntos[1:]):
            _linea(a, b, color, grosor, batch, refs)

    def _flecha(origen, vector, color, batch, refs):
        destino = origen + vector
        _linea(origen, destino, color, 2.0, batch, refs)
        largo = float(np.linalg.norm(vector))
        if largo < 1e-6:
            return
        direccion = vector / largo
        izquierda = np.array([-direccion[1], direccion[0]])
        base = destino - direccion * 14
        for signo in (1, -1):
            punta = base + izquierda * signo * 7
            _linea(destino, punta, color, 2.0, batch, refs)

    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        refs = []
        puntos = state["puntos"]
        modo = state["modo"]

        # envoltura convexa (solo Bezier, donde tiene la propiedad de contencion).
        if state["ayudas"] and modo == MODO_BEZIER:
            hull = _envoltura_convexa(puntos)
            _polilinea(np.vstack([hull, hull[0]]), COLOR_HULL, 1.0, batch, refs)

        # poligono de control: conecta los cuatro puntos en orden.
        if state["ayudas"]:
            _polilinea(puntos, COLOR_POLIGONO, 1.5, batch, refs)

        # la curva.
        _polilinea(curva_actual(), COLOR_CURVA, 3.0, batch, refs)

        # tangentes segun el modo.
        if modo == MODO_CATMULL:
            t1 = (puntos[2] - puntos[0]) / 2.0
            t2 = (puntos[3] - puntos[1]) / 2.0
            _flecha(puntos[1], t1, COLOR_TANGENTE, batch, refs)
            _flecha(puntos[2], t2, COLOR_TANGENTE, batch, refs)
        elif modo == MODO_HERMITE:
            p0, p3, tan0, tan1 = _tangentes_hermite(puntos)
            _flecha(p0, tan0, COLOR_TANGENTE, batch, refs)
            _flecha(p3, tan1, COLOR_TANGENTE, batch, refs)

        # construccion de De Casteljau en Bezier: niveles de interpolacion y punto en t.
        if modo == MODO_BEZIER:
            punto_t, etapas = _bezier_de_casteljau(puntos, state["t"])
            for nivel in etapas:
                _polilinea(nivel, COLOR_CASTELJAU, 1.5, batch, refs)
                for p in nivel:
                    refs.append(pyglet.shapes.Circle(p[0], p[1], 4, color=COLOR_CASTELJAU, batch=batch))
            refs.append(pyglet.shapes.Circle(punto_t[0], punto_t[1], 7, color=COLOR_CURVA, batch=batch))

        # puntos de control.
        for i, p in enumerate(puntos):
            color = COLOR_PUNTO_ACTIVO if state["arrastrando"] == i else COLOR_PUNTO
            refs.append(pyglet.shapes.Circle(p[0], p[1], 8, color=color, batch=batch))

        batch.draw()
        panel.draw()

    def _punto_cercano(x, y):
        cursor = np.array([x, y], dtype=np.float64)
        distancias = np.linalg.norm(state["puntos"] - cursor, axis=1)
        indice = int(np.argmin(distancias))
        return indice if distancias[indice] < 18 else None

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        state["arrastrando"] = _punto_cercano(x, y)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if state["arrastrando"] is not None:
            state["puntos"][state["arrastrando"]] = (x, y)

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        state["arrastrando"] = None

    def _avanzar_t(dt):
        if state["animar"]:
            state["t"] = (state["t"] + dt * 0.4) % 1.0

    pyglet.clock.schedule_interval(_avanzar_t, 1 / 60)

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol in (key._1, key._2, key._3):
            state["modo"] = {key._1: MODO_BEZIER, key._2: MODO_CATMULL, key._3: MODO_HERMITE}[symbol]
        elif symbol == key.COMMA:
            state["t"] = max(0.0, state["t"] - 0.02)
        elif symbol == key.PERIOD:
            state["t"] = min(1.0, state["t"] + 0.02)
        elif symbol == key.SPACE:
            state["animar"] = not state["animar"]
        elif symbol == key.H:
            state["ayudas"] = not state["ayudas"]
        elif symbol == key.R:
            state["puntos"] = _puntos_iniciales(width, height)
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    curvas_parametricas()
