import time
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline


# Un L-system (sistema de Lindenmayer) define una planta con tres cosas:
# un axioma (la cadena inicial), un conjunto de reglas de reescritura y un
# angulo. En cada iteracion todos los simbolos se reemplazan a la vez segun
# las reglas; los simbolos sin regla se copian tal cual.
#
# El alfabeto que interpreta la tortuga:
#   F   avanzar dibujando un segmento de rama
#   A   yema: es el simbolo que las reglas reescriben para seguir creciendo
#   +/- girar a la izquierda / derecha  (rotacion en torno a Z local)
#   &/^ inclinar hacia abajo / arriba   (rotacion en torno a X local)
#   //\ rolar en un sentido / el otro   (rotacion en torno a Y local)
#   [   guardar el estado de la tortuga en la pila
#   ]   restaurar el ultimo estado guardado
#
# Los corchetes son la razon por la que un L-system genera arboles y no solo
# curvas: al restaurar el estado la tortuga vuelve al punto de bifurcacion y
# sale hacia otro lado. Son un push y un pop de la misma pila de matrices que
# usa un grafo de escena.
#
# Las hojas no tienen simbolo propio: se dibujan donde quedo una yema sin
# reescribir, o sea en las puntas. Si en cambio se pusiera un simbolo de hoja
# dentro de la regla, las hojas creadas en la primera iteracion sobrevivirian
# hasta el final y quedarian colgando del tronco, con el tamano de la rama de
# ese momento. Ese es el motivo de fondo para separar la yema de la rama: la
# yema es lo que sigue creciendo, y solo deja de crecer cuando se acaban las
# iteraciones.

ESPECIES = {
    # Tres ramas por nodo separadas 120 grados: cuatro simbolos de roll a 30
    # grados dan la vuelta completa entre una rama y la siguiente.
    "arbol": {
        "nombre": "arbol de tres ramas",
        "axioma": "A",
        "reglas": {"A": "F[&A]////[&A]////[&A]"},
        "angulo": 30.0,
        "iteraciones": 5,
        "decaimiento": 0.78,
    },
    # Solo usa + y -, que rotan en torno al mismo eje, asi que el arbol queda
    # contenido en un plano. Es el caso mas simple para leer que hacen los
    # corchetes, y deja ver que la tercera dimension la traen los simbolos de
    # roll y de inclinacion, no la tortuga por si sola.
    "binario": {
        "nombre": "binario (plano)",
        "axioma": "A",
        "reglas": {"A": "F[+A][-A]"},
        "angulo": 28.0,
        "iteraciones": 7,
        "decaimiento": 0.72,
    },
    # Dos reglas en vez de una: A es la guia central, que en cada iteracion sube
    # un segmento y deja un piso de tres ramas laterales; B es una lateral, que
    # se ramifica aparte. De ahi salen los pisos escalonados de un pino, y de
    # paso muestra que un L-system admite varias reglas a la vez.
    "monopodial": {
        "nombre": "monopodial (pino)",
        "axioma": "A",
        "reglas": {"A": "F[&B]////[&B]////[&B]FA", "B": "F[-B][+B]"},
        "angulo": 40.0,
        "iteraciones": 6,
        "decaimiento": 0.86,
    },
    "arbusto": {
        "nombre": "arbusto",
        "axioma": "A",
        "reglas": {"A": "FF+[+A-A-A]-[-A+A+A]"},
        "angulo": 22.5,
        "iteraciones": 5,
        "decaimiento": 0.78,
    },
}

ORDEN = ["arbol", "binario", "monopodial", "arbusto"]

LADOS = 7          # caras del tronco de cono con que se dibuja cada rama
LARGO_BASE = 0.55  # largo del primer segmento, en unidades de mundo
RADIO_BASE = 0.045

COLOR_TRONCO = np.array([0.42, 0.28, 0.18])
COLOR_RAMA = np.array([0.48, 0.38, 0.22])
COLOR_HOJA = np.array([0.28, 0.62, 0.24])

# El largo de la cadena crece de forma exponencial con las iteraciones, y la
# base de esa exponencial es la cantidad de F que tenga la regla: el arbusto
# multiplica sus ramas por 8 en cada paso, asi que a la septima ya pasa los dos
# millones. Por eso el tope se pone en ramas y no en iteraciones: asi vale para
# cualquier regla, incluida una que se agregue despues.
MAX_RAMAS = 40000

FOVY = 45.0


def reescribir(axioma, reglas, iteraciones):
    """Aplica las reglas de reescritura en paralelo, una vez por iteracion.

    Todos los simbolos se reemplazan simultaneamente mirando la cadena de la
    iteracion anterior. Esa simultaneidad es lo que distingue a un L-system de
    una gramatica formal comun, donde se reescribe un simbolo a la vez.
    """
    cadena = axioma
    for _ in range(iteraciones):
        cadena = "".join(reglas.get(simbolo, simbolo) for simbolo in cadena)
    return cadena


def _anillo(lados):
    """Puntos de un circulo unitario en el plano XZ (el eje del tronco es Y)."""
    angulos = np.linspace(0.0, 2.0 * np.pi, lados, endpoint=False)
    return np.stack([np.cos(angulos), np.zeros(lados), np.sin(angulos)], axis=1)


def _patron_manto(lados):
    """Indices de los dos triangulos de cada cara del manto, para un segmento.

    Se calcula una vez y despues cada rama solo le suma el indice de su primer
    vertice. Es lo mismo que armarlos con un ciclo por rama, pero sin pagar ese
    ciclo miles de veces.
    """
    i = np.arange(lados)
    j = (i + 1) % lados
    abajo_i, abajo_j = i, j
    arriba_i, arriba_j = lados + i, lados + j
    return np.stack(
        [abajo_i, arriba_i, arriba_j, abajo_i, arriba_j, abajo_j], axis=1
    ).ravel()


PATRON_HOJA = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)


class _Malla:
    """Acumula los trozos de geometria que va dejando la tortuga.

    Lleva la cuenta corriente de vertices para saber en que indice empieza cada
    trozo nuevo. Sin ese contador habria que recorrer lo acumulado en cada
    rama, y el costo de construir el arbol crecería con el cuadrado del numero
    de ramas.
    """

    def __init__(self):
        self.posiciones = []
        self.normales = []
        self.colores = []
        self.indices = []
        self.vertices = 0

    def agregar(self, posiciones, normales, color, patron):
        self.posiciones.append(posiciones)
        self.normales.append(normales)
        self.colores.append(np.tile(color, (len(posiciones), 1)))
        self.indices.append(patron + self.vertices)
        self.vertices += len(posiciones)

    def compilar(self):
        return (
            np.concatenate(self.posiciones).astype(np.float32),
            np.concatenate(self.normales).astype(np.float32),
            np.concatenate(self.colores).astype(np.float32),
            np.concatenate(self.indices).astype(np.uint32),
        )


def _emitir_rama(malla, matriz, largo, radio_base, radio_punta, anillo, patron, color):
    """Agrega un tronco de cono que va del origen local a (0, largo, 0).

    La tortuga avanza en su eje +Y, asi que la rama se construye ahi y se lleva
    al espacio del arbol con la matriz actual. Como la tortuga solo acumula
    rotaciones y traslaciones, su parte 3x3 es ortonormal y sirve tal cual para
    las normales, sin necesidad de la inversa traspuesta.
    """
    abajo = anillo * radio_base
    arriba = anillo * radio_punta + np.array([0.0, largo, 0.0])
    locales = np.concatenate([abajo, arriba])

    homogeneas = np.concatenate([locales, np.ones((len(locales), 1))], axis=1)
    mundo = (matriz @ homogeneas.T).T[:, :3]

    # la normal de un punto del manto apunta hacia afuera del eje: es el mismo
    # vector del anillo, rotado por la orientacion actual de la tortuga.
    normales_locales = np.concatenate([anillo, anillo])
    normales_mundo = (matriz[:3, :3] @ normales_locales.T).T

    malla.agregar(mundo, normales_mundo, color, patron)


def _emitir_hoja(malla, matriz, tamano):
    """Agrega una hoja: un cuadrilatero chico en el plano XY local de la tortuga."""
    ancho = tamano * 0.45
    locales = np.array([
        [-ancho, 0.0, 0.0],
        [ancho, 0.0, 0.0],
        [ancho, tamano, 0.0],
        [-ancho, tamano, 0.0],
    ])
    homogeneas = np.concatenate([locales, np.ones((4, 1))], axis=1)
    mundo = (matriz @ homogeneas.T).T[:, :3]

    normal = matriz[:3, :3] @ np.array([0.0, 0.0, 1.0])
    malla.agregar(mundo, np.tile(normal, (4, 1)), COLOR_HOJA, PATRON_HOJA)


def construir_geometria(cadena, yemas, angulo_grados, decaimiento, desorden, semilla):
    """Recorre la cadena con la tortuga y devuelve la malla del arbol.

    La tortuga es una matriz 4x4 (donde esta y hacia donde mira) mas el largo y
    el grosor del proximo segmento. La pila guarda esos tres datos: si solo se
    guardara la matriz, al volver de una rama el tronco seguiria adelgazandose
    como si nunca hubiera subido por ella.

    `yemas` son los simbolos que tienen regla. Los que quedan en la cadena final
    son las yemas que no alcanzaron a crecer, o sea las puntas, y ahi van las
    hojas. Definirlo asi y no con un simbolo fijo hace que una especie nueva
    tenga follaje sin tocar esta funcion, incluso si sus reglas usan varios
    simbolos de crecimiento.

    `desorden` perturba cada angulo al azar. Con 0 el arbol es exactamente
    autosimilar, que es lo que delata a una planta generada por computador; al
    subirlo las ramas dejan de repetirse identicas entre si.
    """
    rng = np.random.default_rng(semilla)
    anillo = _anillo(LADOS)
    patron = _patron_manto(LADOS)
    malla = _Malla()

    matriz = tr.identity()
    largo = LARGO_BASE
    radio = RADIO_BASE
    pila = []

    def giro(constructor, signo):
        grados = angulo_grados
        if desorden > 0.0:
            grados += rng.uniform(-desorden, desorden) * angulo_grados
        return constructor(np.radians(signo * grados))

    for simbolo in cadena:
        if simbolo == "F":
            radio_punta = radio * decaimiento
            # el color va del cafe del tronco al verde segun cuanto adelgazo la
            # rama: las puntas delgadas son las mas nuevas.
            mezcla = min(1.0, 1.0 - radio / RADIO_BASE)
            color = COLOR_TRONCO * (1.0 - mezcla) + COLOR_RAMA * mezcla
            _emitir_rama(malla, matriz, largo, radio, radio_punta, anillo, patron, color)
            matriz = matriz @ tr.translate(0.0, largo, 0.0)
            largo *= decaimiento
            radio = radio_punta
        elif simbolo in yemas:
            # el tamano de la hoja se ata al grosor de la ramita y no a su
            # largo: entre especies el grosor decae de forma parecida, mientras
            # que el largo depende del decaimiento de cada regla y dejaba hojas
            # desproporcionadas en las especies de ramas largas.
            _emitir_hoja(malla, matriz, radio * 10.0)
        elif simbolo == "+":
            matriz = matriz @ giro(tr.rotationZ, 1)
        elif simbolo == "-":
            matriz = matriz @ giro(tr.rotationZ, -1)
        elif simbolo == "&":
            matriz = matriz @ giro(tr.rotationX, 1)
        elif simbolo == "^":
            matriz = matriz @ giro(tr.rotationX, -1)
        elif simbolo == "/":
            matriz = matriz @ giro(tr.rotationY, 1)
        elif simbolo == "\\":
            matriz = matriz @ giro(tr.rotationY, -1)
        elif simbolo == "[":
            pila.append((matriz.copy(), largo, radio))
        elif simbolo == "]":
            matriz, largo, radio = pila.pop()

    return malla.compilar()


@click.command("lsystem", short_help="Arbolitos generados con L-systems")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
@click.option("--especie", type=click.Choice(ORDEN), default="arbol")
def lsystem(width, height, especie):
    window = pyglet.window.Window(width, height)

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    estado = {
        "especie": ORDEN.index(especie),
        "iteraciones": ESPECIES[especie]["iteraciones"],
        "desorden": 0.0,
        "semilla": 7,
        "malla": None,
        "triangulos": 0,
        "segmentos": 0,
        "ms": 0.0,
    }

    # la camara orbita en torno al centro del arbol; regenerar() ajusta el
    # centro y la distancia segun el tamano que resulte.
    camara = {
        "angulo": 0.6,
        "elevacion": 0.15,
        "distancia": 5.5,
        "centro": np.zeros(3),
    }

    panel = (
        InfoPanel(x=14, y_top=height - 24, background=(16, 22, 16), background_width=470)
        .add("especie")
        .add("regla")
        .add("iteraciones")
        .add("cadena")
        .add("desorden")
        .add("malla")
        .footer("1-4 especie   -/= iteraciones   d desorden   r semilla   flechas camara")
    )

    def especie_actual():
        return ESPECIES[ORDEN[estado["especie"]]]

    def regenerar():
        """Reescribe la cadena y sube la malla nueva. Falso si excede el tope."""
        receta = especie_actual()
        inicio = time.perf_counter()
        cadena = reescribir(receta["axioma"], receta["reglas"], estado["iteraciones"])

        ramas = cadena.count("F")
        if ramas > MAX_RAMAS:
            print(
                f"[lsystem] {ramas} ramas exceden el tope de {MAX_RAMAS}; "
                f"se mantiene el arbol anterior"
            )
            return False

        posiciones, normales, colores, indices = construir_geometria(
            cadena,
            set(receta["reglas"]),
            receta["angulo"],
            receta["decaimiento"],
            estado["desorden"],
            estado["semilla"],
        )
        estado["ms"] = (time.perf_counter() - inicio) * 1000.0

        if estado["malla"] is not None:
            estado["malla"].delete()
        malla = pipeline.vertex_list_indexed(
            len(posiciones), GL.GL_TRIANGLES, indices.tolist()
        )
        malla.position[:] = posiciones.flatten()
        malla.normal[:] = normales.flatten()
        malla.color[:] = colores.flatten()

        estado["malla"] = malla
        estado["triangulos"] = len(indices) // 3
        estado["segmentos"] = cadena.count("F")

        # cada especie crece a su tamano, y ademas el arbol se alarga con cada
        # iteracion, asi que la camara se reencuadra sola: mira al centro de la
        # caja que contiene al arbol y se aleja lo justo para que quepa entero.
        minimo, maximo = posiciones.min(axis=0), posiciones.max(axis=0)
        camara["centro"] = (minimo + maximo) / 2.0
        radio = float(np.linalg.norm(maximo - minimo)) / 2.0
        camara["distancia"] = radio / np.tan(np.radians(FOVY / 2.0)) * 1.15

        regla = next(iter(receta["reglas"].items()))
        panel["especie"] = f"especie      {receta['nombre']}"
        panel["regla"] = f"regla        {regla[0]} -> {regla[1]}"
        panel["iteraciones"] = (
            f"iteraciones  {estado['iteraciones']}   angulo {receta['angulo']:.1f} grados"
        )
        panel["cadena"] = (
            f"cadena       {len(cadena)} simbolos, {estado['segmentos']} ramas"
        )
        panel["desorden"] = f"desorden     {estado['desorden']:.2f}"
        panel["malla"] = (
            f"malla        {estado['triangulos']} triangulos en {estado['ms']:.0f} ms"
        )
        print(
            f"[lsystem] {receta['nombre']} n={estado['iteraciones']} "
            f"cadena={len(cadena)} ramas={estado['segmentos']} "
            f"triangulos={estado['triangulos']} ({estado['ms']:.0f} ms)"
        )
        return True

    regenerar()

    projection = tr.perspective(FOVY, width / height, 0.1, 120.0)

    @window.event
    def on_draw():
        GL.glClearColor(0.62, 0.74, 0.82, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        centro = camara["centro"]
        radio_horizontal = camara["distancia"] * np.cos(camara["elevacion"])
        ojo = centro + np.array([
            radio_horizontal * np.cos(camara["angulo"]),
            camara["distancia"] * np.sin(camara["elevacion"]),
            radio_horizontal * np.sin(camara["angulo"]),
        ])
        view = tr.lookAt(
            ojo.astype(np.float32),
            centro.astype(np.float32),
            np.array([0.0, 1.0, 0.0]),
        )

        pipeline.use()
        pipeline["view"] = view.reshape(16, 1, order="F")
        pipeline["projection"] = projection.reshape(16, 1, order="F")
        pipeline["light_direction"] = (-0.4, 1.0, 0.5)
        pipeline["ambient_strength"] = 0.35
        estado["malla"].draw(GL.GL_TRIANGLES)

        with ui_overlay():
            panel.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        teclas = pyglet.window.key
        numeros = {teclas._1: 0, teclas._2: 1, teclas._3: 2, teclas._4: 3}
        if symbol in numeros:
            estado["especie"] = numeros[symbol]
            estado["iteraciones"] = ESPECIES[ORDEN[numeros[symbol]]]["iteraciones"]
            regenerar()
        elif symbol == teclas.EQUAL:
            estado["iteraciones"] += 1
            if not regenerar():
                estado["iteraciones"] -= 1
        elif symbol == teclas.MINUS:
            if estado["iteraciones"] > 1:
                estado["iteraciones"] -= 1
                regenerar()
        elif symbol == teclas.D:
            estado["desorden"] = 0.0 if estado["desorden"] > 0.0 else 0.35
            regenerar()
        elif symbol == teclas.R:
            estado["semilla"] += 1
            regenerar()

    @window.event
    def on_text_motion(motion):
        teclas = pyglet.window.key
        if motion == teclas.MOTION_LEFT:
            camara["angulo"] -= 0.12
        elif motion == teclas.MOTION_RIGHT:
            camara["angulo"] += 0.12
        elif motion == teclas.MOTION_UP:
            camara["elevacion"] = min(1.4, camara["elevacion"] + 0.1)
        elif motion == teclas.MOTION_DOWN:
            camara["elevacion"] = max(-1.4, camara["elevacion"] - 0.1)

    pyglet.app.run()
