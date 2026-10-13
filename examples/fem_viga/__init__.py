"""Elementos finitos frente a masa-resorte: una viga de goma en voladizo.

Dos vigas iguales cuelgan de una pared bajo su propio peso. Comparten la malla
de triangulos, la masa de cada vertice y el integrador (Euler semi-implicito):
la de arriba calcula sus fuerzas con resortes en las aristas y la de abajo con
elementos finitos. Cada triangulo se colorea segun la energia por unidad de
area que acumula su deformacion.

La rigidez k de los resortes esta elegida para que las dos vigas se doblen lo
mismo con la malla inicial. Al cambiar la cantidad de filas, la viga de
elementos finitos se acerca a la flecha teorica de una viga de modulo E, y la
de resortes cambia de rigidez, porque k depende de la forma de los triangulos.
Con triangulos alargados, ademas, el paso maximo de los dos modelos se achica.
Si se estira el extremo y se suelta, la viga de resortes puede quedar doblada:
un triangulo dado vuelta tiene los mismos largos que en reposo, asi que sus
resortes no lo devuelven.

Controles:
- , / .:    baja / sube el modulo de Young E (elementos finitos).
- - / =:    baja / sube el coeficiente de Poisson nu (elementos finitos).
- F:        cambia la cantidad de filas de la malla (2, 4 u 8).
- z / x:    baja / sube las subdivisiones temporales (substeps).
- mouse:    arrastra un vertice; el mismo vertice de la otra viga se mueve igual.
- R:        reinicia las vigas en reposo.
- ESC:      cierra la ventana.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
from pyglet import gl

from grafica.ui import InfoPanel
from grafica.utils import load_pipeline

from .viga import crear_malla, fuerzas_fem, fuerzas_resortes, nuevo_estado, paso, paso_maximo

# una viga de goma blanda de 2,4 m x 0,4 m, en unidades del sistema internacional.
LARGO = 2.4
ALTO = 0.4
DENSIDAD = 1100.0
GRAVEDAD = 9.8
AMORTIGUACION = 2.0
FILAS = (2, 4, 8)

DEFAULTS = {
    "young": 8.0e6,
    "poisson": 0.45,
    "rigidez": 4.4e6,  # k que iguala la flecha de las dos vigas con 4 filas.
    "filas_index": 1,
    "substeps": 26,
}

ESCALA = 250.0  # pixeles por metro.
MODELOS = ("resortes", "fem")
NOMBRES = {"resortes": "masa-resorte", "fem": "elementos finitos"}

# mismo colormap que difusion_calor: poca energia oscuro, mucha energia claro.
PALETA = np.array(
    [
        [0.0, 0.0, 0.05],
        [0.35, 0.05, 0.45],
        [0.85, 0.25, 0.20],
        [0.98, 0.75, 0.20],
        [1.0, 1.0, 0.90],
    ]
)
COLOR_ARISTAS = (0.45, 0.45, 0.55)
BLANCO = (230, 230, 230, 255)
ROJO = (255, 110, 110, 255)


def _colores(densidad_energia, maximo):
    """Colormap de la energia por area; la raiz la deja proporcional a la tension."""
    nivel = np.sqrt(densidad_energia / maximo)
    paradas = np.linspace(0.0, 1.0, len(PALETA))
    return np.stack([np.interp(nivel, paradas, PALETA[:, canal]) for canal in range(3)], axis=1)


@click.command("fem_viga", short_help="Viga en voladizo: FEM frente a masa-resorte")
@click.option("--columnas", type=int, default=24, help="celdas de la malla a lo largo de la viga")
def fem_viga(columnas):
    width, height = 900, 760
    window = pyglet.window.Window(width, height, caption="elementos finitos frente a masa-resorte")
    gl.glClearColor(0.13, 0.13, 0.16, 1.0)

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / ".." / "hello_world" / "fragment_program.glsl",
    )
    pipeline.use()
    pipeline["resolution"] = (float(width), float(height))

    # esquina superior del borde empotrado de cada viga, en pixeles.
    origenes = {"resortes": np.array([110.0, 570.0]), "fem": np.array([110.0, 300.0])}

    state = dict(DEFAULTS)
    state["agarrado"] = None
    fuerzas = {
        "resortes": lambda posiciones: fuerzas_resortes(
            state["malla"], posiciones, state["rigidez"]
        ),
        "fem": lambda posiciones: fuerzas_fem(
            state["malla"], posiciones, state["young"], state["poisson"]
        ),
    }

    panel = (
        InfoPanel(x=20, y_top=height - 28)
        .add("resortes")
        .add("fem")
        .add("substeps")
        .add("malla")
        .footer(", . E   - = nu   F filas   z x substeps   mouse arrastra   R reset")
    )

    # la pared y el nombre de cada viga; la lista vive en `state` para que el
    # recolector de basura no borre las figuras.
    escena = pyglet.graphics.Batch()
    state["decorado"] = []
    for modelo, (x, y) in origenes.items():
        alto_pared = ALTO * ESCALA + 60
        state["decorado"].append(
            pyglet.shapes.Rectangle(
                x - 24, y + 30 - alto_pared, 24, alto_pared, color=(110, 110, 120), batch=escena
            )
        )
        state["decorado"].append(
            pyglet.text.Label(
                NOMBRES[modelo], font_name="Fira Code", font_size=12,
                x=x, y=y + 14, color=BLANCO, batch=escena,
            )
        )
    flecha_teorica = pyglet.shapes.Line(0, 0, 0, 0, thickness=2, color=(207, 56, 137), batch=escena)
    texto_flecha = pyglet.text.Label(
        "flecha teorica", font_name="Fira Code", font_size=10,
        color=(207, 56, 137, 255), batch=escena,
    )

    state["paso_maximo"] = {}
    state["gpu"] = {}

    def recalcular_paso_maximo(modelos):
        # dt < 2 / omega_max: depende de la rigidez y de la forma de los triangulos.
        for modelo in modelos:
            state["paso_maximo"][modelo] = paso_maximo(fuerzas[modelo], state["malla"])

    def reconstruir():
        """Crea la malla con la cantidad de filas elegida y las vigas en reposo."""
        state["malla"] = crear_malla(
            LARGO, ALTO, columnas, FILAS[state["filas_index"]], DENSIDAD
        )
        state["estados"] = {modelo: nuevo_estado(state["malla"]) for modelo in MODELOS}

        # cada viga tiene sus triangulos (un color por triangulo, asi que no
        # comparten vertices) y sus aristas.
        cantidad_triangulos = len(state["malla"]["triangulos"])
        cantidad_aristas = len(state["malla"]["aristas"])
        for listas in state["gpu"].values():
            for lista in listas.values():
                lista.delete()
        for modelo in MODELOS:
            state["gpu"][modelo] = {
                "triangulos": pipeline.vertex_list(
                    3 * cantidad_triangulos, gl.GL_TRIANGLES, position="f", color="f"
                ),
                "aristas": pipeline.vertex_list(
                    2 * cantidad_aristas, gl.GL_LINES, position="f", color="f"
                ),
            }
            state["gpu"][modelo]["aristas"].color[:] = np.tile(COLOR_ARISTAS, 2 * cantidad_aristas)
        recalcular_paso_maximo(MODELOS)

    def apply_state():
        """Traduce el estado actual a las etiquetas y al terminal.

        Es el unico lugar que convierte el estado en texto: cada tecla muta
        `state` y llama aqui, asi las etiquetas nunca se desincronizan.
        """
        paso_actual = 1.0 / 60.0 / state["substeps"]
        filas = FILAS[state["filas_index"]]
        parametros = {
            "resortes": f"k = {state['rigidez'] / 1e6:.1f} MN/m",
            "fem": f"E = {state['young'] / 1e6:.1f} MPa, nu = {state['poisson']:.2f}",
        }
        for modelo in MODELOS:
            limite = state["paso_maximo"][modelo]
            if state["estados"][modelo]["exploto"]:
                panel[modelo] = f"{NOMBRES[modelo]}: INESTABLE (reinicia con R)"
            else:
                panel[modelo] = (
                    f"{NOMBRES[modelo]}: {parametros[modelo]}, "
                    f"paso maximo {limite * 1e3:.2f} ms"
                )
            panel.color(modelo, ROJO if paso_actual > limite else BLANCO)
        panel["substeps"] = f"substeps: {state['substeps']}, paso {paso_actual * 1e3:.2f} ms"
        panel["malla"] = (
            f"malla: {columnas} x {filas} celdas, "
            f"{len(state['malla']['triangulos'])} triangulos"
        )

        # flecha de una viga de Euler-Bernoulli bajo su peso: 3 rho g L^4 / (2 E H^2).
        flecha = 3.0 * DENSIDAD * GRAVEDAD * LARGO**4 / (2.0 * state["young"] * ALTO**2)
        x_punta = origenes["fem"][0] + LARGO * ESCALA
        y_flecha = origenes["fem"][1] - (ALTO / 2.0 + flecha) * ESCALA
        flecha_teorica.x, flecha_teorica.y = x_punta - 30, y_flecha
        flecha_teorica.x2, flecha_teorica.y2 = x_punta + 30, y_flecha
        texto_flecha.x, texto_flecha.y = x_punta + 38, y_flecha - 5

        print(
            f"[fem_viga] {parametros['resortes']} {parametros['fem']} filas={filas} "
            f"substeps={state['substeps']} paso={paso_actual * 1e3:.2f} ms "
            f"maximo resortes={state['paso_maximo']['resortes'] * 1e3:.2f} ms "
            f"fem={state['paso_maximo']['fem'] * 1e3:.2f} ms"
        )

    def avanzar_simulacion(_dt):
        # paso fijo de 1/60 s (no el dt real del reloj) para que la estabilidad
        # de la simulacion no dependa de la tasa de cuadros.
        fijos = () if state["agarrado"] is None else (state["agarrado"]["vertice"],)
        for modelo in MODELOS:
            exploto = state["estados"][modelo]["exploto"]
            paso(
                state["estados"][modelo], state["malla"], fuerzas[modelo],
                1.0 / 60.0, state["substeps"], GRAVEDAD, AMORTIGUACION, fijos,
            )
            if state["estados"][modelo]["exploto"] != exploto:
                apply_state()

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.COMMA:
            state["young"] /= 1.25
            recalcular_paso_maximo(["fem"])
        elif symbol == key.PERIOD:
            state["young"] *= 1.25
            recalcular_paso_maximo(["fem"])
        elif symbol == key.MINUS:
            state["poisson"] = round(max(0.0, state["poisson"] - 0.05), 2)
            recalcular_paso_maximo(["fem"])
        elif symbol == key.EQUAL:
            state["poisson"] = round(min(0.45, state["poisson"] + 0.05), 2)
            recalcular_paso_maximo(["fem"])
        elif symbol == key.F:
            state["filas_index"] = (state["filas_index"] + 1) % len(FILAS)
            reconstruir()
        elif symbol == key.Z:
            state["substeps"] = max(2, state["substeps"] - 2)
        elif symbol == key.X:
            state["substeps"] += 2
        elif symbol == key.R:
            state["estados"] = {modelo: nuevo_estado(state["malla"]) for modelo in MODELOS}
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        # el vertice libre mas cercano al cursor, en cualquiera de las dos vigas.
        libres = np.flatnonzero(state["malla"]["libres"])
        mejor = None
        for modelo in MODELOS:
            posiciones = state["estados"][modelo]["posiciones"][libres]
            distancias = np.linalg.norm(origenes[modelo] + ESCALA * posiciones - (x, y), axis=1)
            if mejor is None or distancias.min() < mejor[0]:
                mejor = (distancias.min(), libres[np.argmin(distancias)])
        if mejor[0] > 20.0:
            return
        vertice = mejor[1]
        state["agarrado"] = {
            "vertice": vertice,
            "mouse": np.array([x, y], dtype=float),
            "posiciones": {
                modelo: state["estados"][modelo]["posiciones"][vertice].copy()
                for modelo in MODELOS
            },
        }

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        agarrado = state["agarrado"]
        if agarrado is None:
            return
        # las dos vigas reciben el mismo desplazamiento desde donde se agarro.
        desplazamiento = (np.array([x, y]) - agarrado["mouse"]) / ESCALA
        for modelo in MODELOS:
            posiciones = state["estados"][modelo]["posiciones"]
            posiciones[agarrado["vertice"]] = agarrado["posiciones"][modelo] + desplazamiento

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        state["agarrado"] = None

    @window.event
    def on_draw():
        window.clear()
        malla = state["malla"]
        # una viga que exploto deja de dibujarse; su fila del panel lo avisa.
        visibles = [modelo for modelo in MODELOS if not state["estados"][modelo]["exploto"]]
        densidades = {
            modelo: state["estados"][modelo]["energia"] / malla["areas"] for modelo in visibles
        }
        # el triangulo mas claro es el de mayor energia entre las dos vigas.
        maximo = max([1e-9] + [densidad.max() for densidad in densidades.values()])
        for modelo in visibles:
            pantalla = origenes[modelo] + ESCALA * state["estados"][modelo]["posiciones"]
            pantalla = pantalla.astype(np.float32)
            colores = np.repeat(_colores(densidades[modelo], maximo), 3, axis=0)
            gpu = state["gpu"][modelo]
            gpu["triangulos"].position[:] = pantalla[malla["triangulos"]].ravel()
            gpu["triangulos"].color[:] = colores.ravel()
            gpu["aristas"].position[:] = pantalla[malla["aristas"]].ravel()

        pipeline.use()
        for modelo in visibles:
            state["gpu"][modelo]["triangulos"].draw(gl.GL_TRIANGLES)
            state["gpu"][modelo]["aristas"].draw(gl.GL_LINES)
        escena.draw()
        panel.draw()

    reconstruir()
    apply_state()
    pyglet.clock.schedule_interval(avanzar_simulacion, 1.0 / 60.0)
    pyglet.app.run()


if __name__ == "__main__":
    fem_viga()
