"""Ejemplo atomico de deformacion: una hebra masa-resorte colgante.

Aisla el sistema masa-resorte (enfoque Lagrangiano) y la eleccion del
integrador. La hebra es una cadena 1D de masas: el nodo superior esta
anclado y el resto cuelga bajo gravedad, unido por resortes.

El integrador se elige con el teclado. Con resortes rigidos (k alto) el
Euler explicito gana energia y explota; el semi-implicito y Verlet se
mantienen acotados con el mismo paso temporal. Subir las subdivisiones
temporales relaja la condicion de estabilidad y estabiliza al explicito.

Controles:
- M:        cambia el metodo de integracion.
- , / .:    baja / sube la rigidez k.
- - / =:    baja / sube la amortiguacion c.
- z / x:    baja / sube las subdivisiones temporales (substeps).
- espacio:  da un empujon lateral a la hebra.
- mouse:    arrastra una masa para perturbar la hebra.
- R:        reinicia la hebra en reposo.
- ESC:      cierra la ventana.
"""

from pathlib import Path

import click
import numpy as np
import pyglet

from .strand import METODOS, HebraMasaResorte

DEFAULTS = {
    "metodo_index": 1,  # arranca en semi-implicito (estable) para luego romperlo.
    "k": 300.0,
    "c": 2.0,
    "substeps": 8,
}


def _color_por_estiramiento(estiramiento):
    """Azul si el resorte esta comprimido, rojo si esta estirado, gris en reposo."""
    # nivel en [-1, 1]: -1 muy comprimido, 0 en reposo, 1 muy estirado.
    nivel = float(np.clip(estiramiento / 0.4, -1.0, 1.0))
    if nivel >= 0.0:
        return (int(120 + 135 * nivel), int(120 - 90 * nivel), int(120 - 90 * nivel))
    return (int(120 + 90 * nivel), int(120 + 60 * nivel), int(120 - 135 * nivel))


class Panel:
    """Panel de informacion en pantalla: etiquetas FiraCode arriba a la izquierda.

    Agrupa la creacion de las etiquetas para que no ensucie el cuerpo principal.
    apply_state() solo escribe en `panel.metodo`, `panel.k`, etc.; este objeto se
    encarga de la posicion, la fuente y el dibujado.
    """

    def __init__(self, height):
        self.batch = pyglet.graphics.Batch()

        def etiqueta(fila, color=(220, 220, 220, 255), size=12):
            return pyglet.text.Label(
                "", font_name="Fira Code", font_size=size,
                x=20, y=height - 25 - fila * 22, color=color, batch=self.batch,
            )

        self.metodo = etiqueta(0, size=14)
        self.k = etiqueta(1)
        self.c = etiqueta(2)
        self.substeps = etiqueta(3)
        self.energia = etiqueta(4)

        # linea fija con los controles, al pie de la ventana.
        self._controles = pyglet.text.Label(
            "M metodo   , . rigidez   - = damping   z x substeps   "
            "espacio empujon   mouse arrastra   R reset",
            font_name="Fira Code", font_size=10, x=20, y=14,
            color=(150, 150, 150, 255), batch=self.batch,
        )

    def draw(self):
        self.batch.draw()


@click.command("masa_resorte", short_help="Hebra masa-resorte: integradores y estabilidad")
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=700)
@click.option("--nodos", type=int, default=8, help="numero de masas de la hebra")
def masa_resorte(width, height, nodos):
    window = pyglet.window.Window(width, height, caption="masa-resorte: hebra colgante")

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    separacion = 40.0
    ancla = (width / 2.0, height - 120.0)
    hebra = HebraMasaResorte(nodos, ancla, separacion, masa=1.0, gravedad=400.0)

    state = dict(DEFAULTS)
    state["agarrado"] = None  # indice del nodo que sigue al mouse, o None.

    panel = Panel(height)

    def apply_state():
        """Traduce el estado actual a las etiquetas y al terminal.

        Es el unico lugar que convierte el estado en texto: cada tecla muta
        `state` y llama aqui, asi las etiquetas nunca se desincronizan.
        """
        metodo = METODOS[state["metodo_index"]]
        panel.metodo.text = f"metodo: {metodo}"
        panel.k.text = f"rigidez k: {state['k']:.0f}"
        panel.c.text = f"amortiguacion c: {state['c']:.1f}"
        panel.substeps.text = f"substeps: {state['substeps']}"
        print(
            f"[masa_resorte] metodo={metodo} k={state['k']:.0f} "
            f"c={state['c']:.1f} substeps={state['substeps']}"
        )

    def avanzar_simulacion(_dt):
        # paso fijo de 1/60 s (no el dt real del reloj) para que la estabilidad
        # de la simulacion no dependa de la tasa de cuadros.
        fijos = () if state["agarrado"] is None else (state["agarrado"],)
        hebra.paso(
            1.0 / 60.0,
            METODOS[state["metodo_index"]],
            state["k"], state["c"], state["substeps"],
            fijos=fijos,
        )
        # la energia cinetica se dispara cuando el integrador explicito explota.
        if hebra.exploto:
            panel.energia.text = "energia: INESTABLE (reinicia con R)"
            panel.energia.color = (255, 90, 90, 255)
        else:
            panel.energia.text = f"energia cinetica: {hebra.energia_cinetica():.0f}"
            panel.energia.color = (220, 220, 220, 255)

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.M:
            state["metodo_index"] = (state["metodo_index"] + 1) % len(METODOS)
        elif symbol == key.COMMA:
            state["k"] = max(10.0, state["k"] - 50.0)
        elif symbol == key.PERIOD:
            state["k"] += 50.0
        elif symbol == key.MINUS:
            state["c"] = max(0.0, state["c"] - 0.5)
        elif symbol == key.EQUAL:
            state["c"] += 0.5
        elif symbol == key.Z:
            state["substeps"] = max(1, state["substeps"] - 1)
        elif symbol == key.X:
            state["substeps"] += 1
        elif symbol == key.SPACE:
            # empujon lateral a las masas libres (el ancla no se mueve).
            hebra.velocities[1:, 0] += 250.0
        elif symbol == key.R:
            hebra.reset()
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    def nodo_mas_cercano(x, y, radio=30.0):
        # devuelve el indice de la masa mas cercana al cursor (excluye el ancla,
        # indice 0), o None si ninguna esta dentro del radio de agarre.
        distancias = np.linalg.norm(hebra.positions[1:] - np.array([x, y]), axis=1)
        indice = int(np.argmin(distancias)) + 1
        return indice if distancias[indice - 1] <= radio else None

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        state["agarrado"] = nodo_mas_cercano(x, y)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if state["agarrado"] is not None:
            hebra.positions[state["agarrado"]] = (x, y)
            hebra.velocities[state["agarrado"]] = 0.0

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        state["agarrado"] = None

    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        primitivas = []

        # resortes: una linea por arista, coloreada segun el estiramiento.
        estiramientos = hebra.estiramiento_relativo()
        for i, estiramiento in enumerate(estiramientos):
            x0, y0 = hebra.positions[i]
            x1, y1 = hebra.positions[i + 1]
            primitivas.append(
                pyglet.shapes.Line(
                    x0, y0, x1, y1,
                    thickness=3,
                    color=_color_por_estiramiento(estiramiento),
                    batch=batch,
                )
            )

        # masas: circulos. el ancla es un cuadrado para distinguirla.
        ax, ay = hebra.positions[0]
        ancla_rect = pyglet.shapes.Rectangle(
            ax - 9, ay - 9, 18, 18, color=(200, 200, 80), batch=batch
        )
        primitivas.append(ancla_rect)
        for i in range(1, hebra.n_nodos):
            cx, cy = hebra.positions[i]
            color = (240, 200, 80) if i == state["agarrado"] else (90, 170, 230)
            primitivas.append(
                pyglet.shapes.Circle(cx, cy, 8, color=color, batch=batch)
            )

        batch.draw()
        panel.draw()

    apply_state()
    pyglet.clock.schedule_interval(avanzar_simulacion, 1.0 / 60.0)
    pyglet.app.run()


if __name__ == "__main__":
    masa_resorte()
