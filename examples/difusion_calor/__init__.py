"""Ejemplo atomico euleriano: difusion de calor 2D sobre una grilla fija.

Complementa a masa_resorte. Aqui no se discretiza el material sino el espacio:
la grilla no se mueve y el campo de temperatura T evoluciona sobre ella segun la
ecuacion del calor, resuelta con Euler explicito.

El campo se sube como una textura de un canal y un fragment program le aplica un
colormap. El alumno inyecta calor con el mouse y observa como difunde.

La leccion de estabilidad es la misma que en masa_resorte. El metodo explicito
exige el numero de difusion r = alpha * dt / dx^2 <= 1/4. Si se sube alpha o se
bajan las subdivisiones temporales, r supera el limite y el campo desarrolla un
patron de tablero que crece sin control.

Controles:
- , / .:    baja / sube la difusividad alpha.
- z / x:    baja / sube las subdivisiones temporales (substeps).
- B:        alterna borde dirichlet (calor escapa) / neumann (aislado).
- mouse:    arrastra para inyectar calor.
- R:        reinicia el campo.
- ESC:      cierra la ventana.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL

from grafica.utils import load_pipeline

from .field import BORDES, CFL_MAXIMO, CampoCalor

DEFAULTS = {
    "alpha": 30.0,
    "substeps": 4,
    "borde_index": 1,  # neumann: el calor se conserva y se ve mejor la difusion.
}
RADIO_PINCEL = 6  # en celdas.


def _crear_textura(n):
    """Textura de un canal float (R32F) de n x n, sin datos iniciales."""
    textura = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, textura)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_R32F, n, n, 0, GL.GL_RED, GL.GL_FLOAT, None
    )
    return textura


class Panel:
    """Panel de informacion en pantalla: fondo translucido y etiquetas FiraCode.

    Agrupa la creacion de las etiquetas para que no ensucie el cuerpo principal.
    apply_state() solo escribe en `panel.alpha`, `panel.substeps`, etc.; este
    objeto se encarga de la posicion, la fuente y el dibujado.
    """

    def __init__(self, lado):
        self.batch = pyglet.graphics.Batch()

        # fondo oscuro para que el texto se lea sobre el campo de calor.
        self._fondo = pyglet.shapes.Rectangle(
            0, lado - 116, 360, 116, color=(20, 20, 20), batch=self.batch
        )
        self._fondo.opacity = 170

        def etiqueta(fila, size=12):
            return pyglet.text.Label(
                "",
                font_name="Fira Code",
                font_size=size,
                x=14,
                y=lado - 22 - fila * 22,
                color=(230, 230, 230, 255),
                batch=self.batch,
            )

        self.alpha = etiqueta(0)
        self.substeps = etiqueta(1)
        self.borde = etiqueta(2)
        self.cfl = etiqueta(3)

        # linea fija con los controles, al pie de la ventana.
        self._controles = pyglet.text.Label(
            ", . alpha   z x substeps   B borde   mouse calor   R reset",
            font_name="Fira Code",
            font_size=10,
            x=14,
            y=12,
            color=(150, 150, 150, 255),
            batch=self.batch,
        )

    def draw(self):
        self.batch.draw()


@click.command("difusion_calor", short_help="Difusion de calor 2D (enfoque euleriano)")
@click.option("--n", type=int, default=150, help="celdas por lado de la grilla")
@click.option("--escala", type=int, default=5, help="pixeles por celda")
def difusion_calor(n, escala):
    lado = n * escala
    window = pyglet.window.Window(lado, lado, caption="difusion de calor 2D")

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    campo = CampoCalor(n)
    campo.inyectar(n // 2, n // 2, 10, 1.0)  # un punto caliente inicial.

    state = dict(DEFAULTS)

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    # quad que cubre toda la pantalla en coordenadas NDC.
    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    uv = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    quad = pipeline.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    quad.position[:] = vertices
    quad.uv[:] = uv

    textura = _crear_textura(n)
    pipeline.use()
    pipeline["campo_temperatura"] = 0  # unidad de textura 0.

    panel = Panel(lado)

    def apply_state():
        """Traduce el estado actual a las etiquetas y al terminal.

        Es el unico lugar que convierte el estado en texto: cada tecla muta
        `state` y llama aqui, asi las etiquetas nunca se desincronizan.
        """
        borde = BORDES[state["borde_index"]]
        # numero de difusion r = alpha * dt / dx^2 (con dx = 1). El metodo
        # explicito solo es estable si r <= 1/4 (condicion CFL)
        numero_difusion = campo.numero_difusion(
            state["alpha"], 1.0 / 60.0, state["substeps"]
        )
        estable = numero_difusion <= CFL_MAXIMO
        panel.alpha.text = f"difusividad alpha: {state['alpha']:.0f}"
        panel.substeps.text = f"substeps: {state['substeps']}"
        panel.borde.text = f"borde: {borde}"
        panel.cfl.text = (
            f"numero de difusion r = {numero_difusion:.3f}  "
            f"(estable si <= {CFL_MAXIMO})"
        )
        panel.cfl.color = (130, 220, 130, 255) if estable else (255, 120, 120, 255)
        print(
            f"[difusion_calor] alpha={state['alpha']:.0f} substeps={state['substeps']} "
            f"borde={borde} r={numero_difusion:.3f} {'estable' if estable else 'INESTABLE'}"
        )

    def avanzar_simulacion(_dt):
        # usamos un paso fijo de 1/60 s, no el dt real del reloj, para que la
        # estabilidad de la simulacion no dependa de la tasa de cuadros.
        campo.paso(
            1.0 / 60.0,
            state["alpha"],
            state["substeps"],
            BORDES[state["borde_index"]],
        )

    def pintar_calor(x, y):
        # convertimos el pixel del mouse (origen abajo-izquierda) a celda de la
        # grilla y depositamos un disco caliente alrededor.
        columna = int(x / escala)
        fila = int(y / escala)
        if 0 <= columna < n and 0 <= fila < n:
            campo.inyectar(columna, fila, RADIO_PINCEL, 1.0)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        pintar_calor(x, y)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        pintar_calor(x, y)

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.COMMA:
            state["alpha"] = max(1.0, state["alpha"] - 5.0)
        elif symbol == key.PERIOD:
            state["alpha"] += 5.0
        elif symbol == key.Z:
            state["substeps"] = max(1, state["substeps"] - 1)
        elif symbol == key.X:
            state["substeps"] += 1
        elif symbol == key.B:
            state["borde_index"] = (state["borde_index"] + 1) % len(BORDES)
        elif symbol == key.R:
            campo.reset()
            campo.inyectar(n // 2, n // 2, 10, 1.0)
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    @window.event
    def on_draw():
        window.clear()
        # subimos el campo actual al canal R de la textura.
        datos = np.ascontiguousarray(campo.T, dtype=np.float32)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, textura)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, n, n, GL.GL_RED, GL.GL_FLOAT, datos
        )
        pipeline.use()
        quad.draw(GL.GL_TRIANGLES)
        panel.draw()

    apply_state()
    pyglet.clock.schedule_interval(avanzar_simulacion, 1.0 / 60.0)
    pyglet.app.run()


if __name__ == "__main__":
    difusion_calor()
