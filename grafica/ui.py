"""Utilidades para dibujar elementos 2D de UI encima de una escena 3D.

El problema: en Windows algunos drivers producen z-fighting entre los glifos
de texto de pyglet y la geometría 3D si el depth test o la escritura al depth
buffer siguen activos al llamar a `label.draw()`. Este módulo ofrece un
context manager que fija el estado correcto y lo restaura al salir.

También ofrece `InfoPanel`, una fachada sobre `pyglet.graphics.Batch` y
`pyglet.text.Label` que esconde el andamiaje de UI de los ejemplos (crear
etiquetas, posicionarlas, mantener sus referencias y dibujarlas) para que el
código de cada ejemplo se quede con su concepto medular.
"""

from contextlib import contextmanager
from pathlib import Path

import pyglet
import pyglet.gl as GL

FIRACODE_PATH = (
    Path(__file__).parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf"
)
_firacode_cargada = False


def load_firacode():
    """Registra la fuente FiraCode en pyglet una sola vez (idempotente).

    Los ejemplos usan FiraCode en sus etiquetas porque las fuentes del sistema
    no están disponibles en todas las plataformas (Windows en particular).
    """
    global _firacode_cargada
    if not _firacode_cargada:
        pyglet.font.add_file(str(FIRACODE_PATH))
        _firacode_cargada = True


@contextmanager
def ui_overlay():
    """Configura estado de OpenGL para dibujar UI 2D por encima de la escena.

    Desactiva depth test y escritura al depth buffer, activa blending con
    alpha premultiplicado por src. Restaura el estado previo de depth test
    y blending al salir del bloque (la máscara de depth se deja en TRUE,
    que es el valor por defecto que asumen todos los demás ejemplos).

    Uso:
        with ui_overlay():
            label.draw()
            batch.draw()
    """
    depth_test_was_enabled = GL.glIsEnabled(GL.GL_DEPTH_TEST)
    blend_was_enabled = GL.glIsEnabled(GL.GL_BLEND)

    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDepthMask(GL.GL_FALSE)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    try:
        yield
    finally:
        GL.glDepthMask(GL.GL_TRUE)
        if depth_test_was_enabled:
            GL.glEnable(GL.GL_DEPTH_TEST)
        if not blend_was_enabled:
            GL.glDisable(GL.GL_BLEND)


class InfoPanel:
    """Panel de informacion 2D: etiquetas FiraCode en filas, con fondo opcional.

    Es una fachada sobre `pyglet.graphics.Batch` y `pyglet.text.Label`. Esconde
    la carga de la fuente, la creacion de cada etiqueta, su posicion en filas,
    la conservacion de sus referencias (sin esto el recolector de basura las
    elimina y desaparecen de la pantalla) y el dibujado.

    El ejemplo declara las filas con `add()` y luego escribe el texto por clave.
    Asi el patron `apply_state()` queda limpio: solo asigna `panel["clave"]`.

    Uso:
        panel = (InfoPanel(x=14, y_top=alto - 22, background=(20, 20, 20))
                 .add("alpha").add("cfl")
                 .footer(", . alpha   R reset"))
        panel["alpha"] = "difusividad alpha: 30"
        panel.color("cfl", (255, 120, 120, 255))
        # en on_draw:
        panel.draw()

    El color de las filas se controla por clave con `color()`; el de los
    controles del pie queda fijo. El fondo, si se pide, cubre el bloque de
    filas (no el pie) y se dibuja detras del texto mediante grupos ordenados.
    """

    TEXTO = (230, 230, 230, 255)
    CONTROLES = (150, 150, 150, 255)

    def __init__(
        self,
        x=14,
        y_top=678,
        line_height=22,
        font_size=12,
        color=TEXTO,
        background=None,
        background_width=360,
    ):
        load_firacode()
        self.batch = pyglet.graphics.Batch()
        # dos grupos ordenados: el fondo (orden 0) siempre detras del texto (1).
        self._grupo_fondo = pyglet.graphics.Group(order=0)
        self._grupo_texto = pyglet.graphics.Group(order=1)

        self._x = x
        self._y_top = y_top
        self._line_height = line_height
        self._font_size = font_size
        self._color = color
        self._background = background
        self._background_width = background_width

        self._labels = {}
        self._extras = []  # referencias vivas del pie y del fondo.
        self._fondo = None

    def add(self, clave, size=None):
        """Agrega una fila etiquetada vacia. Devuelve self para encadenar."""
        fila = len(self._labels)
        self._labels[clave] = pyglet.text.Label(
            "",
            font_name="Fira Code",
            font_size=size or self._font_size,
            x=self._x,
            y=self._y_top - fila * self._line_height,
            color=self._color,
            batch=self.batch,
            group=self._grupo_texto,
        )
        return self

    def footer(self, texto, font_size=10):
        """Agrega la linea fija de controles al pie. Devuelve self."""
        self._extras.append(
            pyglet.text.Label(
                texto,
                font_name="Fira Code",
                font_size=font_size,
                x=self._x,
                y=14,
                color=self.CONTROLES,
                batch=self.batch,
                group=self._grupo_texto,
            )
        )
        return self

    def __setitem__(self, clave, texto):
        self._labels[clave].text = texto

    def color(self, clave, rgba):
        self._labels[clave].color = rgba

    def _crear_fondo(self):
        # alto calculado desde el numero de filas ya declaradas.
        alto = (len(self._labels) + 1) * self._line_height
        top = self._y_top + self._line_height * 0.8
        self._fondo = pyglet.shapes.Rectangle(
            0,
            top - alto,
            self._background_width,
            alto,
            color=self._background,
            batch=self.batch,
            group=self._grupo_fondo,
        )
        self._fondo.opacity = 170
        self._extras.append(self._fondo)

    def draw(self):
        # el fondo se crea perezosamente en el primer dibujado, cuando ya se
        # conocen todas las filas declaradas con add().
        if self._background is not None and self._fondo is None:
            self._crear_fondo()
        self.batch.draw()
