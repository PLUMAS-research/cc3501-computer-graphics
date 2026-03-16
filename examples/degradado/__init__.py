import colorsys

import click
import numpy as np
import pyglet
from pyglet import shapes
from pyglet.window import key


def _lerp(a, b, t):
    """Interpolación lineal entre a y b."""
    return a + (b - a) * t


def _lerp_hue(h0, h1, t):
    """Interpolación de matiz por el camino más corto en el círculo [0, 1)."""
    diff = h1 - h0
    if diff > 0.5:
        diff -= 1.0
    elif diff < -0.5:
        diff += 1.0
    return (h0 + diff * t) % 1.0


def _bilinear_rgb(esquinas, altura, ancho):
    """Interpolación bilineal en espacio RGB.

    esquinas: (abajo-izq, abajo-der, arriba-izq, arriba-der) como arrays RGB [0,1].
    """
    bl, br, tl, tr = [np.array(c) for c in esquinas]
    imagen = np.zeros((altura, ancho, 3))

    for y in range(altura):
        ty = y / (altura - 1) if altura > 1 else 0
        for x in range(ancho):
            tx = x / (ancho - 1) if ancho > 1 else 0
            bot = _lerp(bl, br, tx)
            top = _lerp(tl, tr, tx)
            imagen[y, x] = _lerp(bot, top, ty)

    return np.clip(imagen, 0, 1)


def _bilinear_hsv(esquinas, altura, ancho):
    """Interpolación bilineal en espacio HSV, resultado en RGB.

    esquinas: (abajo-izq, abajo-der, arriba-izq, arriba-der) como arrays RGB [0,1].
    Se convierten a HSV, se interpolan, y se devuelven en RGB.
    """
    hsv_corners = [colorsys.rgb_to_hsv(*c) for c in esquinas]
    bl, br, tl, tr = hsv_corners
    imagen = np.zeros((altura, ancho, 3))

    for y in range(altura):
        ty = y / (altura - 1) if altura > 1 else 0
        for x in range(ancho):
            tx = x / (ancho - 1) if ancho > 1 else 0

            # interpolación bilineal en HSV
            bot_h = _lerp_hue(bl[0], br[0], tx)
            bot_s = _lerp(bl[1], br[1], tx)
            bot_v = _lerp(bl[2], br[2], tx)

            top_h = _lerp_hue(tl[0], tr[0], tx)
            top_s = _lerp(tl[1], tr[1], tx)
            top_v = _lerp(tl[2], tr[2], tx)

            h = _lerp_hue(bot_h, top_h, ty)
            s = _lerp(bot_s, top_s, ty)
            v = _lerp(bot_v, top_v, ty)

            imagen[y, x] = colorsys.hsv_to_rgb(h, s, v)

    return np.clip(imagen, 0, 1)


def _build_apunte_image(altura, ancho):
    """Degradado rojo horizontal (lineal) y azul vertical (cuadrático) en RGB.

    Ejemplo del apunte de Imágenes, Píxeles y Colores.
    """
    imagen = np.zeros((altura, ancho, 3))

    for x in range(ancho):
        imagen[:, x, 0] = x / ancho

    for y in range(altura):
        imagen[y, :, 2] = np.power(y, 2) / np.power(altura, 2)

    return imagen


def _build_pacman(altura, ancho):
    """Pac-Man en pixel art, escalado a la grilla."""
    # sprite de 15x15
    K = (0.0, 0.0, 0.0)  # negro (fondo)
    Y = (1.0, 0.8, 0.0)  # amarillo
    E = (0.1, 0.1, 0.3)  # ojo
    sprite = [
        [K, K, K, K, K, Y, Y, Y, Y, Y, K, K, K, K, K],
        [K, K, K, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K, K],
        [K, K, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K],
        [K, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K],
        [K, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K],
        [Y, Y, Y, E, E, Y, Y, Y, Y, Y, Y, Y, K, K, K],
        [Y, Y, Y, E, E, Y, Y, Y, Y, Y, K, K, K, K, K],
        [Y, Y, Y, Y, Y, Y, Y, Y, K, K, K, K, K, K, K],
        [Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K, K, K, K],
        [Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K, K],
        [K, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K],
        [K, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K],
        [K, K, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K],
        [K, K, K, Y, Y, Y, Y, Y, Y, Y, Y, Y, K, K, K],
        [K, K, K, K, K, Y, Y, Y, Y, Y, K, K, K, K, K],
    ]

    # el sprite se define de arriba a abajo; invertir para que y=0 sea abajo
    sprite = sprite[::-1]

    imagen = np.zeros((altura, ancho, 3))
    for y in range(altura):
        sy = int(y * 15 / altura)
        for x in range(ancho):
            sx = int(x * 15 / ancho)
            imagen[y, x] = sprite[sy][sx]

    return imagen


def _build_hsv_rainbow(altura, ancho):
    """Degradado de matiz horizontal y valor vertical (cuadrático) en HSV."""
    imagen = np.zeros((altura, ancho, 3))

    for y in range(altura):
        for x in range(ancho):
            h = x / ancho
            s = 1.0
            v = np.power(y, 2) / np.power(altura, 2)
            imagen[y, x] = colorsys.hsv_to_rgb(h, s, v)

    return imagen


@click.command(
    "degradado", short_help="Ejemplo de imágenes raster con degradados de colores"
)
@click.option("--size", default=15, help="Cantidad de celdas por lado")
def degradado(size):
    """Genera imágenes con degradados de color.

    La barra de espacio alterna entre distintos modos.
    """
    altura, ancho = size, size

    # esquinas para la interpolación bilineal (RGB)
    # rojo, verde, azul, amarillo: colores donde la diferencia RGB/HSV es notoria
    esquinas = [
        (1.0, 0.0, 0.0),  # abajo-izq: rojo
        (0.0, 0.0, 1.0),  # abajo-der: azul
        (1.0, 1.0, 0.0),  # arriba-izq: amarillo
        (0.0, 1.0, 1.0),  # arriba-der: cian
    ]

    modos = [
        ("RGB (apunte)", _build_apunte_image(altura, ancho)),
        ("Interpolación RGB", _bilinear_rgb(esquinas, altura, ancho)),
        ("Interpolación HSV", _bilinear_hsv(esquinas, altura, ancho)),
        ("HSV (arcoíris)", _build_hsv_rainbow(altura, ancho)),
        ("waka waka", _build_pacman(altura, ancho)),
    ]

    cell_size = 40
    padding = 1

    win_width = ancho * cell_size + padding
    win_height = altura * cell_size + padding

    nombre, _ = modos[0]
    win = pyglet.window.Window(win_width, win_height, caption=nombre)

    batch = pyglet.graphics.Batch()
    rects = []

    for y in range(altura):
        for x in range(ancho):
            r, g, b = modos[0][1][y, x]
            color = (int(r * 255), int(g * 255), int(b * 255))

            px = x * cell_size + padding
            py = y * cell_size + padding

            rect = shapes.Rectangle(
                px,
                py,
                cell_size - padding,
                cell_size - padding,
                color=color,
                batch=batch,
            )
            rects.append(rect)

    modo_actual = 0

    def _apply_image(imagen):
        for y in range(altura):
            for x in range(ancho):
                r, g, b = imagen[y, x]
                rects[y * ancho + x].color = (int(r * 255), int(g * 255), int(b * 255))

    @win.event
    def on_key_press(symbol, modifiers):
        nonlocal modo_actual
        if symbol == key.SPACE:
            modo_actual = (modo_actual + 1) % len(modos)
            nombre, imagen = modos[modo_actual]
            _apply_image(imagen)
            win.set_caption(nombre)

    @win.event
    def on_draw():
        win.clear()
        batch.draw()

    pyglet.app.run()
