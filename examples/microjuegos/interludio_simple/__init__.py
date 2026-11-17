"""Interludio por omisión: las vidas como círculos, el puntaje y el aviso del evento.

Es lo mínimo que muestra un interludio. Usa las figuras y etiquetas de pyglet,
así que su único valor es servir de referencia del contrato.
"""

import math

import pyglet

from grafica.ui import load_firacode

ROL = "interludio"

AVISOS = {
    "inicio": "Espacio para empezar",
    "normal": "",
    "acelera": "¡Más rápido!",
    "dificultad": "Dificultad {dificultad}",
    "jefe": "¡Viene el jefe!",
    "fin": "Fin del juego. Espacio para jugar de nuevo",
}
AZUL, MAGENTA, GRIS = (10, 14, 80), (207, 56, 137), (200, 200, 215)
# el fondo cambia según el resultado de la ronda anterior
FONDOS = {True: (205, 235, 200), False: (245, 205, 205), None: (245, 240, 230)}


def preparar(ventana):
    load_firacode()
    ancho, alto = ventana.width, ventana.height
    lote = pyglet.graphics.Batch()
    texto = {"font_name": "Fira Code", "anchor_x": "center", "anchor_y": "center", "batch": lote}
    return {
        "lote": lote,
        "fondo": pyglet.shapes.Rectangle(0, 0, ancho, alto, color=(245, 240, 230), batch=lote),
        "vidas": [],
        "lote_vidas": pyglet.graphics.Batch(),
        "centro": (ancho / 2, alto / 2),
        "puntaje": pyglet.text.Label("", x=ancho / 2, y=alto * 0.72, font_size=64, color=(*AZUL, 255), **texto),
        "aviso": pyglet.text.Label("", x=ancho / 2, y=alto * 0.25, font_size=22, color=(*MAGENTA, 255), **texto),
        "resultado": pyglet.text.Label("", x=ancho / 2, y=alto * 0.88, font_size=20, color=(*AZUL, 255), **texto),
    }


def iniciar(recursos, situacion):
    recursos["fondo"].color = FONDOS[situacion["resultado"]]
    recursos["puntaje"].text = str(situacion["puntaje"])
    recursos["aviso"].text = AVISOS[situacion["evento"]].format(**situacion)
    recursos["resultado"].text = {True: "¡Bien!", False: "¡Ups!", None: ""}[situacion["resultado"]]

    cx, cy = recursos["centro"]
    separacion = 70
    primera = cx - separacion * (situacion["vidas_iniciales"] - 1) / 2
    recursos["vidas"] = [
        pyglet.shapes.Circle(primera + separacion * indice, cy, 24,
                             color=MAGENTA if indice < situacion["vidas"] else GRIS,
                             batch=recursos["lote_vidas"])
        for indice in range(situacion["vidas_iniciales"])
    ]
    # la vida recién perdida es la que se encoge
    perdida = situacion["vidas"] if situacion["resultado"] is False else None
    return {"recursos": recursos, "tiempo": 0.0, "perdida": perdida}


def actualizar(estado, dt):
    estado["tiempo"] += dt
    recursos = estado["recursos"]
    _, cy = recursos["centro"]
    # el puntaje aparece grande y vuelve a su tamaño
    recursos["puntaje"].font_size = 64 + 36 * math.exp(-10 * estado["tiempo"])
    for indice, vida in enumerate(recursos["vidas"]):
        # cada vida cae desde arriba con un rebote amortiguado, una tras otra
        t = max(0.0, estado["tiempo"] - 0.06 * indice)
        vida.y = cy + 90 * math.exp(-6 * t) * abs(math.cos(12 * t))
        if indice == estado["perdida"]:
            vida.radius = 24 * max(0.0, 1.0 - 2 * estado["tiempo"])
        else:
            vida.radius = 24 + 4 * math.sin(10 * estado["tiempo"] + indice)


def dibujar(estado):
    estado["recursos"]["lote"].draw()
    estado["recursos"]["lote_vidas"].draw()
