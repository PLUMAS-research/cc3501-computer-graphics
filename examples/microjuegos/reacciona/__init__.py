"""Microjuego de ejemplo: golpear al topo con las flechas del teclado.

Hay cuatro hoyos, dispuestos como las flechas del teclado. Cuando un topo se
asoma, hay que golpearlo con la flecha de su hoyo antes de que se esconda. Un
golpe en un hoyo sin topo pierde el microjuego.

La escena se dibuja con sprites de pyglet, sin GLSL. Un sprite es un rectángulo
con una textura (unidad 11) y tiene una posición, una rotación y una escala
(unidad 08). El mazo gira alrededor del punto de anclaje de su imagen, que
está en el mango, y el topo golpeado se aplasta con una escala distinta en
cada eje.

Los sprites se dibujan de atrás hacia adelante, según el orden de sus grupos.
Cada hoyo tiene tres capas: el montículo con el hoyo, el topo y la parte de
adelante del montículo. Esta última imagen es transparente sobre el borde del
hoyo (unidad 12), así que tapa la parte del topo que sigue bajo tierra. La
fila de abajo se dibuja después que la de arriba, porque está más cerca.
"""

import random
from pathlib import Path

import pyglet

ROL = "microjuego"
INSTRUCCION = "¡Reacciona!"
GANA_SI_SOBREVIVE = False
DURACION = 6.0

CARPETA = Path(__file__).parent

# centro de cada hoyo, en píxeles de la ventana
HOYOS = {"arriba": (400, 430), "izquierda": (190, 230), "abajo": (400, 230), "derecha": (610, 230)}
FILAS = {"arriba": 0, "izquierda": 1, "abajo": 1, "derecha": 1}
GIRO_TECLA = {"arriba": 0, "derecha": 90, "abajo": 180, "izquierda": 270}  # grados, en sentido horario

META = {1: 4, 2: 5, 3: 6}  # topos que hay que golpear para ganar
ASOMADO = {1: 0.7, 2: 0.6, 3: 0.5}  # segundos que pasa cada topo fuera del hoyo
SUBIDA = 0.1  # segundos que tarda el topo en salir y en esconderse
PAUSA = (0.1, 0.3)  # segundos entre un topo y el siguiente, al azar en este rango
ATURDIDO = 0.2  # segundos que el topo golpeado tarda en esconderse
# el primer topo sale cuando el anfitrión deja de mostrar la instrucción, que
# tapa la cabeza de los topos de la fila de abajo
PRIMER_TOPO = 0.8
DURACION_MAZO = 0.2

# En monticulo.png, el centro del hoyo está a esta altura desde abajo.
CENTRO_HOYO = 126
# Altura de la base del topo respecto del centro del hoyo. Escondido, su cabeza
# queda bajo el borde delantero del hoyo; asomado, solo la base sigue adentro.
BASE_ESCONDIDO = -123
BASE_ASOMADO = -25

SONIDOS = {
    nombre: pyglet.media.load(str(CARPETA / "sonidos" / f"{nombre}.wav"), streaming=False)
    for nombre in ("aparece", "golpe", "fallo", "logrado")
}


def imagen(nombre, anclaje_x, anclaje_y):
    """Carga una imagen de la carpeta imagenes y fija el punto que se ubica en la posición del sprite."""
    textura = pyglet.image.load(str(CARPETA / "imagenes" / f"{nombre}.png")).get_texture()
    textura.anchor_x, textura.anchor_y = anclaje_x, anclaje_y
    return textura


def preparar(ventana):
    imagenes = {
        "pasto": imagen("pasto", 0, 0),
        "monticulo": imagen("monticulo", 100, CENTRO_HOYO),
        "monticulo_frente": imagen("monticulo_frente", 100, CENTRO_HOYO),
        # el topo se ancla en el centro de su base y la tecla en su centro
        "topo": imagen("topo", 55, 0),
        "topo_golpeado": imagen("topo_golpeado", 55, 0),
        "tecla": imagen("tecla", 22, 22),
        # el mazo gira alrededor del extremo del mango
        "mazo": imagen("mazo", 35, 0),
    }
    lote = pyglet.graphics.Batch()

    def capa(orden):
        return pyglet.graphics.Group(order=orden)

    recursos = {"lote": lote, "imagenes": imagenes, "topos": {}, "decorado": []}
    recursos["decorado"].append(pyglet.sprite.Sprite(imagenes["pasto"], 0, 0, batch=lote, group=capa(0)))
    for tecla, (x, y) in HOYOS.items():
        atras = 1 + 3 * FILAS[tecla]
        recursos["decorado"].append(pyglet.sprite.Sprite(imagenes["monticulo"], x, y, batch=lote, group=capa(atras)))
        recursos["topos"][tecla] = pyglet.sprite.Sprite(imagenes["topo"], x, y, batch=lote, group=capa(atras + 1))
        recursos["decorado"].append(
            pyglet.sprite.Sprite(imagenes["monticulo_frente"], x, y, batch=lote, group=capa(atras + 2)))
        etiqueta = pyglet.sprite.Sprite(imagenes["tecla"], x, y - 80, batch=lote, group=capa(atras + 2))
        etiqueta.rotation = GIRO_TECLA[tecla]
        recursos["decorado"].append(etiqueta)
    recursos["mazo"] = pyglet.sprite.Sprite(imagenes["mazo"], batch=lote, group=capa(7))
    # un topo pequeño por cada acierto que pide la meta
    recursos["contador"] = []
    for indice in range(max(META.values())):
        icono = pyglet.sprite.Sprite(imagenes["topo"], 40 + 45 * indice, 545, batch=lote, group=capa(8))
        icono.scale = 0.35
        recursos["contador"].append(icono)
    return recursos


def iniciar(dificultad, espacio):
    # este microjuego no usa el espacio de pymunk
    return {
        "meta": META[dificultad],
        "asomado": ASOMADO[dificultad],
        "aciertos": 0,
        "hoyo": None,  # el hoyo del topo que está afuera
        "anterior": None,  # el hoyo del topo anterior, para no repetirlo
        "vida": 0.0,  # segundos desde que salió el topo
        "salida": 0.0,  # vida a la que el topo termina de esconderse
        "golpeado": False,
        "espera": PRIMER_TOPO,  # segundos hasta que salga el próximo topo
        "mazo": None,  # el hoyo que golpeó el mazo
        "vida_mazo": 0.0,
        "teclas_antes": {tecla: False for tecla in HOYOS},
        "resultado": None,
    }


def altura_topo(estado):
    """0 con el topo escondido, 1 con el topo asomado."""
    if estado["hoyo"] is None:
        return 0.0
    return min(max(min(estado["vida"], estado["salida"] - estado["vida"]) / SUBIDA, 0.0), 1.0)


def golpear(estado, tecla):
    estado["mazo"], estado["vida_mazo"] = tecla, 0.0
    if tecla != estado["hoyo"]:
        # presionar varias flechas a la vez también pierde, porque una de ellas falla
        estado["resultado"] = False
        SONIDOS["fallo"].play()
    elif not estado["golpeado"]:
        estado["golpeado"] = True
        estado["salida"] = estado["vida"] + ATURDIDO
        estado["aciertos"] += 1
        SONIDOS["golpe"].play()
        if estado["aciertos"] == estado["meta"]:
            estado["resultado"] = True
            SONIDOS["logrado"].play()


def actualizar(estado, dt, entrada):
    # el anfitrión informa qué flechas están presionadas; una tecla se acaba de
    # presionar si está presionada ahora y no lo estaba en el cuadro anterior
    recien = [tecla for tecla in HOYOS if entrada[tecla] and not estado["teclas_antes"][tecla]]
    estado["teclas_antes"] = {tecla: entrada[tecla] for tecla in HOYOS}

    if estado["hoyo"] is None:
        estado["espera"] -= dt
        if estado["espera"] <= 0 and estado["resultado"] is None:
            hoyo = random.choice([tecla for tecla in HOYOS if tecla != estado["anterior"]])
            estado.update(hoyo=hoyo, vida=0.0, salida=estado["asomado"], golpeado=False)
            SONIDOS["aparece"].play()
    else:
        estado["vida"] += dt
        if estado["vida"] >= estado["salida"]:
            estado.update(anterior=estado["hoyo"], hoyo=None, espera=random.uniform(*PAUSA))

    for tecla in recien:
        if estado["resultado"] is None:
            golpear(estado, tecla)
    estado["vida_mazo"] += dt


def dibujar(recursos, estado):
    altura = altura_topo(estado)
    for tecla, (x, y) in HOYOS.items():
        topo = recursos["topos"][tecla]
        topo.visible = tecla == estado["hoyo"] and altura > 0
        if topo.visible:
            topo.y = y + BASE_ESCONDIDO + altura * (BASE_ASOMADO - BASE_ESCONDIDO)
            topo.image = recursos["imagenes"]["topo_golpeado" if estado["golpeado"] else "topo"]
            # el golpe lo aplasta: más ancho y más bajo
            topo.scale_x, topo.scale_y = (1.2, 0.8) if estado["golpeado"] else (1.0, 1.0)

    mazo = recursos["mazo"]
    mazo.visible = estado["mazo"] is not None and estado["vida_mazo"] < DURACION_MAZO
    if mazo.visible:
        x, y = HOYOS[estado["mazo"]]
        mazo.position = (x + 90, y + 10, 0)
        # baja en 0.06 s, girando en sentido antihorario hasta quedar sobre el hoyo
        mazo.rotation = -15 - 46 * min(estado["vida_mazo"] / 0.06, 1.0)

    for indice, icono in enumerate(recursos["contador"]):
        icono.visible = indice < estado["meta"]
        icono.opacity = 255 if indice < estado["aciertos"] else 70

    recursos["lote"].draw()
