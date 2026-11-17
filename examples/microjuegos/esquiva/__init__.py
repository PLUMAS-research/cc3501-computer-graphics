"""Microjuego de ejemplo: esquivar las rocas que caen.

Las rocas son círculos de pymunk que caen con la gravedad del espacio que crea
el anfitrión y chocan entre sí (unidad 15). La escena se dibuja en el fragment
program, que calcula para cada píxel su distancia al jugador y a cada roca
(unidad 03). El anfitrión avanza la física y dibuja la ventana con ese
programa, así que este archivo no tiene llamadas a OpenGL.
"""

import random

import pymunk

ROL = "microjuego"
INSTRUCCION = "¡Esquiva!"
GANA_SI_SOBREVIVE = True

ANCHO, ALTO = 800, 600  # la ventana del anfitrión, en píxeles
ROCAS_POR_DIFICULTAD = {1: 5, 2: 8, 3: 12}  # a lo más 16, el largo de rocas[] en el shader
RADIO_JUGADOR = 22.0
RADIO_ROCA = 18.0
RAPIDEZ_JUGADOR = 420.0  # px/s
ALTURA_JUGADOR = 60.0


def lanzar(roca, dificultad, altura_extra):
    """Pone la roca sobre la ventana, en una columna al azar."""
    roca.position = (random.uniform(RADIO_ROCA, ANCHO - RADIO_ROCA), ALTO + RADIO_ROCA + random.uniform(0, altura_extra))
    # desde la dificultad 2 las rocas también se mueven de lado
    roca.velocity = (random.uniform(-80, 80) * (dificultad - 1), 0)


def iniciar(dificultad, espacio):
    # paredes a los lados, para que las rocas reboten
    for x in (0, ANCHO):
        pared = pymunk.Segment(espacio.static_body, (x, 0), (x, 3 * ALTO), 1.0)
        pared.elasticity = 0.8
        espacio.add(pared)

    # el jugador es cinemático: se mueve con la velocidad que le damos y los
    # choques no lo desvían
    jugador = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    jugador.position = (ANCHO / 2, ALTURA_JUGADOR)
    espacio.add(jugador, pymunk.Circle(jugador, RADIO_JUGADOR))

    rocas = []
    for _ in range(ROCAS_POR_DIFICULTAD[dificultad]):
        roca = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, RADIO_ROCA))
        # escalonadas sobre la ventana, para que no caigan todas juntas
        lanzar(roca, dificultad, ALTO)
        forma = pymunk.Circle(roca, RADIO_ROCA)
        forma.elasticity = 0.8
        espacio.add(roca, forma)
        rocas.append(roca)

    return {"dificultad": dificultad, "jugador": jugador, "rocas": rocas, "resultado": None}


def actualizar(estado, dt, entrada):
    jugador = estado["jugador"]
    direccion = entrada["derecha"] - entrada["izquierda"]
    x = jugador.position.x
    if estado["resultado"] is not None or (x < RADIO_JUGADOR and direccion < 0) or (x > ANCHO - RADIO_JUGADOR and direccion > 0):
        direccion = 0
    jugador.velocity = (direccion * RAPIDEZ_JUGADOR, 0)

    for roca in estado["rocas"]:
        # la roca que sale por abajo vuelve a caer desde arriba, en otra columna
        if roca.position.y < -RADIO_ROCA:
            lanzar(roca, estado["dificultad"], ALTO / 2)
        if (roca.position - jugador.position).length < RADIO_JUGADOR + RADIO_ROCA:
            if estado["resultado"] is None:
                estado["resultado"] = False


def uniformes(estado):
    return {
        "jugador": estado["jugador"].position,
        "rocas": [roca.position for roca in estado["rocas"]],
        "cantidad_rocas": len(estado["rocas"]),
        "radio_jugador": RADIO_JUGADOR,
        "radio_roca": RADIO_ROCA,
        "golpeado": estado["resultado"] is False,
    }
