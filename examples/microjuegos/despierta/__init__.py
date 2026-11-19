"""Microjuego de ejemplo: no quedarse dormido.

Suena el despertador y los párpados se cierran solos, cada vez más rápido.
Cada vez que se presiona espacio, el jugador se da una cachetada y abre los
ojos un poco. Si los párpados se cierran del todo antes de que se acabe el
tiempo, se queda dormido y pierde.

La escena se dibuja en el fragment program, como en el ejemplo esquiva (unidad
03). El sueño se nota en tres efectos que calcula cada píxel: los párpados, que
dejan ver solo el interior de una elipse; la visión doble, que evalúa la escena
dos veces con un desplazamiento; y la pérdida de color.

Los sonidos se sintetizan con NumPy. Cada sonido es una función del tiempo
evaluada en cada muestra, así como el fragment program evalúa una función en
cada píxel.
"""

import io
import math
import wave

import numpy as np
import pyglet

ROL = "microjuego"
INSTRUCCION = "¡Despierta!"
GANA_SI_SOBREVIVE = True
DURACION = 5.0

# rapidez con que se cierran los párpados, en fracción de la apertura por segundo
RAPIDEZ_SUENO = {1: 0.30, 2: 0.40, 3: 0.50}
CACHETADA = 0.15  # fracción de la apertura que recupera cada cachetada
PERIODO_ALARMA = 1.0  # segundos entre el comienzo de una ráfaga de pitidos y la siguiente
PERIODO_RONQUIDO = 1.6

# %% Sonidos

MUESTREO = 44100  # muestras por segundo


def tiempos(duracion):
    """El instante de cada muestra de un sonido de esta duración."""
    return np.arange(int(duracion * MUESTREO)) / MUESTREO


def fuente(senal):
    """Convierte una señal con valores en [-1, 1] en una fuente de pyglet.

    La señal se escribe como un archivo WAV en memoria, con muestras de 16 bits.
    """
    archivo = io.BytesIO()
    with wave.open(archivo, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(MUESTREO)
        wav.writeframes((np.clip(senal, -1, 1) * 32767).astype("<i2").tobytes())
    archivo.seek(0)
    return pyglet.media.load("sonido.wav", file=archivo, streaming=False)


def alarma():
    """Cuatro pitidos de 70 ms a 1050 Hz, con un armónico cuadrado que los hace más ásperos."""
    t = tiempos(0.5)
    fase = t % 0.125
    # cada pitido sube y baja en 5 ms, porque un corte abrupto suena como un clic
    puerta = np.clip(np.minimum(fase, 0.07 - fase) / 0.005, 0, 1)
    tono = np.sin(2 * np.pi * 1050 * t) + 0.3 * np.sign(np.sin(2 * np.pi * 2100 * t))
    return 0.3 * puerta * tono


def cachetada():
    """Ruido que se apaga en pocos milisegundos, sobre un golpe grave."""
    t = tiempos(0.15)
    ruido = np.random.default_rng(1).uniform(-1, 1, t.size)
    golpe = np.sin(2 * np.pi * 160 * t)
    return 0.6 * ruido * np.exp(-t / 0.015) + 0.5 * golpe * np.exp(-t / 0.04)


def ronquido():
    """Ruido grave que vibra 30 veces por segundo, como el paladar al roncar."""
    t = tiempos(1.0)
    # el promedio móvil de 40 muestras es un filtro pasa bajos: deja el ruido grave
    ruido = np.convolve(np.random.default_rng(2).uniform(-1, 1, t.size), np.ones(40) / 40, mode="same")
    ruido /= np.abs(ruido).max()
    vibracion = (0.5 + 0.5 * np.sin(2 * np.pi * 30 * t)) ** 2
    envolvente = np.sin(np.pi * t / t[-1]) ** 2
    return 0.7 * ruido * vibracion * envolvente


SONIDOS = {"alarma": fuente(alarma()), "cachetada": fuente(cachetada()), "ronquido": fuente(ronquido())}

# %% Juego


def iniciar(dificultad, espacio):
    # este microjuego no usa el espacio de pymunk
    return {
        "rapidez": RAPIDEZ_SUENO[dificultad],
        "sueno": 0.1,  # 0 con los ojos abiertos, 1 con los ojos cerrados
        "golpe": 0.0,  # 1 justo después de una cachetada; decae a 0
        "tiempo": 0.0,
        "ultimo_sonido": -math.inf,
        "resultado": None,
    }


def actualizar(estado, dt, entrada):
    estado["tiempo"] += dt
    if estado["resultado"] is None:
        # el sueño gana rapidez a medida que se cierran los ojos
        estado["sueno"] += estado["rapidez"] * (0.5 + estado["sueno"]) * dt
        if entrada["espacio_presionado"]:
            estado["sueno"] = max(0.0, estado["sueno"] - CACHETADA)
            estado["golpe"] = 1.0
            SONIDOS["cachetada"].play()
        if estado["sueno"] >= 1.0:
            estado["sueno"] = 1.0
            estado["resultado"] = False
            estado["ultimo_sonido"] = -math.inf  # el primer ronquido suena de inmediato
    estado["golpe"] *= math.exp(-8.0 * dt)

    # mientras está despierto suena la alarma; dormido, ronca
    dormido = estado["resultado"] is False
    periodo = PERIODO_RONQUIDO if dormido else PERIODO_ALARMA
    if estado["tiempo"] - estado["ultimo_sonido"] >= periodo:
        SONIDOS["ronquido" if dormido else "alarma"].play()
        estado["ultimo_sonido"] = estado["tiempo"]


def uniformes(estado):
    dormido = estado["resultado"] is False
    return {
        "sueno": estado["sueno"],
        "golpe": estado["golpe"],
        "dormido": dormido,
        # el despertador tiembla mientras se escuchan los pitidos
        "sonando": not dormido and estado["tiempo"] - estado["ultimo_sonido"] < SONIDOS["alarma"].duration,
    }
