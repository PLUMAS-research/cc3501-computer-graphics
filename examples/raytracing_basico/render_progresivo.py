"""Render por franjas de un computo CPU lento, con cache por parametros.

Los dos ejemplos de trazado por CPU (`raytracing_basico` y `raytracing_cpu`)
comparten este andamiaje: el trazado tarda segundos, asi que no se calcula de
golpe (congelaria la ventana) sino por bandas de filas a lo largo de varios
cuadros. Cada cuadro se traza una franja, se sube la textura parcial y la imagen
se llena de arriba hacia abajo. Cada imagen terminada se guarda indexada por sus
parametros, asi volver a una configuracion ya trazada se muestra al instante.

El estado vive en un dict plano (`nuevo_estado`) que el ejemplo inspecciona y
estas funciones mutan, el mismo patron `state` + funciones del resto del repo.
Esto NO es el concepto que el ejemplo ensena (eso es el trazado: que rayo, que
escena, que iluminacion, que el ejemplo aporta via el callback `trazar`). Es
andamiaje transversal a los dos ejemplos de ray tracing, por eso vive aqui, en
el ejemplo atomico, y `raytracing_cpu` lo importa, igual que comparten `Camara`.
"""

import time

import numpy as np
import pyglet
import pyglet.gl as GL


def nuevo_estado(
    trazar,
    resoluciones,
    aspecto,
    resolucion_inicial=None,
    filas_por_paso=2,
    presupuesto_tick=0.05,
):
    """Crea el dict de estado del render progresivo.

    trazar -- callback `trazar(ancho, alto, fila_inicio, fila_fin)` que devuelve
        la banda de filas pedida como arreglo (n_filas, ancho, 3) en [0, 1].
    resoluciones -- anchos disponibles; las teclas , . recorren la lista.
    """
    if resolucion_inicial is None:
        indice = 0
    else:
        indice = int(np.argmin([abs(r - resolucion_inicial) for r in resoluciones]))

    return {
        "trazar": trazar,
        "resoluciones": list(resoluciones),
        "aspecto": aspecto,
        "filas_por_paso": filas_por_paso,
        "presupuesto_tick": presupuesto_tick,
        "indice": indice,
        "textura": None,
        "buffer": None,
        "fila": 0,
        "ancho": 0,
        "alto": 0,
        "activo": False,
        "completo": False,
        "segundos": 0.0,
        "inicio": 0.0,
        "clave_actual": None,
        # imagenes terminadas, indexadas por (ancho, alto, *perillas del ejemplo)
        "cache": {},
    }


def resolucion_actual(estado):
    ancho = estado["resoluciones"][estado["indice"]]
    return ancho, int(ancho / estado["aspecto"])


def menos_resolucion(estado):
    estado["indice"] = max(0, estado["indice"] - 1)


def mas_resolucion(estado):
    estado["indice"] = min(len(estado["resoluciones"]) - 1, estado["indice"] + 1)


def clave(estado, extra=()):
    ancho, alto = resolucion_actual(estado)
    return (ancho, alto, *extra)


def porcentaje(estado):
    return int(100 * estado["fila"] / max(1, estado["alto"]))


def mostrar(estado, extra=()):
    """Muestra la imagen cacheada para estos parametros, o lanza el render."""
    clave_pedida = clave(estado, extra)
    if clave_pedida in estado["cache"]:
        estado["textura"], estado["segundos"] = estado["cache"][clave_pedida]
        estado["clave_actual"] = clave_pedida
        estado["completo"] = True
        estado["activo"] = False
    else:
        iniciar(estado, extra)


def iniciar(estado, extra=()):
    """Fuerza un render desde cero, ignorando la cache."""
    estado["ancho"], estado["alto"] = resolucion_actual(estado)
    estado["buffer"] = np.zeros((estado["alto"], estado["ancho"], 3), dtype=np.float32)
    estado["fila"] = 0
    estado["completo"] = False
    estado["activo"] = True
    estado["inicio"] = time.time()
    estado["clave_actual"] = clave(estado, extra)
    _subir_textura(estado)


def avanzar(estado, dt):
    """Traza franjas hasta agotar el presupuesto del cuadro.

    No hace nada si no hay un render activo. Devuelve True el cuadro en que el
    render termina (para que el ejemplo loguee el tiempo).
    """
    if not estado["activo"]:
        return False

    inicio = time.time()
    while estado["fila"] < estado["alto"] and time.time() - inicio < estado["presupuesto_tick"]:
        fila0 = estado["fila"]
        fila1 = min(fila0 + estado["filas_por_paso"], estado["alto"])
        estado["buffer"][fila0:fila1] = estado["trazar"](estado["ancho"], estado["alto"], fila0, fila1)
        estado["fila"] = fila1
    _subir_textura(estado)

    if estado["fila"] >= estado["alto"]:
        estado["activo"] = False
        estado["completo"] = True
        estado["segundos"] = time.time() - estado["inicio"]
        # guardamos la imagen terminada para volver a ella sin recalcular
        estado["cache"][estado["clave_actual"]] = (estado["textura"], estado["segundos"])
        return True
    return False


def _subir_textura(estado):
    # pyglet espera filas de abajo hacia arriba; nuestra fila 0 es la de arriba
    bytes_imagen = (np.flipud(np.clip(estado["buffer"], 0.0, 1.0)) * 255).astype(np.uint8)
    textura = pyglet.image.ImageData(
        estado["ancho"], estado["alto"], "RGB", bytes_imagen.tobytes()
    ).get_texture()
    # mantenemos el aspecto pixelado al escalar (sin interpolacion)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    estado["textura"] = textura
