"""Puertas de Doom, que se abren al subir el techo de su sector.

Una puerta cerrada es un sector con el techo a la altura del piso. Al
usar la línea que la separa del jugador, el techo sube hasta 4 unidades bajo el
más bajo de los techos vecinos; según el tipo de la línea, la puerta queda
abierta o espera unos cuatro segundos y vuelve a bajar. Como las líneas no se
mueven, los árboles no cambian. El BSP de Doom viene calculado en el WAD, y por
eso sus puertas y ascensores suben y bajan, pero ninguna pared se mueve de lado.
"""
import numpy as np

from .arboles import impactos
from .primera_persona import ALTURA_JUGADOR

ALCANCE = 64  # la distancia máxima a la línea que se usa
ESPERA = 150 / 35  # segundos que espera abierta: 150 tics, a 35 tics por segundo

# tipos de línea que abren la puerta de atrás al usarlos desde el frente, con su
# velocidad en unidades por segundo (2 u 8 por tic) y si la puerta vuelve a
# cerrarse; las de llave (26 a 28 y 32 a 34) se abren sin ella, porque el
# ejemplo no tiene objetos que recoger
PUERTAS = {
    1: (70, True), 26: (70, True), 27: (70, True), 28: (70, True), 117: (280, True),
    31: (70, False), 32: (70, False), 33: (70, False), 34: (70, False), 118: (280, False),
}


def usar(mapa, arbol, solida, puertas, posicion, angulo):
    """Usa la línea que tiene enfrente el jugador, como la tecla de usar de Doom.

    Recorre las líneas que cruza un rayo de 64 unidades hasta la primera con una
    acción; antes de llegar a ella, una línea sin hueco entre el piso y el techo
    detiene la búsqueda. Devuelve el sector de la puerta, o -1 si no usó ninguna.
    """
    direccion = np.array([np.cos(angulo), np.sin(angulo)])
    for t, pared in impactos(arbol, posicion, direccion, solida, {"nodos": 0, "paredes": 0}):
        linea = arbol["linea"][pared]
        sectores = [mapa["sector_derecho"][linea], mapa["sector_izquierdo"][linea]]
        if t > ALCANCE:
            return -1
        if mapa["especial"][linea]:
            # la línea se usa desde su frente, y la puerta es el sector de atrás
            a, b = arbol["a"][pared], arbol["b"][pared]
            de_frente = (posicion - a) @ np.array([b[1] - a[1], a[0] - b[0]]) > 0
            if mapa["especial"][linea] not in PUERTAS or not de_frente or sectores[1] < 0:
                return -1
            _accionar(mapa, puertas, int(sectores[1]), *PUERTAS[mapa["especial"][linea]])
            return int(sectores[1])
        if min(sectores) < 0 or mapa["techo"][sectores].min() <= mapa["piso"][sectores].max():
            return -1
    return -1


def _accionar(mapa, puertas, sector, velocidad, se_cierra):
    """Pone en marcha la puerta de un sector. Si ya está en marcha y es de las
    que se cierran solas, cambia de sentido: si baja, vuelve a subir, y si sube
    o espera abierta, baja."""
    if sector in puertas:
        if se_cierra:
            puertas[sector]["direccion"] = 1 if puertas[sector]["direccion"] == -1 else -1
        return
    derecho, izquierdo = mapa["sector_derecho"], mapa["sector_izquierdo"]
    vecinos = np.concatenate([izquierdo[(derecho == sector) & (izquierdo >= 0)], derecho[izquierdo == sector]])
    puertas[sector] = {"direccion": 1, "velocidad": velocidad, "se_cierra": se_cierra,
                       "tope": mapa["techo"][vecinos].min() - 4, "espera": 0.0}


def mover(mapa, puertas, dt, sector_jugador):
    """Avanza `dt` segundos las puertas en marcha, que cambian el techo de su sector.

    Una puerta que baja hasta la cabeza del jugador vuelve a subir, como en Doom.
    """
    for sector, puerta in list(puertas.items()):
        if puerta["direccion"] == 0:
            puerta["espera"] -= dt
            if puerta["espera"] <= 0:
                puerta["direccion"] = -1
            continue
        piso = mapa["piso"][sector]
        destino = puerta["tope"] if puerta["direccion"] == 1 else piso
        techo = mapa["techo"][sector] + puerta["direccion"] * puerta["velocidad"] * dt
        if puerta["direccion"] == -1 and sector == sector_jugador and techo < piso + ALTURA_JUGADOR:
            puerta["direccion"] = 1
        elif (techo - destino) * puerta["direccion"] < 0:
            mapa["techo"][sector] = techo
        else:
            # llegó: la que se cierra sola espera abierta, y las demás terminan
            mapa["techo"][sector] = destino
            if puerta["direccion"] == 1 and puerta["se_cierra"]:
                puerta["direccion"], puerta["espera"] = 0, ESPERA
            else:
                del puertas[sector]
