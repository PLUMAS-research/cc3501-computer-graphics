"""Vista en primera persona de un mapa de Doom, con un rayo por columna.

Es el ray casting de Wolfenstein 3D con el mapa de Doom: cada columna de la
imagen lanza un rayo horizontal y recorre el árbol para obtener, ordenados por
distancia, los cruces con las líneas del mapa hasta la primera pared sólida.
Cada cruce con una línea de dos lados pasa de un sector a otro; si el piso del
sector siguiente es más alto se ve un escalón, y si su techo es más bajo, un
dintel. Una ventana vertical, que se cierra con cada trozo dibujado, evita
pintar lo que ya quedó tapado, como hace Doom al dibujar de adelante hacia
atrás.
"""
import numpy as np

from .arboles import impactos

ALTURA_OJO = 41  # sobre el piso, en unidades de Doom
ALTURA_PASO = 24  # el escalón más alto que se sube caminando
ALTURA_JUGADOR = 56
RADIO_JUGADOR = 16

COLOR_PARED = np.array([0.66, 0.58, 0.50])
COLOR_ESCALON = np.array([0.56, 0.50, 0.42])
COLOR_DINTEL = np.array([0.55, 0.53, 0.50])
COLOR_PISO = np.array([0.40, 0.36, 0.32])
COLOR_TECHO = np.array([0.33, 0.33, 0.36])
COLOR_CIELO = np.array([0.50, 0.58, 0.72])


def _luz(mapa, sector, distancia):
    """La luz del sector, que se apaga con la distancia como en Doom."""
    return mapa["luz"][sector] * np.minimum(np.maximum(1.25 - distancia / 1600.0, 0.3), 1.0)


def _plano(planos, mapa, sector, cual, ojo, focal, alto):
    """El color de cada fila de la imagen para el piso o el techo de un sector.

    En cada columna, una pared está toda a la misma distancia, pero un plano se
    aleja fila a fila: si se despeja la distancia en la proyección de _fila, la
    fila y muestra el punto del plano que está a (altura - ojo) * focal /
    (horizonte - y). Esa distancia no cambia de una columna a otra, así que las
    filas de cada plano se calculan una vez por cuadro; Doom también guarda la
    distancia de cada fila (cacheddistance en R_MapPlane).
    """
    if (sector, cual) not in planos:
        filas = np.arange(alto) + 0.5
        distancia = (mapa[cual][sector] - ojo) * focal / (alto / 2.0 - filas)
        color = COLOR_PISO if cual == "piso" else COLOR_TECHO
        planos[sector, cual] = color * _luz(mapa, sector, distancia)[:, None, None]
    return planos[sector, cual]


def _lados(mapa, arbol, pared, posicion):
    """El sector del lado del observador y el del otro lado de una pared (-1 si no hay)."""
    a, b = arbol["a"][pared], arbol["b"][pared]
    linea = arbol["linea"][pared]
    derecho, izquierdo = mapa["sector_derecho"][linea], mapa["sector_izquierdo"][linea]
    # el frente de una línea de Doom es su lado derecho
    desde_la_derecha = (posicion - a) @ np.array([b[1] - a[1], a[0] - b[0]]) > 0
    return (derecho, izquierdo) if desde_la_derecha else (izquierdo, derecho)


def dibujar(mapa, arbol, solida, posicion, angulo, ojo, ancho=320, alto=200, columnas=320):
    """Imagen (alto, ancho, 3) con valores en [0, 1] y los conteos del recorrido.

    El campo de visión horizontal es de 90 grados, como en Doom. Con menos
    columnas que el ancho, cada rayo pinta varias columnas seguidas.
    """
    imagen = np.zeros((alto, ancho, 3))
    focal = ancho / 2.0
    paso = ancho // columnas
    cuenta = {"nodos": 0, "paredes": 0}
    planos = {}  # el color de las filas de cada piso y techo visto en este cuadro
    for x in range(0, ancho, paso):
        desvio = np.arctan((ancho / 2.0 - x - paso / 2.0) / focal)
        direccion = np.array([np.cos(angulo + desvio), np.sin(angulo + desvio)])
        cruces = impactos(arbol, posicion, direccion, solida, cuenta)
        _columna(imagen[:, x:x + paso], mapa, arbol, cruces, posicion, np.cos(desvio), ojo, focal, planos)
    return imagen, cuenta


def _fila(altura, ojo, escala, horizonte, arriba, abajo):
    """La fila de la imagen donde se ve una altura, dentro de la ventana [arriba, abajo]."""
    return min(max(int(horizonte - (altura - ojo) * escala + 0.5), arriba), abajo)


def _columna(columna, mapa, arbol, cruces, posicion, coseno, ojo, focal, planos):
    arriba, abajo = 0, len(columna)
    horizonte = len(columna) / 2.0
    for t, pared in cruces:
        # la distancia perpendicular al plano de la imagen evita el ojo de pez
        distancia = max(t * coseno, 1.0)
        escala = focal / distancia
        cerca, lejos = _lados(mapa, arbol, pared, posicion)
        if cerca < 0:
            cerca, lejos = lejos, -1
        luz = _luz(mapa, cerca, distancia)
        a, b = arbol["a"][pared], arbol["b"][pared]
        # contraste falso de Doom: las paredes este-oeste más oscuras, las norte-sur más claras
        contraste = 0.9 if abs(a[1] - b[1]) < 1e-6 else 1.1 if abs(a[0] - b[0]) < 1e-6 else 1.0

        # techo y piso del sector cercano, desde el borde de la ventana hasta esta pared
        techo = _fila(mapa["techo"][cerca], ojo, escala, horizonte, arriba, abajo)
        piso = _fila(mapa["piso"][cerca], ojo, escala, horizonte, techo, abajo)
        if mapa["cielo"][cerca]:
            columna[arriba:techo] = COLOR_CIELO
        else:
            columna[arriba:techo] = _plano(planos, mapa, cerca, "techo", ojo, focal, len(columna))[arriba:techo]
        columna[piso:abajo] = _plano(planos, mapa, cerca, "piso", ojo, focal, len(columna))[piso:abajo]
        arriba, abajo = techo, piso
        if lejos < 0:
            columna[arriba:abajo] = np.clip(COLOR_PARED * luz * contraste, 0, 1)
            return
        # dintel y escalón: lo que cambia entre un sector y el siguiente
        if mapa["techo"][lejos] < mapa["techo"][cerca] and not (mapa["cielo"][cerca] and mapa["cielo"][lejos]):
            dintel = _fila(mapa["techo"][lejos], ojo, escala, horizonte, arriba, abajo)
            columna[arriba:dintel] = np.clip(COLOR_DINTEL * luz * contraste, 0, 1)
            arriba = dintel
        if mapa["piso"][lejos] > mapa["piso"][cerca]:
            escalon = _fila(mapa["piso"][lejos], ojo, escala, horizonte, arriba, abajo)
            columna[escalon:abajo] = np.clip(COLOR_ESCALON * luz * contraste, 0, 1)
            abajo = escalon
        if arriba >= abajo:
            return


def sector_en(mapa, arbol, solida, posicion):
    """El sector que contiene una posición: el del lado de ella en la primera línea que cruza un rayo."""
    cruces = impactos(arbol, posicion, np.array([1.0, 0.0]), solida, {"nodos": 0, "paredes": 0})
    return _lados(mapa, arbol, cruces[0][1], posicion)[0] if cruces else -1


def puede_avanzar(mapa, arbol, solida, posicion, desplazamiento):
    """Si el jugador puede moverse en línea recta: no cruza paredes sólidas, ni
    escalones de más de 24 unidades, ni pasos de menos de 56 de alto."""
    largo = np.linalg.norm(desplazamiento)
    if largo == 0:
        return True
    direccion = desplazamiento / largo
    for t, pared in impactos(arbol, posicion, direccion, solida, {"nodos": 0, "paredes": 0}):
        if t > largo + RADIO_JUGADOR:
            return True
        cerca, lejos = _lados(mapa, arbol, pared, posicion)
        if cerca < 0 or lejos < 0:
            return False
        hueco = min(mapa["techo"][cerca], mapa["techo"][lejos]) - max(mapa["piso"][cerca], mapa["piso"][lejos])
        if mapa["piso"][lejos] - mapa["piso"][cerca] > ALTURA_PASO or hueco < ALTURA_JUGADOR:
            return False
    return True
