"""Simplificacion de mallas por error cuadratico (QEM, Garland y Heckbert 1997).

Reduce el numero de triangulos colapsando aristas, una a la vez, eligiendo
siempre la mas barata. El costo de colapsar una arista mide cuanto se aleja la
malla simplificada de los planos de las caras originales.

La idea: a cada vertice se le asocia una cuadrica Q, una matriz 4x4 que, para
un punto homogeneo v = (x, y, z, 1), entrega la suma de distancias al cuadrado
a los planos de las caras que tocan al vertice como v^T Q v. La cuadrica de una
cara con plano (n, d) es la matriz K = p p^T con p = (n_x, n_y, n_z, d). La
cuadrica de un vertice es la suma de las K de sus caras incidentes.

Para colapsar la arista (a, b) se suma Q_a + Q_b y se elige la posicion del
vertice resultante que minimiza el error. Aqui restringimos esa posicion a los
extremos o al punto medio: evita que el vertice salte lejos y de vuelta caras
(foldover), al precio de un error algo mayor que la solucion optima exacta.

El algoritmo mantiene una cola de prioridad de aristas por costo. Al sacar una
arista se recalcula su costo (las cuadricas vecinas cambiaron desde que entro);
si subio, se reinserta; si no, se colapsa. Tras cada colapso se actualizan la
cuadrica y los costos de las aristas que tocan al vertice sobreviviente.
"""

import heapq

import numpy as np

from grafica.half_edge import BORDE


def cuadricas_iniciales(malla):
    """Cuadrica 4x4 por vertice: suma de las K de sus caras incidentes."""
    Q = np.zeros((len(malla.positions), 4, 4), dtype=np.float64)
    for f in range(len(malla.f_he)):
        if not malla.f_alive[f]:
            continue
        h = malla.f_he[f]
        ia = malla.tail(h)
        ib = malla.head(h)
        ic = malla.head(malla.he_next[h])
        pa, pb, pc = malla.positions[ia], malla.positions[ib], malla.positions[ic]
        normal = np.cross(pb - pa, pc - pa)
        largo = np.linalg.norm(normal)
        if largo < 1e-12:
            continue
        normal /= largo
        plano = np.array([normal[0], normal[1], normal[2], -normal @ pa])
        K = np.outer(plano, plano)
        Q[ia] += K
        Q[ib] += K
        Q[ic] += K
    return Q


def _costo_y_posicion(Qa, Qb, pa, pb):
    """Costo de colapsar la arista y posicion elegida (extremo o punto medio)."""
    Qbar = Qa + Qb
    mejor_costo, mejor_posicion = None, None
    for candidato in (pa, pb, 0.5 * (pa + pb)):
        homogeneo = np.array([candidato[0], candidato[1], candidato[2], 1.0])
        costo = float(homogeneo @ Qbar @ homogeneo)
        if mejor_costo is None or costo < mejor_costo:
            mejor_costo, mejor_posicion = costo, candidato
    return max(mejor_costo, 0.0), mejor_posicion


def simplificar_a_niveles(malla, caras_objetivo, reporte=None):
    """Colapsa aristas y captura la malla cuando cruza cada objetivo de caras.

    `caras_objetivo` es una lista de conteos de caras (de mayor a menor).
    Devuelve una lista de (positions, faces) compactadas, una por objetivo.
    `reporte` es un callback opcional que recibe el numero de caras actual.
    """
    Q = cuadricas_iniciales(malla)
    heap = []

    def encolar(h):
        a, b = malla.tail(h), malla.head(h)
        costo, _ = _costo_y_posicion(Q[a], Q[b], malla.positions[a], malla.positions[b])
        heapq.heappush(heap, (costo, h))

    for h in range(len(malla.he_to)):
        if malla.he_alive[h] and malla.can_collapse(h):
            encolar(h)

    objetivos = sorted(set(caras_objetivo), reverse=True)
    snapshots = []
    indice_objetivo = 0

    def capturar_si_corresponde():
        nonlocal indice_objetivo
        while indice_objetivo < len(objetivos) and malla.n_faces() <= objetivos[indice_objetivo]:
            snapshots.append(malla.to_arrays())
            indice_objetivo += 1

    capturar_si_corresponde()  # captura el nivel de resolucion completa.

    while indice_objetivo < len(objetivos) and heap:
        objetivo = objetivos[indice_objetivo]
        while malla.n_faces() > objetivo and heap:
            costo, h = heapq.heappop(heap)
            if not malla.he_alive[h] or not malla.can_collapse(h):
                continue
            a, b = malla.tail(h), malla.head(h)
            costo_actual, posicion = _costo_y_posicion(
                Q[a], Q[b], malla.positions[a], malla.positions[b]
            )
            if costo_actual > costo + 1e-9:
                # las cuadricas vecinas cambiaron: este costo era viejo, reinsertar.
                heapq.heappush(heap, (costo_actual, h))
                continue
            if malla.collapse(h, posicion):
                Q[b] = Q[a] + Q[b]
                vecinas, _ = malla.outgoing_ring(b)
                for ho in vecinas:
                    if malla.can_collapse(ho):
                        encolar(ho)
                if reporte is not None and malla.n_faces() % 200 == 0:
                    reporte(malla.n_faces())
        capturar_si_corresponde()

    while indice_objetivo < len(objetivos):
        snapshots.append(malla.to_arrays())
        indice_objetivo += 1
    return snapshots
