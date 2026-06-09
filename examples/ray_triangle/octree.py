"""Octree de triángulos para acelerar consultas de rayos sobre una malla.

La idea: en vez de probar el rayo contra todos los triángulos de la malla,
se prueba primero contra cajas (AABB) que subdividen el espacio en ocho
octantes recursivos. Si el rayo no toca una caja, ninguno de sus triángulos
puede intersectarlo y se descarta la rama completa. Solo los triángulos de
las hojas que el rayo atraviesa pasan a la prueba exacta (Möller-Trumbore).

Decisiones de construcción:

- Un nodo se subdivide si tiene más triángulos que la capacidad y no alcanzó
  el nivel máximo. Los octantes sin triángulos no generan nodo, así las
  cajas abrazan la forma de la malla.
- Cada triángulo se asigna a todos los octantes cuyo AABB se solapa con el
  AABB del triángulo. Un triángulo que cruza un plano de corte vive en
  varias hojas; la consulta elimina los duplicados al final. Asignarlo solo
  a un octante (por ejemplo, por centroide) perdería intersecciones.

Este módulo no depende de OpenGL: se puede probar headless.
"""

import numpy as np


def interseccion_rayo_aabb(origen, direccion, minimo, maximo):
    """Prueba rayo-AABB por el método de los slabs.

    La caja es la intersección de tres pares de semiespacios paralelos
    (slabs). Se calcula el intervalo de t en que el rayo está dentro de cada
    slab; el rayo toca la caja si la intersección de los tres intervalos es
    no vacía y termina en t >= 0.

    Devuelve (hay_interseccion, t_entrada).
    """
    # las divisiones por componentes nulas dan +-inf, que el min/max maneja
    # bien: el rayo paralelo a un slab está dentro o fuera para todo t.
    with np.errstate(divide="ignore"):
        inverso_direccion = 1.0 / direccion
    t_caras_cercanas = (minimo - origen) * inverso_direccion
    t_caras_lejanas = (maximo - origen) * inverso_direccion

    t_entrada = np.minimum(t_caras_cercanas, t_caras_lejanas).max()
    t_salida = np.maximum(t_caras_cercanas, t_caras_lejanas).min()

    return t_salida >= max(t_entrada, 0.0), t_entrada


class NodoOctree:
    """Un nodo del octree: su caja, sus hijos y (si es hoja) sus triángulos."""

    __slots__ = ("minimo", "maximo", "nivel", "hijos", "triangulos")

    def __init__(self, minimo, maximo, nivel):
        self.minimo = minimo
        self.maximo = maximo
        self.nivel = nivel
        self.hijos = []
        self.triangulos = None  # índices de caras; solo en las hojas

    @property
    def es_hoja(self):
        return not self.hijos


class Octree:
    """Octree construido sobre los triángulos de una malla.

    Parámetros:
    posiciones -- arreglo (n_vertices, 3) con las posiciones.
    caras -- arreglo (n_caras, 3) con índices de vértices.
    capacidad -- máximo de triángulos por hoja antes de subdividir.
    nivel_maximo -- profundidad máxima de subdivisión.
    """

    def __init__(self, posiciones, caras, capacidad=64, nivel_maximo=6):
        self.capacidad = capacidad
        self.nivel_maximo = nivel_maximo

        # AABB de cada triángulo, precalculado una sola vez: (n_caras, 3).
        triangulos = posiciones[caras]
        self.triangulo_minimo = triangulos.min(axis=1)
        self.triangulo_maximo = triangulos.max(axis=1)

        self.nodos_por_nivel = {}  # nivel -> lista de nodos, para visualizar
        self.cantidad_hojas = 0

        # un margen pequeño evita que triángulos exactamente en el borde
        # queden fuera por error de redondeo.
        margen = 1e-6
        self.raiz = self._construir(
            posiciones.min(axis=0) - margen,
            posiciones.max(axis=0) + margen,
            np.arange(len(caras)),
            nivel=0,
        )

    def _construir(self, minimo, maximo, indices, nivel):
        nodo = NodoOctree(minimo, maximo, nivel)
        self.nodos_por_nivel.setdefault(nivel, []).append(nodo)

        if len(indices) <= self.capacidad or nivel == self.nivel_maximo:
            nodo.triangulos = indices
            self.cantidad_hojas += 1
            return nodo

        centro = (minimo + maximo) / 2.0
        # los ocho octantes: cada eje elige la mitad inferior o superior.
        for octante in range(8):
            seleccion = np.array(
                [octante & 1, (octante >> 1) & 1, (octante >> 2) & 1]
            )
            hijo_minimo = np.where(seleccion, centro, minimo)
            hijo_maximo = np.where(seleccion, maximo, centro)

            # triángulos cuyo AABB se solapa con el del octante (puede haber
            # triángulos repetidos entre octantes hermanos).
            solapa = (
                (self.triangulo_minimo[indices] <= hijo_maximo).all(axis=1)
                & (self.triangulo_maximo[indices] >= hijo_minimo).all(axis=1)
            )
            if solapa.any():
                nodo.hijos.append(
                    self._construir(hijo_minimo, hijo_maximo, indices[solapa], nivel + 1)
                )
        return nodo

    def consultar_rayo(self, origen, direccion):
        """Triángulos candidatos para un rayo, descartando ramas por AABB.

        Devuelve (candidatos, nodos_visitados, cajas_probadas):
        candidatos -- índices únicos de caras en las hojas que el rayo toca.
        nodos_visitados -- nodos cuya caja el rayo SÍ atraviesa (para dibujar).
        cajas_probadas -- cuántas pruebas rayo-AABB se hicieron.
        """
        candidatos = []
        nodos_visitados = []
        cajas_probadas = 0

        pendientes = [self.raiz]
        while pendientes:
            nodo = pendientes.pop()
            cajas_probadas += 1
            hay_interseccion, _ = interseccion_rayo_aabb(
                origen, direccion, nodo.minimo, nodo.maximo
            )
            if not hay_interseccion:
                continue
            nodos_visitados.append(nodo)
            if nodo.es_hoja:
                candidatos.append(nodo.triangulos)
            else:
                pendientes.extend(nodo.hijos)

        if candidatos:
            # un triángulo puede vivir en varias hojas: deduplicar.
            candidatos = np.unique(np.concatenate(candidatos))
        else:
            candidatos = np.array([], dtype=int)
        return candidatos, nodos_visitados, cajas_probadas
