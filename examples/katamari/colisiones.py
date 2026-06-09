"""Detección de colisiones del katamari: fase ancha y fase angosta.

La fase ancha usa una grilla de hash espacial sobre el plano XZ: cada objeto
se registra en todas las celdas que toca el AABB de su esfera contenedora.
Una consulta solo revisa las celdas que toca el AABB de la esfera consultante,
así que dos esferas que se intersectan siempre comparten al menos una celda.

La fase angosta es la prueba más simple posible: dos esferas se intersectan
si la distancia entre sus centros es menor que la suma de sus radios. Se
compara al cuadrado para evitar la raíz cuadrada.

Este módulo no depende de OpenGL: se puede probar headless.
"""

import math

import numpy as np


def esferas_se_intersectan(centro_a, radio_a, centro_b, radio_b):
    """Prueba esfera-esfera: |centro_a - centro_b| <= radio_a + radio_b."""
    diferencia = np.asarray(centro_a) - np.asarray(centro_b)
    distancia_cuadrada = float(np.dot(diferencia, diferencia))
    suma_radios = radio_a + radio_b
    return distancia_cuadrada <= suma_radios * suma_radios


class GrillaHashEspacial:
    """Fase ancha: diccionario de celda (i, j) a conjunto de nombres.

    La celda de un punto se obtiene dividiendo sus coordenadas por el lado
    de la celda y truncando hacia abajo. Insertar y consultar recorren el
    rango de celdas que cubre el AABB de la esfera (centro más menos radio),
    por lo que un objeto grande puede vivir en varias celdas a la vez.
    """

    def __init__(self, lado_celda):
        self.lado_celda = lado_celda
        self.celdas = {}

    def _rango_de_celdas(self, centro_xz, radio):
        """Celdas (i, j) que toca el AABB de la esfera proyectada en XZ."""
        x, z = centro_xz
        i_min = math.floor((x - radio) / self.lado_celda)
        i_max = math.floor((x + radio) / self.lado_celda)
        j_min = math.floor((z - radio) / self.lado_celda)
        j_max = math.floor((z + radio) / self.lado_celda)
        return [
            (i, j)
            for i in range(i_min, i_max + 1)
            for j in range(j_min, j_max + 1)
        ]

    def insertar(self, nombre, centro_xz, radio):
        for celda in self._rango_de_celdas(centro_xz, radio):
            self.celdas.setdefault(celda, set()).add(nombre)

    def remover(self, nombre, centro_xz, radio):
        for celda in self._rango_de_celdas(centro_xz, radio):
            if celda in self.celdas:
                self.celdas[celda].discard(nombre)
                if not self.celdas[celda]:
                    del self.celdas[celda]

    def consultar(self, centro_xz, radio):
        """Candidatos de la fase ancha y celdas revisadas.

        Devuelve (candidatos, celdas_consultadas). Los candidatos son los
        nombres registrados en las celdas que toca la esfera consultante;
        la fase angosta decide cuáles colisionan de verdad.
        """
        celdas_consultadas = self._rango_de_celdas(centro_xz, radio)
        candidatos = set()
        for celda in celdas_consultadas:
            candidatos |= self.celdas.get(celda, set())
        return candidatos, celdas_consultadas
