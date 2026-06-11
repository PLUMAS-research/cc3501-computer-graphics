"""Dibujo de las cajas contenedoras (AABB) que usa el trazador como fase ancha.

Las mismas cajas que aceleran el ray tracing (un rayo solo se prueba contra una
malla si cruza su AABB) se pueden ver sobre la escena rasterizada. Asi el alumno
mira la estructura de aceleracion: las cajas que abrazan cada Pokemon y la del
piso. Es un wireframe de lineas con la misma camara que el resto.
"""

import os
from pathlib import Path

import numpy as np
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.utils import load_pipeline


# las 12 aristas de un cubo como pares de indices de sus 8 esquinas
_ARISTAS = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # cara inferior
    (4, 5), (5, 6), (6, 7), (7, 4),  # cara superior
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticales
]


def _aristas_caja(caja_min, caja_max):
    """Devuelve los 24 vertices (12 aristas) del AABB como lineas."""
    x0, y0, z0 = caja_min
    x1, y1, z1 = caja_max
    esquinas = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        dtype=np.float32,
    )
    return np.array([esquinas[i] for arista in _ARISTAS for i in arista], dtype=np.float32)


class CajasAABB:
    """Wireframe de las cajas contenedoras de cada objeto, en coordenadas de mundo."""

    def __init__(self, geometrias, color=(1.0, 0.85, 0.2)):
        self.pipeline = load_pipeline(
            Path(os.path.dirname(__file__)) / ".." / "scene_graphs" / "mesh_vertex_program.glsl",
            Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl",
        )
        self.color = np.asarray(color, dtype=np.float32)

        vertices = np.concatenate(
            [_aristas_caja(geo.caja_min, geo.caja_max) for geo in geometrias]
        )
        self._gpu = self.pipeline.vertex_list(len(vertices), GL.GL_LINES)
        self._gpu.position[:] = vertices.flatten()

    def draw(self, view, projection):
        # las cajas se dibujan siempre visibles (sin depth) para que se vean
        # sobre la escena, tanto en la vista rasterizada como sobre el render RT
        profundidad_activa = GL.glIsEnabled(GL.GL_DEPTH_TEST)
        if profundidad_activa:
            GL.glDisable(GL.GL_DEPTH_TEST)

        self.pipeline.use()
        self.pipeline["transform"] = tr.identity().reshape(16, 1, order="F")
        self.pipeline["view"] = view.reshape(16, 1, order="F")
        self.pipeline["projection"] = projection.reshape(16, 1, order="F")
        self.pipeline["color"] = self.color
        self._gpu.draw(GL.GL_LINES)
        self.pipeline.stop()

        if profundidad_activa:
            GL.glEnable(GL.GL_DEPTH_TEST)
