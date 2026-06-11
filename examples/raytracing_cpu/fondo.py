"""Quad de fondo que reproduce el cielo del trazador en la rasterizacion.

Reconstruye la direccion del rayo por pixel con la base de la camara (la misma
formula de `Camara.rayo`) y le aplica el gradiente segun la altura. Asi el fondo
del modo rasterizado queda identico al del modo ray tracing, y al alternar con la
tecla `T` el cielo no salta.
"""

import os
from pathlib import Path

import numpy as np
import pyglet
import pyglet.gl as GL

from grafica.utils import load_pipeline


class FondoCielo:
    def __init__(self, camara, color_cielo_arriba, color_cielo_abajo):
        self.pipeline = load_pipeline(
            Path(os.path.dirname(__file__)) / "fondo_vertex_program.glsl",
            Path(os.path.dirname(__file__)) / "fondo_fragment_program.glsl",
        )

        self.pipeline["forward"] = camara.forward.astype(np.float32)
        self.pipeline["right"] = camara.right.astype(np.float32)
        self.pipeline["up_vector"] = camara.up.astype(np.float32)
        self.pipeline["half_width"] = float(camara.half_width)
        self.pipeline["half_height"] = float(camara.half_height)
        self.pipeline["color_cielo_arriba"] = np.asarray(color_cielo_arriba, dtype=np.float32)
        self.pipeline["color_cielo_abajo"] = np.asarray(color_cielo_abajo, dtype=np.float32)

        posiciones = np.array(
            [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32
        )
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        self._gpu = self.pipeline.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
        self._gpu.position[:] = posiciones

    def draw(self):
        # el fondo se dibuja sin escribir profundidad para que quede detras de todo
        profundidad_activa = GL.glIsEnabled(GL.GL_DEPTH_TEST)
        if profundidad_activa:
            GL.glDisable(GL.GL_DEPTH_TEST)

        self.pipeline.use()
        self._gpu.draw(GL.GL_TRIANGLES)
        self.pipeline.stop()

        if profundidad_activa:
            GL.glEnable(GL.GL_DEPTH_TEST)
