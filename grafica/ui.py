"""Utilidades para dibujar elementos 2D de UI encima de una escena 3D.

El problema: en Windows algunos drivers producen z-fighting entre los glifos
de texto de pyglet y la geometría 3D si el depth test o la escritura al depth
buffer siguen activos al llamar a `label.draw()`. Este módulo ofrece un
context manager que fija el estado correcto y lo restaura al salir.
"""

from contextlib import contextmanager

import pyglet.gl as GL


@contextmanager
def ui_overlay():
    """Configura estado de OpenGL para dibujar UI 2D por encima de la escena.

    Desactiva depth test y escritura al depth buffer, activa blending con
    alpha premultiplicado por src. Restaura el estado previo de depth test
    y blending al salir del bloque (la máscara de depth se deja en TRUE,
    que es el valor por defecto que asumen todos los demás ejemplos).

    Uso:
        with ui_overlay():
            label.draw()
            batch.draw()
    """
    depth_test_was_enabled = GL.glIsEnabled(GL.GL_DEPTH_TEST)
    blend_was_enabled = GL.glIsEnabled(GL.GL_BLEND)

    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDepthMask(GL.GL_FALSE)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    try:
        yield
    finally:
        GL.glDepthMask(GL.GL_TRUE)
        if depth_test_was_enabled:
            GL.glEnable(GL.GL_DEPTH_TEST)
        if not blend_was_enabled:
            GL.glDisable(GL.GL_BLEND)
