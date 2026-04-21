"""Fondo de pantalla con degradado vertical, reutilizable entre ejemplos.

Uso típico dentro de `on_draw`:

    background = GradientBackground()
    ...
    def on_draw():
        window.clear()
        GL.glViewport(0, 0, width, height)
        background.draw()
        GL.glEnable(GL.GL_DEPTH_TEST)
        # ... resto de la escena
"""

import numpy as np
import pyglet
import pyglet.gl as GL


_VERTEX_SOURCE = """#version 330
in vec2 position;
in vec3 vertex_color;

out vec3 frag_color;

void main() {
    frag_color = vertex_color;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


_FRAGMENT_SOURCE = """#version 330
in vec3 frag_color;
out vec4 out_color;

void main() {
    out_color = vec4(frag_color, 1.0);
}
"""


# Paleta inspirada en el fondo de `examples/camera_frustum`
DEFAULT_BOTTOM_COLOR = (0.06, 0.08, 0.25)  # azul marino
DEFAULT_TOP_COLOR = (0.26, 0.08, 0.38)     # violeta


class GradientBackground:
    """Cuadrilátero a pantalla completa con un degradado vertical por vértice.

    Desactiva depth test mientras dibuja (restaura el estado previo al salir)
    y asume que el viewport ya cubre la región a pintar.
    """

    def __init__(
        self,
        bottom_color=DEFAULT_BOTTOM_COLOR,
        top_color=DEFAULT_TOP_COLOR,
    ):
        vertex_shader = pyglet.graphics.shader.Shader(_VERTEX_SOURCE, "vertex")
        fragment_shader = pyglet.graphics.shader.Shader(_FRAGMENT_SOURCE, "fragment")
        self.pipeline = pyglet.graphics.shader.ShaderProgram(
            vertex_shader, fragment_shader
        )

        positions = np.array(
            [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            dtype=np.float32,
        )
        vertex_colors = np.array(
            [
                *bottom_color,  # inf izq
                *bottom_color,  # inf der
                *top_color,     # sup der
                *top_color,     # sup izq
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self._gpu = self.pipeline.vertex_list_indexed(
            4, GL.GL_TRIANGLES, indices
        )
        self._gpu.position[:] = positions
        self._gpu.vertex_color[:] = vertex_colors

    def draw(self):
        depth_was_enabled = GL.glIsEnabled(GL.GL_DEPTH_TEST)
        if depth_was_enabled:
            GL.glDisable(GL.GL_DEPTH_TEST)

        self.pipeline.use()
        self._gpu.draw(GL.GL_TRIANGLES)
        self.pipeline.stop()

        if depth_was_enabled:
            GL.glEnable(GL.GL_DEPTH_TEST)
