"""Transición por omisión: el último cuadro gira y se encoge sobre un fondo de franjas.

El anfitrión copia el último cuadro de la escena que termina a una textura y
llama a dibujar() con un progreso que va de 0 a 1. La textura va sobre un
rectángulo que el vertex program rota y escala (unidades 08 y 10); detrás, el
fragment program dibuja franjas que giran en torno al centro.
"""

from pathlib import Path

import pyglet.gl as GL

from grafica.utils import load_pipeline

ROL = "transicion"

CARPETA = Path(__file__).parent
VUELTA = 3.0  # radianes que gira el cuadro durante la transición


def preparar(ventana):
    pipeline = load_pipeline(CARPETA / "vertex_program.glsl", CARPETA / "fragment_program.glsl")
    rectangulo = pipeline.vertex_list(4, GL.GL_TRIANGLE_STRIP, position=("f", (-1, -1, 1, -1, -1, 1, 1, 1)))
    return {"pipeline": pipeline, "rectangulo": rectangulo, "razon": ventana.width / ventana.height}


def dibujar(recursos, textura, progreso):
    pipeline, rectangulo = recursos["pipeline"], recursos["rectangulo"]
    # empieza lento y termina rápido
    avance = progreso * progreso

    pipeline.use()
    pipeline["razon"] = recursos["razon"]
    pipeline["progreso"] = progreso

    # primero el fondo, a pantalla completa
    pipeline["fondo"] = True
    pipeline["escala"] = 1.0
    pipeline["angulo"] = 0.0
    rectangulo.draw(GL.GL_TRIANGLE_STRIP)

    # encima, el último cuadro girando y encogiéndose
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(textura.target, textura.id)
    pipeline["cuadro"] = 0
    pipeline["fondo"] = False
    pipeline["escala"] = 1.0 - avance
    pipeline["angulo"] = VUELTA * avance
    rectangulo.draw(GL.GL_TRIANGLE_STRIP)
