"""Visor de un volumen de microscopia de fluorescencia: dos canales y varios tiempos.

Ejemplo de la unidad de rendering de volumen que acompana a raycasting_volumen. El
recorrido del rayo es el mismo (un rayo por pixel, la caja por el metodo de las
losas, muestras a paso fijo y composicion front-to-back); cambian los datos:

- No hay una escala fija como las unidades Hounsfield. Cada canal tiene su propia
  ventana, que sale de los datos (ver datos.ventana_por_canal).
- Hay dos canales en una misma textura 3D (GL_RG8), cada uno con su color. En el
  MIP se toma el maximo de cada canal por separado y los colores se suman.
- La funcion de transferencia de la composicion es una rampa: la fluorescencia es
  luz emitida, asi que la opacidad crece con el brillo y el color es el del canal.
- El embrion es una esfera, asi que se puede desplegar como un planisferio (modo
  mapa): la parametrizacion de una esfera con longitud y latitud, con la
  distorsion de los polos a la vista.
- Hay varios tiempos. Cada uno se sube como una textura 3D distinta, y al cambiar
  de tiempo solo cambia la textura enlazada.

Lee un hyperstack de ImageJ (TIFF de 8 bits con ejes TZCYX y dos canales), que no
viene con el repositorio. La primera ejecucion lo reduce y guarda el resultado en
cache/microscopia_volumen; las siguientes parten de inmediato.

Controles:
- arrastra:          orbita la camara; rueda: acerca o aleja.
- M:                 cicla el modo (MIP por canal / composicion / mapa).
- L:                 alterna la iluminacion de Phong con el gradiente (composicion).
- P:                 alterna el MIP con profundidad (MIP).
- 1 / 2:             muestra u oculta cada canal.
- izquierda/derecha: tiempo anterior / siguiente; espacio: reproduce en bucle.
- - / +:             baja / sube el brillo (angosta la ventana de cada canal).
- U:                 alterna el umbral inferior de la ventana (sin el, el ruido de
                     fondo se acumula en cada rayo como una niebla).
- [ / ]:             baja / sube la densidad (solo en la composicion).
- , / .:             baja / sube el paso de muestreo k.
- H:                 oculta o muestra el panel de texto.
- R:                 reinicia la camara y las perillas.
- ESC:               cierra la ventana.
"""

import copy
import ctypes
from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL

from grafica.ui import InfoPanel
from grafica.utils import load_pipeline

from .datos import ajustar_esfera, cargar_hiperstack, ventana_por_canal

MODOS = ["MIP por canal", "composicion", "mapa del embrion"]
PASOS_K = [0.25, 0.5, 1.0, 2.0]
COLORES_CANAL = [(0.15, 1.00, 0.30), (1.00, 0.20, 0.85)]  # verde y magenta
PERIODO_REPRODUCCION = 0.6  # segundos por tiempo al reproducir.

DEFAULTS = {
    "modo": 0,
    "k_index": 1,             # k = 0.5 por defecto.
    "canales": [True, True],
    "brillo": 1.0,
    "densidad": 0.1,
    "tiempo": 0,
    "reproducir": False,
    "iluminacion": False,
    "profundidad": False,
    "panel": True,
    "umbral": True,
    "yaw": 0.6,
    "pitch": 0.35,
    "distancia": 3.0,
}


def _subir_tiempo(volumen):
    """Sube un tiempo (Z, Y, X, 2) uint8 a una textura 3D GL_RG8 y devuelve su id."""
    profundidad, alto, ancho, _ = volumen.shape
    texture_id = GL.GLuint(0)
    GL.glGenTextures(1, ctypes.byref(texture_id))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, texture_id)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage3D(
        GL.GL_TEXTURE_3D, 0, GL.GL_RG8,
        ancho, alto, profundidad,
        0, GL.GL_RG, GL.GL_UNSIGNED_BYTE,
        volumen.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    for parametro in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
        GL.glTexParameteri(GL.GL_TEXTURE_3D, parametro, GL.GL_LINEAR)
    for eje in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R):
        GL.glTexParameteri(GL.GL_TEXTURE_3D, eje, GL.GL_CLAMP_TO_EDGE)
    return texture_id


@click.command("microscopia_volumen",
               short_help="Volumen de microscopia de fluorescencia: dos canales y varios tiempos")
@click.argument("ruta", type=click.Path(exists=True, dir_okay=False))
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=800)
@click.option("--reduccion", type=int, default=2,
              help="Factor por el que se reduce cada eje al cargar el volumen.")
@click.option("--escala-z", type=float, default=1.0,
              help="Largo de un voxel en z dividido por su largo en x e y.")
def microscopia_volumen(ruta, width, height, reduccion, escala_z):
    tiempos = cargar_hiperstack(ruta, reduccion)
    profundidad, alto, ancho, _ = tiempos[0].shape

    # La caja conserva las proporciones del volumen: el eje mas largo mide 2.
    extension = np.array([ancho, alto, profundidad * escala_z], dtype=np.float32)
    box_half = extension / extension.max()
    voxel_size = float((2.0 * box_half / np.array([ancho, alto, profundidad])).min())
    minimos, maximos = ventana_por_canal(tiempos[0])
    print(f"[microscopia_volumen] {len(tiempos)} tiempos de {tiempos[0].shape} (ZYXC); "
          f"ventana del canal 0 de {minimos[0] * 255:.0f} a {maximos[0] * 255:.0f}, "
          f"del canal 1 de {minimos[1] * 255:.0f} a {maximos[1] * 255:.0f}")

    centro, radio = ajustar_esfera(tiempos[0], minimos, maximos, box_half)
    print(f"[microscopia_volumen] esfera del embrion: centro {np.round(centro, 3)}, radio {radio:.3f}")

    window = pyglet.window.Window(width, height, caption="Microscopia de fluorescencia")
    state = copy.deepcopy(DEFAULTS)

    texturas = [_subir_tiempo(volumen) for volumen in tiempos]

    carpeta = Path(__file__).parent
    raycast = load_pipeline(carpeta / "vertex_program.glsl", carpeta / "fragment_program.glsl")
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    screen_quad = raycast.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    screen_quad.position[:] = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)

    raycast.use()
    raycast["volume"] = 0
    raycast["box_half"] = tuple(float(v) for v in box_half)
    raycast["voxel_size"] = voxel_size
    raycast["color_canal_0"] = COLORES_CANAL[0]
    raycast["color_canal_1"] = COLORES_CANAL[1]
    raycast["centro_esfera"] = tuple(float(v) for v in centro)
    raycast["radio_esfera"] = radio

    panel = (
        InfoPanel(x=14, y_top=height - 22, color=(230, 230, 230, 255),
                  background=(20, 22, 30), background_width=470)
        .add("fuente", size=14)
        .add("tiempo")
        .add("modo")
        .add("iluminacion")
        .add("canales")
        .add("brillo")
        .add("ventana")
        .add("densidad")
        .add("paso")
        .footer("arrastra orbita, rueda acerca   M L P 1 2 <- -> espacio - + U [ ] , . H R")
    )

    def apply_state():
        """Propaga el estado a los uniforms del ray casting y al panel."""
        k = PASOS_K[state["k_index"]]
        raycast.use()
        raycast["mode"] = state["modo"]
        raycast["camera_yaw"] = state["yaw"]
        raycast["camera_pitch"] = state["pitch"]
        raycast["distancia_camara"] = state["distancia"]
        raycast["step_size"] = k * voxel_size
        inferiores = minimos if state["umbral"] else (0.0, 0.0)
        superiores = tuple(minimo + (maximo - minimo) / state["brillo"]
                           for minimo, maximo in zip(inferiores, maximos))
        raycast["minimo_canal"] = inferiores
        raycast["maximo_canal"] = superiores
        raycast["canal_visible"] = tuple(1.0 if visible else 0.0 for visible in state["canales"])
        raycast["densidad"] = state["densidad"]
        raycast["iluminacion"] = int(state["iluminacion"])
        raycast["profundidad_mip"] = int(state["profundidad"])

        nombres = ["verde", "magenta"]
        visibles = [nombre for nombre, visible in zip(nombres, state["canales"]) if visible]
        panel["fuente"] = f"fuente: {Path(ruta).name}"
        panel["tiempo"] = (f"tiempo: {state['tiempo'] + 1} de {len(tiempos)}"
                           + ("  (reproduciendo)" if state["reproducir"] else ""))
        con_profundidad = state["modo"] == 0 and state["profundidad"]
        panel["modo"] = f"modo: {MODOS[state['modo']]}" + (" con profundidad" if con_profundidad else "")
        panel["iluminacion"] = (f"iluminacion de Phong: {'activa' if state['iluminacion'] else 'inactiva'}"
                                + ("" if state["modo"] == 1 else " (solo composicion)"))
        panel["canales"] = f"canales: {', '.join(visibles) if visibles else 'ninguno'}"
        panel["brillo"] = f"brillo: x{state['brillo']:.2f}"
        panel["ventana"] = ("ventana: " + ", ".join(
            f"{nombre} de {inferior * 255:.0f} a {superior * 255:.0f}"
            for nombre, inferior, superior in zip(nombres, inferiores, superiores))
            + ("" if state["umbral"] else " (sin umbral)"))
        panel["densidad"] = f"densidad: {state['densidad']:.3f} por voxel"
        panel["paso"] = f"paso k: {k:.2f} (del tamano de un voxel)"

    @window.event
    def on_draw():
        window.clear()
        ancho_buffer, alto_buffer = window.get_framebuffer_size()
        GL.glViewport(0, 0, ancho_buffer, alto_buffer)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, texturas[state["tiempo"]])
        raycast.use()
        raycast["viewport_size"] = (float(ancho_buffer), float(alto_buffer))
        screen_quad.draw(GL.GL_TRIANGLES)
        if state["panel"]:
            panel.draw()

    def reproducir(dt):
        if state["reproducir"]:
            state["tiempo"] = (state["tiempo"] + 1) % len(tiempos)
            apply_state()

    pyglet.clock.schedule_interval(reproducir, PERIODO_REPRODUCCION)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        state["yaw"] += dx * 0.01
        state["pitch"] = float(np.clip(state["pitch"] - dy * 0.01, -1.4, 1.4))
        apply_state()

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        state["distancia"] = float(np.clip(state["distancia"] - 0.15 * scroll_y, 1.2, 6.0))
        apply_state()

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.M:
            state["modo"] = (state["modo"] + 1) % len(MODOS)
        elif symbol == key.H:
            state["panel"] = not state["panel"]
        elif symbol == key.U:
            state["umbral"] = not state["umbral"]
        elif symbol == key.L:
            state["iluminacion"] = not state["iluminacion"]
        elif symbol == key.P:
            state["profundidad"] = not state["profundidad"]
        elif symbol in (key._1, key._2):
            canal = 0 if symbol == key._1 else 1
            state["canales"][canal] = not state["canales"][canal]
        elif symbol == key.LEFT:
            state["tiempo"] = (state["tiempo"] - 1) % len(tiempos)
        elif symbol == key.RIGHT:
            state["tiempo"] = (state["tiempo"] + 1) % len(tiempos)
        elif symbol == key.SPACE:
            state["reproducir"] = not state["reproducir"]
        elif symbol == key.MINUS:
            state["brillo"] = max(0.25, state["brillo"] / 1.25)
        elif symbol in (key.PLUS, key.EQUAL):
            state["brillo"] = min(8.0, state["brillo"] * 1.25)
        elif symbol == key.BRACKETLEFT:
            state["densidad"] = max(0.005, state["densidad"] / 1.5)
        elif symbol == key.BRACKETRIGHT:
            state["densidad"] = min(1.0, state["densidad"] * 1.5)
        elif symbol == key.COMMA:
            state["k_index"] = max(0, state["k_index"] - 1)
        elif symbol == key.PERIOD:
            state["k_index"] = min(len(PASOS_K) - 1, state["k_index"] + 1)
        elif symbol == key.R:
            state.update(copy.deepcopy(DEFAULTS))
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    microscopia_volumen()
