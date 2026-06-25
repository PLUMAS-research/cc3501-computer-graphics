"""Visor de volumen: ray casting 3D mas tres planos ortogonales (MPR) con recorte.

Ejemplo rico de la unidad de rendering de volumen (el atomico es pyvista_orbital,
la herramienta). Implementa el ray casting de volumen desde cero (un rayo por pixel
atraviesa un volumen de voxeles y las muestras se combinan en un color) y lo
presenta como un visor medico de cuatro vistas:

- Arriba a la izquierda, la vista 3D interactiva (ray casting de la textura 3D).
- Los otros tres cuadrantes son cortes ortogonales (axial, coronal, sagital), cada
  uno en escala de grises con ventana radiologica.

El recorte (clipping) define que se ve: una caja de recorte en [0,1] por eje limita
que muestras entran a la vista 3D. Se ajusta arrastrando los bordes del rectangulo
de recorte en cualquiera de los tres cortes; la zona que queda fuera se atenua en
los cortes. La cruz de cada corte marca donde cruzan los otros dos planos, y
arrastrando dentro del corte se mueven esos planos.

Por defecto el volumen es un fantoma de Shepp-Logan en HU (ver fantoma.py); la
opcion --ct carga una tomografia real (ver ct.py), tambien en HU.

Controles:
- vista 3D: arrastra para orbitar la camara.
- cortes:   arrastra un borde del recorte para recortar ese eje; arrastra al
            centro para mover la cruz (los otros dos cortes); rueda para cambiar
            el corte mostrado.
- M:        cicla modo de composicion 3D (MIP / promedio / compositing).
- I:        alterna interpolacion (vecino mas cercano / trilineal).
- E:        alterna terminacion temprana (solo afecta al compositing).
- , / .:    baja / sube el paso de muestreo k.
- R:        reinicia camara, recorte y cortes.
- ESC:      cierra la ventana.
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

from .fantoma import generar_fantoma

MODOS = ["MIP (maximo)", "promedio", "compositing"]
PASOS_K = [0.1, 0.25, 0.5, 1.0]
VENTANA = (200.0, 1600.0)  # ventana radiologica de los cortes: (nivel, ancho) en HU.

COLOR_RECORTE = (0.95, 0.75, 0.20)  # rectangulo de la caja de recorte.
COLOR_CRUZ = (0.30, 0.70, 0.95)     # cruz que marca los otros dos cortes.
UMBRAL_BORDE = 0.05                 # cercania (en uv) para agarrar un borde de recorte.
GAP_RECORTE = 0.02                  # separacion minima entre los limites de un eje.

# Un panel MPR fija un eje del mundo (fuera) y muestra el plano perpendicular. axis
# es el identificador que espera el shader; plano son los dos ejes en el plano,
# mapeados a (uv.x, uv.y). Convencion de mundo: x = izquierda-derecha,
# y = superior-inferior (arriba = craneo), z = anterior-posterior. Asi cada panel
# queda derecho: el axial mira de arriba (fija y), el coronal de frente (fija z) y
# el sagital de lado (fija x), los tres con superior-inferior en la vertical.
PANELES = [
    {"nombre": "axial",   "axis": 1, "fuera": 1, "plano": (0, 2)},
    {"nombre": "coronal", "axis": 0, "fuera": 2, "plano": (0, 1)},
    {"nombre": "sagital", "axis": 2, "fuera": 0, "plano": (2, 1)},
]

DEFAULTS = {
    "modo": 0,
    "k_index": 2,            # k = 0.5 por defecto.
    "trilineal": True,
    "terminacion": True,
    "yaw": 0.7,
    "pitch": 0.4,
    "clip_min": [0.0, 0.0, 0.0],
    "clip_max": [1.0, 1.0, 1.0],
    "slice": [0.5, 0.5, 0.5],   # posicion del corte por eje del mundo (x, y, z).
}


def _subir_volumen(volumen):
    """Sube el volumen HU a una textura 3D R32F y devuelve su id."""
    datos = np.ascontiguousarray(volumen, dtype=np.float32)
    profundidad, alto, ancho = datos.shape

    texture_id = GL.GLuint(0)
    GL.glGenTextures(1, ctypes.byref(texture_id))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, texture_id)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage3D(
        GL.GL_TEXTURE_3D, 0, GL.GL_R32F,
        ancho, alto, profundidad,
        0, GL.GL_RED, GL.GL_FLOAT,
        datos.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
    return texture_id


def _layout(ancho, alto):
    """Cuatro cuadrantes (x, y, w, h) con origen abajo a la izquierda."""
    medio_x, medio_y = ancho // 2, alto // 2
    return {
        "3d":      (0,       medio_y, medio_x,         alto - medio_y),
        "axial":   (medio_x, medio_y, ancho - medio_x, alto - medio_y),
        "coronal": (0,       0,       medio_x,         medio_y),
        "sagital": (medio_x, 0,       ancho - medio_x, medio_y),
    }


def _quad_scale(plano, rect, box_half):
    """Escala del quad para que el corte conserve sus proporciones en el panel."""
    eje_a, eje_b = plano
    extension_a, extension_b = float(box_half[eje_a]), float(box_half[eje_b])
    _, _, panel_ancho, panel_alto = rect
    if extension_a / extension_b >= panel_ancho / panel_alto:
        return (1.0, (panel_ancho / panel_alto) * (extension_b / extension_a))
    return ((panel_alto / panel_ancho) * (extension_a / extension_b), 1.0)


@click.command("raycasting_volumen", short_help="Visor de volumen: ray casting 3D + cortes MPR con recorte")
@click.option("--width", type=int, default=1000)
@click.option("--height", type=int, default=800)
@click.option("--resolucion", type=int, default=128, help="Lado del fantoma en voxeles.")
@click.option("--ct", type=click.Choice(["full_head"]), default=None,
              help="Carga una tomografia real (se descarga una vez) en vez del fantoma.")
def raycasting_volumen(width, height, resolucion, ct):
    window = pyglet.window.Window(width, height, caption="Visor de volumen (3D + MPR)")

    state = copy.deepcopy(DEFAULTS)
    state["drag"] = None

    if ct is not None:
        from .ct import cargar_ct
        volumen, box_half = cargar_ct(ct)
        fuente = f"CT {ct}"
    else:
        volumen = generar_fantoma(resolucion)
        box_half = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        fuente = "fantoma Shepp-Logan"
    print(f"[raycasting_volumen] {fuente} {volumen.shape}, "
          f"HU en [{volumen.min():.0f}, {volumen.max():.0f}]")

    # paso de muestreo de referencia = el voxel mas chico en unidades de mundo.
    dimensiones_xyz = np.array(volumen.shape)[::-1]  # (D,H,W) -> (x,y,z)
    voxel_size = float((2.0 * box_half / dimensiones_xyz).min())

    texture_id = _subir_volumen(volumen)

    carpeta = Path(__file__).parent
    raycast = load_pipeline(carpeta / "vertex_program.glsl", carpeta / "fragment_program.glsl")
    slice_pipe = load_pipeline(carpeta / "slice_vertex_program.glsl",
                               carpeta / "slice_fragment_program.glsl")
    line_pipe = load_pipeline(carpeta / "line_vertex_program.glsl",
                              carpeta / "line_fragment_program.glsl")

    # quad de pantalla completa para el ray casting (solo posicion en NDC).
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    raycast_quad = raycast.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    raycast_quad.position[:] = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)

    # quad de los cortes (posicion + uv).
    slice_quad = slice_pipe.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    slice_quad.position[:] = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    slice_quad.uv[:] = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32)

    # overlays de cada panel: rectangulo de recorte (4 segmentos) y cruz (2 segmentos).
    lineas_recorte = {p["nombre"]: line_pipe.vertex_list(8, GL.GL_LINES) for p in PANELES}
    lineas_cruz = {p["nombre"]: line_pipe.vertex_list(4, GL.GL_LINES) for p in PANELES}

    raycast.use()
    raycast["voxel_size"] = voxel_size
    raycast["box_half"] = tuple(float(v) for v in box_half)
    raycast["volume"] = 0
    slice_pipe.use()
    slice_pipe["volume"] = 0
    slice_pipe["window_level"] = VENTANA

    layout = _layout(width, height)

    panel = (
        InfoPanel(x=14, y_top=height - 22, color=(230, 230, 230, 255),
                  background=(20, 22, 30), background_width=520)
        .add("fuente", size=14)
        .add("modo")
        .add("interpolacion")
        .add("paso")
        .add("terminacion")
        .add("recorte")
        .footer("3D arrastra orbita   panel arrastra recorta, rueda cambia corte   M I E , . R reset")
    )

    def apply_state():
        """Propaga el estado a los uniforms del ray casting, al filtro y al panel."""
        raycast.use()
        raycast["mode"] = state["modo"]
        raycast["camera_yaw"] = state["yaw"]
        raycast["camera_pitch"] = state["pitch"]
        raycast["early_termination"] = 1 if state["terminacion"] else 0
        raycast["clip_min"] = tuple(state["clip_min"])
        raycast["clip_max"] = tuple(state["clip_max"])

        k = PASOS_K[state["k_index"]]
        raycast["step_size"] = k * voxel_size

        filtro = GL.GL_LINEAR if state["trilineal"] else GL.GL_NEAREST
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, filtro)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, filtro)

        modo = MODOS[state["modo"]]
        interpolacion = "trilineal" if state["trilineal"] else "vecino mas cercano"
        activa_terminacion = state["modo"] == 2 and state["terminacion"]
        cmin, cmax = state["clip_min"], state["clip_max"]
        panel["fuente"] = f"fuente: {fuente}"
        panel["modo"] = f"modo: {modo}"
        panel["interpolacion"] = f"interpolacion: {interpolacion}"
        panel["paso"] = f"paso k: {k:.2f} (del tamano de un voxel)"
        panel["terminacion"] = (
            f"terminacion temprana: {'activa' if activa_terminacion else 'inactiva'}"
        )
        panel["recorte"] = (
            f"recorte x[{cmin[0]:.2f},{cmax[0]:.2f}] "
            f"y[{cmin[1]:.2f},{cmax[1]:.2f}] z[{cmin[2]:.2f},{cmax[2]:.2f}]"
        )

    def _actualizar_overlays(p):
        """Reescribe las posiciones del rectangulo de recorte y la cruz del panel."""
        eje_a, eje_b = p["plano"]
        a0 = state["clip_min"][eje_a] * 2 - 1
        a1 = state["clip_max"][eje_a] * 2 - 1
        b0 = state["clip_min"][eje_b] * 2 - 1
        b1 = state["clip_max"][eje_b] * 2 - 1
        lineas_recorte[p["nombre"]].position[:] = [
            a0, b0, a1, b0,   # borde inferior
            a1, b0, a1, b1,   # borde derecho
            a1, b1, a0, b1,   # borde superior
            a0, b1, a0, b0,   # borde izquierdo
        ]
        sa = state["slice"][eje_a] * 2 - 1
        sb = state["slice"][eje_b] * 2 - 1
        lineas_cruz[p["nombre"]].position[:] = [sa, -1, sa, 1, -1, sb, 1, sb]

    def _panel_en(x, y):
        """Devuelve el panel MPR bajo el cursor, '3d', o None."""
        for p in PANELES:
            rx, ry, rw, rh = layout[p["nombre"]]
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return p
        rx, ry, rw, rh = layout["3d"]
        if rx <= x < rx + rw and ry <= y < ry + rh:
            return "3d"
        return None

    def _mouse_a_uv(p, x, y):
        """Coordenada del cursor en uv del corte (puede caer fuera de [0,1])."""
        rx, ry, rw, rh = layout[p["nombre"]]
        escala_x, escala_y = _quad_scale(p["plano"], (rx, ry, rw, rh), box_half)
        ndc_x = ((x - rx) / rw) * 2 - 1
        ndc_y = ((y - ry) / rh) * 2 - 1
        return (ndc_x / escala_x + 1) / 2, (ndc_y / escala_y + 1) / 2

    @window.event
    def on_draw():
        window.clear()

        rx, ry, rw, rh = layout["3d"]
        GL.glViewport(rx, ry, rw, rh)
        raycast.use()
        raycast["viewport_origin"] = (float(rx), float(ry))
        raycast["viewport_size"] = (float(rw), float(rh))
        raycast_quad.draw(GL.GL_TRIANGLES)

        for p in PANELES:
            rect = layout[p["nombre"]]
            GL.glViewport(*rect)
            escala = _quad_scale(p["plano"], rect, box_half)

            slice_pipe.use()
            slice_pipe["quad_scale"] = escala
            slice_pipe["axis"] = p["axis"]
            slice_pipe["slice_pos"] = float(state["slice"][p["fuera"]])
            slice_pipe["clip_min"] = tuple(state["clip_min"])
            slice_pipe["clip_max"] = tuple(state["clip_max"])
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_3D, texture_id)
            slice_quad.draw(GL.GL_TRIANGLES)

            _actualizar_overlays(p)
            line_pipe.use()
            line_pipe["quad_scale"] = escala
            line_pipe["line_color"] = COLOR_CRUZ
            lineas_cruz[p["nombre"]].draw(GL.GL_LINES)
            line_pipe["line_color"] = COLOR_RECORTE
            lineas_recorte[p["nombre"]].draw(GL.GL_LINES)

        GL.glViewport(0, 0, window.width, window.height)
        panel.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        objetivo = _panel_en(x, y)
        if objetivo == "3d":
            state["drag"] = {"modo": "orbita"}
            return
        if objetivo is None:
            state["drag"] = None
            return

        uv_x, uv_y = _mouse_a_uv(objetivo, x, y)
        if not (-0.05 <= uv_x <= 1.05 and -0.05 <= uv_y <= 1.05):
            state["drag"] = None
            return

        # busca el borde de recorte mas cercano entre los dos ejes del plano.
        eje_a, eje_b = objetivo["plano"]
        candidatos = [
            (abs(uv_x - state["clip_min"][eje_a]), eje_a, "clip_min", "x"),
            (abs(uv_x - state["clip_max"][eje_a]), eje_a, "clip_max", "x"),
            (abs(uv_y - state["clip_min"][eje_b]), eje_b, "clip_min", "y"),
            (abs(uv_y - state["clip_max"][eje_b]), eje_b, "clip_max", "y"),
        ]
        distancia, eje, limite, componente = min(candidatos)
        if distancia < UMBRAL_BORDE:
            state["drag"] = {"modo": "recorte", "eje": eje, "limite": limite,
                             "componente": componente, "panel": objetivo}
        else:
            state["drag"] = {"modo": "cruz", "panel": objetivo}

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        drag = state["drag"]
        if drag is None:
            return

        if drag["modo"] == "orbita":
            state["yaw"] += dx * 0.01
            state["pitch"] = float(np.clip(state["pitch"] - dy * 0.01, -1.4, 1.4))
            apply_state()
            return

        p = drag["panel"]
        uv_x, uv_y = _mouse_a_uv(p, x, y)
        eje_a, eje_b = p["plano"]

        if drag["modo"] == "recorte":
            valor = float(np.clip(uv_x if drag["componente"] == "x" else uv_y, 0.0, 1.0))
            eje, limite = drag["eje"], drag["limite"]
            if limite == "clip_min":
                state["clip_min"][eje] = min(valor, state["clip_max"][eje] - GAP_RECORTE)
            else:
                state["clip_max"][eje] = max(valor, state["clip_min"][eje] + GAP_RECORTE)
            apply_state()
        else:  # cruz: mueve los cortes de los dos planos perpendiculares.
            state["slice"][eje_a] = float(np.clip(uv_x, 0.0, 1.0))
            state["slice"][eje_b] = float(np.clip(uv_y, 0.0, 1.0))

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        state["drag"] = None

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        objetivo = _panel_en(x, y)
        if objetivo in (None, "3d"):
            return
        eje = objetivo["fuera"]
        state["slice"][eje] = float(np.clip(state["slice"][eje] + scroll_y * 0.02, 0.0, 1.0))

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.M:
            state["modo"] = (state["modo"] + 1) % len(MODOS)
        elif symbol == key.I:
            state["trilineal"] = not state["trilineal"]
        elif symbol == key.E:
            state["terminacion"] = not state["terminacion"]
        elif symbol == key.COMMA:
            state["k_index"] = max(0, state["k_index"] - 1)
        elif symbol == key.PERIOD:
            state["k_index"] = min(len(PASOS_K) - 1, state["k_index"] + 1)
        elif symbol == key.R:
            drag = state["drag"]
            state.update(copy.deepcopy(DEFAULTS))
            state["drag"] = drag
        elif symbol == key.ESCAPE:
            window.close()
            return
        apply_state()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    raycasting_volumen()
