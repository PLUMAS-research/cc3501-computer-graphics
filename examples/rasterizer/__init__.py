import numpy as np
import pyglet
import OpenGL.GL as GL

from pathlib import Path
from grafica.utils import load_pipeline
import click


def barycentric(px, py, v1, v2, v3):
    """Calcula coordenadas baricentricas (alpha, beta, gamma) de (px, py)
    respecto al triangulo (v1, v2, v3).

    Retorna (alpha, beta, gamma). Si alguna es negativa, el punto esta
    fuera del triangulo.
    """
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return -1, -1, -1
    alpha = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    beta = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    gamma = 1.0 - alpha - beta
    return alpha, beta, gamma


def rasterize_triangle(buffer, v1, v2, v3, c1, c2, c3, mode="color"):
    """Rasteriza un triangulo en el buffer usando coordenadas baricentricas.

    Parameters
    ----------
    buffer : ndarray de forma (h, w, 3), uint8
    v1, v2, v3 : tuplas (x, y) en coordenadas de pixel del buffer
    c1, c2, c3 : tuplas (r, g, b) con valores [0, 1]
    mode : "color" para interpolar colores, "barycentric" para visualizar
           las coordenadas baricentricas como RGB
    """
    h, w = buffer.shape[:2]

    # bounding box del triangulo, restringido al buffer
    xs = [v1[0], v2[0], v3[0]]
    ys = [v1[1], v2[1], v3[1]]
    x_min = max(0, int(np.floor(min(xs))))
    x_max = min(w - 1, int(np.ceil(max(xs))))
    y_min = max(0, int(np.floor(min(ys))))
    y_max = min(h - 1, int(np.ceil(max(ys))))

    for py in range(y_min, y_max + 1):
        for px in range(x_min, x_max + 1):
            # centro del pixel
            cx = px + 0.5
            cy = py + 0.5
            alpha, beta, gamma = barycentric(cx, cy, v1, v2, v3)

            if alpha < 0 or beta < 0 or gamma < 0:
                continue

            if mode == "barycentric":
                r, g, b = alpha, beta, gamma
            else:
                r = alpha * c1[0] + beta * c2[0] + gamma * c3[0]
                g = alpha * c1[1] + beta * c2[1] + gamma * c3[1]
                b = alpha * c1[2] + beta * c2[2] + gamma * c3[2]

            buffer[py, px] = (
                int(np.clip(r * 255, 0, 255)),
                int(np.clip(g * 255, 0, 255)),
                int(np.clip(b * 255, 0, 255)),
            )


def build_grid_lines(raster_w, raster_h, win_width, win_height):
    """Construye una lista de pyglet.shapes.Line para la grilla en pantalla.

    Solo se construye si la resolucion es baja (pixeles visibles).
    """
    if raster_w > 128 or raster_h > 128:
        return [], None

    batch = pyglet.graphics.Batch()
    lines = []
    cell_w = win_width / raster_w
    cell_h = win_height / raster_h

    for i in range(raster_w + 1):
        x = round(i * cell_w)
        lines.append(
            pyglet.shapes.Line(x, 0, x, win_height, color=(80, 80, 80, 160), batch=batch)
        )
    for j in range(raster_h + 1):
        y = round(j * cell_h)
        lines.append(
            pyglet.shapes.Line(0, y, win_width, y, color=(80, 80, 80, 160), batch=batch)
        )

    return lines, batch


@click.command(
    "rasterizer",
    short_help="Rasterizador por software con coordenadas baricentricas",
)
@click.option("--width", type=int, default=800, help="Ancho de la ventana")
@click.option("--height", type=int, default=600, help="Alto de la ventana")
@click.option(
    "--resolution",
    type=int,
    default=40,
    help="Resolucion del raster (pixeles de ancho)",
)
def rasterizer(width, height, resolution):
    window = pyglet.window.Window(width=width, height=height)
    window.set_caption("Rasterizador - arrastra vertices, B: baricentricas, G: grilla")

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    # resolucion del raster (independiente de la ventana)
    raster_w = resolution
    raster_h = int(resolution * height / width)

    state = {
        "mode": "color",  # "color" o "barycentric"
        "show_grid": True,
        "dragging": None,  # indice del vertice que se arrastra
        "raster_w": raster_w,
        "raster_h": raster_h,
        "needs_update": True,
        "grid_batch": None,
        "grid_lines": [],
    }

    # dos triangulos con vertices en coordenadas de raster
    # y colores asignados a cada vertice
    triangles = [
        {
            "vertices": [
                [raster_w * 0.15, raster_h * 0.15],
                [raster_w * 0.75, raster_h * 0.25],
                [raster_w * 0.35, raster_h * 0.85],
            ],
            "colors": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        },
        {
            "vertices": [
                [raster_w * 0.55, raster_h * 0.20],
                [raster_w * 0.90, raster_h * 0.80],
                [raster_w * 0.40, raster_h * 0.65],
            ],
            "colors": [(1, 1, 0), (0, 1, 1), (1, 0, 1)],
        },
    ]

    # textura para el raster
    texture_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )

    initial_data = np.zeros((raster_h, raster_w, 3), dtype=np.uint8)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
        raster_w, raster_h, 0,
        GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
        initial_data.tobytes(),
    )

    # quad que cubre la pantalla
    vertices_quad = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    uv_quad = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32)
    indices_quad = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    gpu_quad = pipeline.vertex_list_indexed(
        4, GL.GL_TRIANGLES, indices_quad,
        position="f", texcoord="f",
    )
    gpu_quad.position[:] = vertices_quad
    gpu_quad.texcoord[:] = uv_quad

    def rebuild_raster():
        rw = state["raster_w"]
        rh = state["raster_h"]
        bg = 240
        buffer = np.full((rh, rw, 3), bg, dtype=np.uint8)

        for tri in triangles:
            v = tri["vertices"]
            c = tri["colors"]
            rasterize_triangle(
                buffer,
                tuple(v[0]), tuple(v[1]), tuple(v[2]),
                c[0], c[1], c[2],
                mode=state["mode"],
            )

        # marcar los vertices con un punto
        for tri in triangles:
            for vx, vy in tri["vertices"]:
                ix, iy = int(round(vx)), int(round(vy))
                if 0 <= ix < rw and 0 <= iy < rh:
                    buffer[iy, ix] = (255, 255, 255)

        return buffer

    def upload_texture(buffer):
        rw = state["raster_w"]
        rh = state["raster_h"]
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
            rw, rh, 0,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
            buffer.tobytes(),
        )

    def rebuild_grid():
        lines, batch = build_grid_lines(
            state["raster_w"], state["raster_h"], width, height
        )
        state["grid_lines"] = lines
        state["grid_batch"] = batch

    def window_to_raster(wx, wy):
        """Convierte coordenadas de ventana a coordenadas de raster."""
        rx = wx / width * state["raster_w"]
        ry = wy / height * state["raster_h"]
        return rx, ry

    def find_nearest_vertex(rx, ry, threshold=None):
        """Encuentra el vertice mas cercano al punto (rx, ry) en coords raster."""
        if threshold is None:
            threshold = max(state["raster_w"], state["raster_h"]) * 0.08
        best = None
        best_dist = threshold
        for ti, tri in enumerate(triangles):
            for vi, (vx, vy) in enumerate(tri["vertices"]):
                d = np.sqrt((vx - rx) ** 2 + (vy - ry) ** 2)
                if d < best_dist:
                    best_dist = d
                    best = (ti, vi)
        return best

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.B:
            state["mode"] = (
                "barycentric" if state["mode"] == "color" else "color"
            )
            state["needs_update"] = True
        elif symbol == pyglet.window.key.G:
            state["show_grid"] = not state["show_grid"]
            state["needs_update"] = True
            rebuild_grid()
        elif symbol == pyglet.window.key.PLUS or symbol == pyglet.window.key.EQUAL:
            new_w = min(256, state["raster_w"] + 10)
            rescale_vertices(state["raster_w"], state["raster_h"], new_w)
            state["needs_update"] = True
        elif symbol == pyglet.window.key.MINUS:
            new_w = max(10, state["raster_w"] - 10)
            rescale_vertices(state["raster_w"], state["raster_h"], new_w)
            state["needs_update"] = True

    def rescale_vertices(old_w, old_h, new_w):
        new_h = int(new_w * height / width)
        for tri in triangles:
            for v in tri["vertices"]:
                v[0] = v[0] / old_w * new_w
                v[1] = v[1] / old_h * new_h
        state["raster_w"] = new_w
        state["raster_h"] = new_h
        rebuild_grid()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            rx, ry = window_to_raster(x, y)
            hit = find_nearest_vertex(rx, ry)
            state["dragging"] = hit

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        state["dragging"] = None

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if state["dragging"] is not None:
            ti, vi = state["dragging"]
            rx, ry = window_to_raster(x, y)
            rx = np.clip(rx, 0, state["raster_w"] - 1)
            ry = np.clip(ry, 0, state["raster_h"] - 1)
            triangles[ti]["vertices"][vi] = [rx, ry]
            state["needs_update"] = True

    @window.event
    def on_draw():
        window.clear()

        if state["needs_update"]:
            buffer = rebuild_raster()
            upload_texture(buffer)
            state["needs_update"] = False

        pipeline.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

        sampler_loc = GL.glGetUniformLocation(pipeline.id, "raster_tex")
        if sampler_loc != -1:
            GL.glUniform1i(sampler_loc, 0)

        gpu_quad.draw(GL.GL_TRIANGLES)

        if state["show_grid"] and state["grid_batch"] is not None:
            state["grid_batch"].draw()

        draw_hud()

    rebuild_grid()

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )
    hud_label = pyglet.text.Label(
        "",
        font_name="FiraCode",
        font_size=11,
        x=10,
        y=height - 20,
        color=(60, 60, 60, 220),
        multiline=True,
        width=width - 20,
    )

    def draw_hud():
        mode_str = (
            "baricentricas (a=R, b=G, g=B)"
            if state["mode"] == "barycentric"
            else "color interpolado"
        )
        hud_label.text = (
            f"Raster: {state['raster_w']}x{state['raster_h']}  |  "
            f"Modo: {mode_str}\n"
            f"[Arrastrar] mover vertice  [B] modo  "
            f"[G] grilla  [+/-] resolucion"
        )
        hud_label.draw()

    pyglet.app.run()
