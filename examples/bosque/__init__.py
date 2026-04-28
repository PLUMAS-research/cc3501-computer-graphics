from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from PIL import Image, ImageDraw, ImageFilter

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.textures import texture_2D_setup
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline


COLOR_CIELO = (0.78, 0.86, 0.93)
COLOR_PASTO = np.array([0.42, 0.62, 0.32], dtype=np.float32)

MODOS_BLEND = ["NINGUNO", "STANDARD", "PREMULTIPLICADO"]


def crear_textura_arbol(size=128, blur_radius=5):
    """Genera una textura de arbol con un borde difuso ancho.

    El blur gaussiano se aplica sobre RGBA, lo que mezcla los pixeles opacos
    (RGB del color del arbol) con los transparentes (RGB cero). El resultado
    son pixeles intermedios con RGB proporcional al alpha. Esa es justo la
    condicion clasica que produce bordes oscuros con blending estandar y que
    la premultiplicacion corrige.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = size // 2
    tronco_w = max(2, size // 12)
    tronco_h = int(size * 0.40)
    draw.rectangle(
        [(cx - tronco_w, size - tronco_h), (cx + tronco_w, size - 1)],
        fill=(110, 70, 40, 255),
    )

    copa_radio = size * 0.32
    copa_y = size * 0.38
    draw.ellipse(
        [
            (cx - copa_radio, copa_y - copa_radio),
            (cx + copa_radio, copa_y + copa_radio),
        ],
        fill=(80, 200, 90, 255),
    )

    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def construir_arboles(seed, n_arboles, area):
    rng = np.random.default_rng(seed)
    posiciones = (rng.random((n_arboles, 2)) - 0.5) * 2.0 * area
    posiciones_3d = np.zeros((n_arboles, 3), dtype=np.float32)
    posiciones_3d[:, 0] = posiciones[:, 0]
    posiciones_3d[:, 2] = posiciones[:, 1]
    alturas = (rng.random(n_arboles) * 0.7 + 1.6).astype(np.float32)
    return posiciones_3d, alturas


def crear_vl_pasto(pipeline, area, color):
    p0 = (-area, 0.0, -area)
    p1 = (area, 0.0, -area)
    p2 = (area, 0.0, area)
    p3 = (-area, 0.0, area)
    posiciones = np.array([p0, p1, p2, p0, p2, p3], dtype=np.float32).flatten()
    colores = np.tile(np.asarray(color, dtype=np.float32), 6)
    vl = pipeline.vertex_list(6, GL.GL_TRIANGLES, position="f", color="f")
    vl.position[:] = posiciones
    vl.color[:] = colores
    return vl


def crear_vl_arbol(pipeline, base_pos, ancho, alto):
    """Un quad por arbol. Cada vertice lleva la posicion base, su offset local
    en el quad y la coordenada UV. El vertex shader arma el billboard."""
    corners = np.array(
        [[-0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [-0.5, 1.0]], dtype=np.float32
    )
    uvs = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    indices = [0, 1, 2, 0, 2, 3]

    base_position = np.tile(np.asarray(base_pos, dtype=np.float32), (6, 1))
    local_offset = np.array(
        [corners[i] * np.array([ancho, alto], dtype=np.float32) for i in indices],
        dtype=np.float32,
    )
    texcoord = np.array([uvs[i] for i in indices], dtype=np.float32)

    vl = pipeline.vertex_list(
        6,
        GL.GL_TRIANGLES,
        base_position="f",
        local_offset="f",
        texcoord="f",
    )
    vl.base_position[:] = base_position.flatten()
    vl.local_offset[:] = local_offset.flatten()
    vl.texcoord[:] = texcoord.flatten()
    return vl


def aplicar_blend_mode(modo):
    if modo == "NINGUNO":
        GL.glDisable(GL.GL_BLEND)
    elif modo == "STANDARD":
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    elif modo == "PREMULTIPLICADO":
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)


def ordenar_arboles(posiciones, view_matrix):
    """Indices de los arboles ordenados por profundidad de mas lejano a mas
    cercano. Usa el centro del arbol en espacio de camara."""
    n = len(posiciones)
    homogeneas = np.hstack([posiciones, np.ones((n, 1), dtype=np.float32)])
    en_camara = (view_matrix @ homogeneas.T).T
    profundidades = -en_camara[:, 2]
    return list(np.argsort(profundidades)[::-1])


@click.command(
    "bosque",
    short_help="Bosque de billboards para alpha test, blending y orden",
)
@click.option("--n_arboles", type=int, default=30)
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=768)
def bosque(n_arboles, width, height):
    window = pyglet.window.Window(width, height)
    window.set_caption("Bosque: alpha test, blending y orden de transparencia")

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    pipeline_pasto = load_pipeline(
        Path(__file__).parent / "ground_vertex.glsl",
        Path(__file__).parent / "ground_fragment.glsl",
    )
    pipeline_billboard = load_pipeline(
        Path(__file__).parent / "billboard_vertex.glsl",
        Path(__file__).parent / "billboard_fragment.glsl",
    )

    textura_arbol = texture_2D_setup(crear_textura_arbol(128))

    posiciones_arboles, alturas_arboles = construir_arboles(
        seed=42, n_arboles=n_arboles, area=3.0
    )

    pasto_vl = crear_vl_pasto(pipeline_pasto, area=4.0, color=COLOR_PASTO)
    arboles_vls = [
        crear_vl_arbol(
            pipeline_billboard,
            posiciones_arboles[i],
            alturas_arboles[i] * 0.7,
            alturas_arboles[i],
        )
        for i in range(n_arboles)
    ]

    near_plane = 0.1
    far_plane = 50.0
    projection = tr.perspective(
        45.0, float(width) / float(height), near_plane, far_plane
    )
    view_inicial = tr.lookAt(
        np.array([5.5, 2.2, 5.5]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    arcball = Arcball(
        np.linalg.inv(view_inicial),
        np.array((width, height), dtype=float),
        2.5,
        np.array([0.0, 1.0, 0.0]),
    )

    estado = {
        "alpha_test": False,
        "blend_mode_index": 1,
        "ordenar": True,
        "depth_test": True,
        "filtro_lineal": True,
    }

    def actualizar_filtro():
        modo = GL.GL_LINEAR if estado["filtro_lineal"] else GL.GL_NEAREST
        GL.glBindTexture(GL.GL_TEXTURE_2D, textura_arbol)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, modo)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, modo)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        elif button == pyglet.window.mouse.MIDDLE:
            arcball.set_state(Arcball.STATE_ZOOM)
        arcball.down((x, y))

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        arcball.set_state(Arcball.STATE_ROTATE)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.A:
            estado["alpha_test"] = not estado["alpha_test"]
        elif symbol == pyglet.window.key.B:
            estado["blend_mode_index"] = (
                estado["blend_mode_index"] + 1
            ) % len(MODOS_BLEND)
        elif symbol == pyglet.window.key.S:
            estado["ordenar"] = not estado["ordenar"]
        elif symbol == pyglet.window.key.D:
            estado["depth_test"] = not estado["depth_test"]
        elif symbol == pyglet.window.key.T:
            estado["filtro_lineal"] = not estado["filtro_lineal"]
            actualizar_filtro()
        elif symbol == pyglet.window.key.R:
            arcball.pose = np.linalg.inv(view_inicial)

    hud_label = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=11,
        x=10,
        y=height - 10,
        anchor_x="left",
        anchor_y="top",
        color=(20, 30, 50, 255),
        multiline=True,
        width=width - 20,
    )

    def actualizar_hud():
        modo_blend = MODOS_BLEND[estado["blend_mode_index"]]
        hud_label.text = (
            f"Arboles: {n_arboles}\n"
            f"[A] Alpha test:    {'ON' if estado['alpha_test'] else 'OFF'}\n"
            f"[B] Blending:      {modo_blend}\n"
            f"[S] Ordenar:       {'ON' if estado['ordenar'] else 'OFF'}\n"
            f"[D] Depth test:    {'ON' if estado['depth_test'] else 'OFF'}\n"
            f"[T] Filtro:        {'LINEAR' if estado['filtro_lineal'] else 'NEAREST'}\n"
            f"[R] Reset camara   Mouse: izq rota, der mueve, rueda zoom"
        )

    @window.event
    def on_draw():
        GL.glClearColor(*COLOR_CIELO, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if estado["depth_test"]:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

        view_matrix = np.linalg.inv(arcball.pose)
        camera_position = arcball.pose[:3, 3].astype(np.float32)

        # Pasto opaco. Profundidad activa, sin blending.
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)

        pipeline_pasto.use()
        pipeline_pasto["view"] = view_matrix.reshape(16, 1, order="F")
        pipeline_pasto["projection"] = projection.reshape(16, 1, order="F")
        pasto_vl.draw(GL.GL_TRIANGLES)

        # Arboles
        modo_blend = MODOS_BLEND[estado["blend_mode_index"]]
        aplicar_blend_mode(modo_blend)

        # Con alpha test los pixeles que pasan son completamente opacos, asi que
        # podemos escribir profundidad y prescindir del orden. Sin alpha test la
        # transparencia es continua y necesitamos el truco clasico: ordenar
        # back-to-front y desactivar la escritura de profundidad.
        if estado["alpha_test"]:
            GL.glDepthMask(GL.GL_TRUE)
            pipeline_billboard.use()
            pipeline_billboard["alpha_test_threshold"] = 0.5
        else:
            GL.glDepthMask(GL.GL_FALSE)
            pipeline_billboard.use()
            pipeline_billboard["alpha_test_threshold"] = -1.0

        pipeline_billboard["view"] = view_matrix.reshape(16, 1, order="F")
        pipeline_billboard["projection"] = projection.reshape(16, 1, order="F")
        pipeline_billboard["camera_position"] = camera_position
        pipeline_billboard["diffuse"] = 0
        pipeline_billboard["premultiply"] = bool(modo_blend == "PREMULTIPLICADO")

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, textura_arbol)

        if estado["ordenar"]:
            orden = ordenar_arboles(posiciones_arboles, view_matrix)
        else:
            orden = list(range(n_arboles))

        for i in orden:
            arboles_vls[i].draw(GL.GL_TRIANGLES)

        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        actualizar_hud()
        with ui_overlay():
            hud_label.draw()

    pyglet.app.run()


if __name__ == "__main__":
    bosque()
