import os
from pathlib import Path

import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import click

import grafica.transformations as tr
from grafica.utils import load_pipeline


@click.command("compositions", short_help="Ejemplo de composición de transformaciones")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
def compositions(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    bunny = tm.load("assets/Stanford_Bunny.stl")

    bunny_scale = tr.uniformScale(1.0 / bunny.scale)
    bunny_translate = tr.translate(*-bunny.centroid)
    bunny_rotate = tr.rotationX(-np.pi / 2)
    bunny.apply_transform(bunny_rotate @ bunny_scale @ bunny_translate)

    bunny_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "mesh_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl",
    )

    bunny_vertex_list = tm.rendering.mesh_to_vertexlist(bunny)
    bunny_gpu = bunny_pipeline.vertex_list_indexed(
        len(bunny_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        bunny_vertex_list[3]
    )
    bunny_gpu.position[:] = bunny_vertex_list[4][1]

    vertex_normals_expanded = bunny.vertex_normals[bunny.faces].reshape(-1, 3)
    bunny_gpu.normal[:] = vertex_normals_expanded.flatten()

    # Transformaciones activas, una por conejo
    transforms = {
        "TL": tr.identity(),
        "TR": tr.identity(),
        "BL": tr.identity(),
        "BR": tr.identity(),
    }

    # Paletas de color: (frío, cálido) por conejo.
    bunny_palettes = {
        "TL": ((0.12, 0.04, 0.55), (1.00, 0.60, 0.05)),  # púrpura → naranja
        "TR": ((0.04, 0.22, 0.78), (0.15, 0.95, 0.88)),  # azul → cian
        "BL": ((0.52, 0.00, 0.32), (1.00, 0.32, 0.68)),  # magenta oscuro → rosa
        "BR": ((0.00, 0.28, 0.22), (0.22, 1.00, 0.48)),  # verde oscuro → verde brillante
    }

    # Parámetros en tiempo real de cada transformación (para las etiquetas)
    params = {
        "TL": {"tx": 0.0, "ty": 0.0, "ry_deg": 0.0},
        "TR": {"tx": 0.0, "ty": 0.0, "rz_deg": 0.0},
        "BL": {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0},
        "BR": {"tx": 0.0, "ty": 0.0, "ry_deg": 0.0, "rx_deg": 0.0},
    }

    total_time = 0.0

    # -----------------------------------------------------------------------
    # Etiquetas de parámetros: una línea por transformación, en la esquina
    # del cuadrante correspondiente, con el color cálido de cada paleta.
    # -----------------------------------------------------------------------
    def _label(x, y, color, anchor_y='top'):
        return pyglet.text.Label(
            '',
            font_name='Fira Code',
            font_size=12,
            color=color,
            x=x, y=y,
            anchor_y=anchor_y,
        )

    # Color cálido de cada paleta (con alfa algo reducido para no competir)
    C_TL = (255, 160,  30, 210)   # naranja
    C_TR = ( 50, 235, 220, 210)   # cian
    C_BL = (255,  90, 175, 210)   # rosa
    C_BR = ( 70, 255, 130, 210)   # verde

    MX = width // 2 + 10   # x de inicio de columnas derechas
    LINE = 18               # separación entre líneas

    # Cada cuadrante tiene 3 etiquetas: posición / ángulo(s) / escala base
    tl_lab = [_label(10,  height - LINE * i, C_TL) for i in range(1, 4)]
    tr_lab = [_label(MX,  height - LINE * i, C_TR) for i in range(1, 4)]
    bl_lab = [_label(10,  LINE * i,          C_BL, anchor_y='bottom') for i in range(1, 4)]
    br_lab = [_label(MX,  LINE * i,          C_BR, anchor_y='bottom') for i in range(1, 4)]

    SCALE = 0.42   # escala base para todos los conejos

    @window.event
    def on_draw():
        GL.glClearColor(0.08, 0.08, 0.12, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        bunny_pipeline.use()

        for name, transform in transforms.items():
            cool, warm = bunny_palettes[name]
            bunny_pipeline["transform"] = transform.reshape(16, 1, order="F")
            bunny_pipeline["color_cool"] = cool
            bunny_pipeline["color_warm"] = warm
            bunny_gpu.draw(GL.GL_TRIANGLES)

        # --- TL ---
        p = params["TL"]
        tl_lab[0].text = f"translate( {p['tx']:+.3f}, {p['ty']:+.3f} )"
        tl_lab[1].text = f"rotationY( {p['ry_deg']:6.1f}° )"
        tl_lab[2].text = f"scale( {SCALE} )"
        for lab in tl_lab: lab.draw()

        # --- TR ---
        p = params["TR"]
        tr_lab[0].text = f"translate( {p['tx']:+.3f}, {p['ty']:+.3f} )"
        tr_lab[1].text = f"rotationZ( {p['rz_deg']:6.1f}° )"
        tr_lab[2].text = f"scale( {SCALE} )"
        for lab in tr_lab: lab.draw()

        # --- BL ---
        p = params["BL"]
        bl_lab[0].text = f"scale( {SCALE} )"
        bl_lab[1].text = f"scale( {p['sx']:.3f}, {p['sy']:.3f}, 1 )"
        bl_lab[2].text = f"translate( {p['tx']:+.3f}, {p['ty']:+.3f} )"
        for lab in bl_lab: lab.draw()

        # --- BR ---
        p = params["BR"]
        br_lab[0].text = f"scale( {SCALE} )"
        br_lab[1].text = f"rotationX( {p['rx_deg']:6.1f}° )"
        br_lab[2].text = f"rotationY( {p['ry_deg']:6.1f}° )"
        for lab in br_lab: lab.draw()

    def update_world(dt, window):
        nonlocal total_time
        total_time += dt
        t = total_time

        # TL: órbita circular dentro del cuadrante + giro en Y.
        orbit_x = -0.5 + 0.18 * np.cos(t * 0.9)
        orbit_y =  0.5 + 0.10 * np.sin(t * 0.9)
        ry_tl = t * 3.5
        transforms["TL"] = (
            tr.translate(orbit_x, orbit_y, 0)
            @ tr.rotationY(ry_tl)
            @ tr.uniformScale(SCALE)
        )
        params["TL"]["tx"]     = orbit_x
        params["TL"]["ty"]     = orbit_y
        params["TL"]["ry_deg"] = np.degrees(ry_tl) % 360

        # TR: trayectoria de Lissajous (figura en 8) + rotación en Z.
        lissajous_x = 0.5 + 0.22 * np.sin(t * 1.1)
        lissajous_y = 0.5 + 0.10 * np.sin(t * 2.2)
        rz_tr = t * 2.2
        transforms["TR"] = (
            tr.translate(lissajous_x, lissajous_y, 0)
            @ tr.rotationZ(rz_tr)
            @ tr.uniformScale(SCALE)
        )
        params["TR"]["tx"]     = lissajous_x
        params["TR"]["ty"]     = lissajous_y
        params["TR"]["rz_deg"] = np.degrees(rz_tr) % 360

        # BL: salto con squash & stretch.
        # jump_height ∈ [0, 0.24] sigue |sin|, que da un perfil de rebote.
        # impact ∈ [0, 1] es máximo en el suelo y cero en la cima:
        #   - suelo: squash (más ancho, más bajo)   sx=1.25, sy=0.80
        #   - cima:  stretch (más alto, más estrecho) sx=0.88, sy=1.18
        jump_phase  = t * 2.2
        jump_height = 0.24 * abs(np.sin(jump_phase))
        impact      = abs(np.cos(jump_phase))
        sq_x = 0.88 + 0.37 * impact
        sq_y = 1.18 - 0.38 * impact
        transforms["BL"] = (
            tr.translate(-0.5, -0.66 + jump_height, 0)
            @ tr.scale(sq_x, sq_y, 1.0)
            @ tr.uniformScale(SCALE)
        )
        params["BL"]["tx"] = -0.5
        params["BL"]["ty"] = -0.66 + jump_height
        params["BL"]["sx"] = sq_x
        params["BL"]["sy"] = sq_y

        # BR: doble rotación en ejes distintos (efecto giroscópico).
        # Ilustra que las rotaciones 3D no conmutan.
        ry_br = t * 3.8
        rx_br = t * 1.5
        transforms["BR"] = (
            tr.translate(0.5, -0.5, 0)
            @ tr.rotationY(ry_br)
            @ tr.rotationX(rx_br)
            @ tr.uniformScale(SCALE)
        )
        params["BR"]["tx"]     = 0.5
        params["BR"]["ty"]     = -0.5
        params["BR"]["ry_deg"] = np.degrees(ry_br) % 360
        params["BR"]["rx_deg"] = np.degrees(rx_br) % 360

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
