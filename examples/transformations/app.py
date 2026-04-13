import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
from pathlib import Path

import click

import grafica.transformations as tr
from grafica.utils import load_pipeline


@click.command("transformed_bunny", short_help='Ejemplo de transformaciones con el conejo de Stanford')
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
def transformed_bunny(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    bunny = tm.load("assets/Stanford_Bunny.stl")

    bunny_scale = tr.uniformScale(2.0 / bunny.scale)
    bunny_translate = tr.translate(*-bunny.centroid)
    bunny_rotate = tr.rotationX(-np.pi / 2)
    bunny.apply_transform(bunny_rotate @ bunny_scale @ bunny_translate)

    state = {
        'total_time': 0.0,
        'transform': tr.identity(),
        # Coordenadas homogéneas: 0.0 = apagado (w = 1), 1.0 = encendido (w = 1 + k·y).
        # Se interpola suavemente para que la transición sea visible.
        'homogeneous_target': 0.0,
        'homogeneous_current': 0.0,
        'show_axes': False,
        'wireframe': False,
        # Parámetros actuales de cada transformación (para el HUD inferior)
        'tilt_rad':  0.0,
        'spin_rad':  0.0,
        'scale_val': 1.0,
    }

    with open(Path(os.path.dirname(__file__)) / "vertex_program.glsl") as f:
        vertex_source_code = f.read()

    with open(Path(os.path.dirname(__file__)) / "fragment_program.glsl") as f:
        fragment_source_code = f.read()

    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)

    bunny_vertex_list = tm.rendering.mesh_to_vertexlist(bunny)
    bunny_gpu = pipeline.vertex_list_indexed(
        len(bunny_vertex_list[4][1]) // 3,
        GL.GL_TRIANGLES,
        bunny_vertex_list[3]
    )
    bunny_gpu.position[:] = bunny_vertex_list[4][1]

    # Normales por vértice de cara, expandidas en el mismo orden que usan las posiciones
    vertex_normals_expanded = bunny.vertex_normals[bunny.faces].reshape(-1, 3)
    bunny_gpu.normal[:] = vertex_normals_expanded.flatten()

    # Pipeline para los ejes de coordenadas: sin normales ni efecto homogéneo
    axes_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "axes_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    # Ejes del sistema de coordenadas del conejo (positivos desde el origen)
    AXIS_LENGTH = 0.80
    axis_x_gpu = axes_pipeline.vertex_list(2, GL.GL_LINES)
    axis_x_gpu.position[:] = np.array([0, 0, 0,  AXIS_LENGTH, 0, 0], dtype=np.float32)
    axis_y_gpu = axes_pipeline.vertex_list(2, GL.GL_LINES)
    axis_y_gpu.position[:] = np.array([0, 0, 0,  0, AXIS_LENGTH, 0], dtype=np.float32)
    axis_z_gpu = axes_pipeline.vertex_list(2, GL.GL_LINES)
    axis_z_gpu.position[:] = np.array([0, 0, 0,  0, 0, AXIS_LENGTH], dtype=np.float32)

    # Etiquetas HUD
    def _label(y):
        return pyglet.text.Label(
            '',
            font_name='Fira Code',
            font_size=13,
            color=(200, 200, 200, 255),
            x=10,
            y=y,
            anchor_y='top',
        )

    label_homogeneous = _label(height - 18)
    label_extras      = _label(height - 38)

    # Etiquetas inferiores: parámetros en tiempo real de cada transformación.
    # El orden de abajo hacia arriba sigue el orden de aplicación: primero scale,
    # luego rotationY, luego rotationX (los vectores se multiplican de derecha a izquierda).
    def _bottom_label(y, color):
        label = pyglet.text.Label(
            '',
            font_name='Fira Code',
            font_size=13,
            color=color,
            x=10,
            y=y,
            anchor_y='bottom',
        )
        return label

    label_param_scale = _bottom_label(18,  (180, 180, 180, 255))   # gris  — scale
    label_param_ry    = _bottom_label(38,  (80,  210, 100, 255))   # verde — rotationY
    label_param_rx    = _bottom_label(58,  (255, 100, 100, 255))   # rojo  — rotationX

    # Máxima intensidad del efecto homogéneo.
    # Con k = 0.32 y y ∈ [-0.9, 0.9] aproximadamente, w queda en [0.71, 1.29]: sin clipping.
    MAX_HOMOGENEOUS_STRENGTH = 0.32

    @window.event
    def on_draw():
        GL.glClearColor(0.07, 0.07, 0.10, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        if state['wireframe']:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

        pipeline.use()
        pipeline["view_transform"] = state['transform'].reshape(16, 1, order="F")
        pipeline["time"] = state['total_time']

        k = state['homogeneous_current'] * MAX_HOMOGENEOUS_STRENGTH
        pipeline["homogeneous_strength"] = k

        bunny_gpu.draw(GL.GL_TRIANGLES)

        # Restaurar relleno antes de los ejes y las etiquetas
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        if state['show_axes']:
            axes_pipeline.use()
            axes_pipeline["view_transform"] = state['transform'].reshape(16, 1, order="F")
            axes_pipeline["axis_color"] = (1.0, 0.28, 0.28)   # X — rojo
            axis_x_gpu.draw(GL.GL_LINES)
            axes_pipeline["axis_color"] = (0.28, 1.0, 0.28)   # Y — verde
            axis_y_gpu.draw(GL.GL_LINES)
            axes_pipeline["axis_color"] = (0.35, 0.60, 1.0)   # Z — azul
            axis_z_gpu.draw(GL.GL_LINES)

        if k < 0.005:
            label_homogeneous.text = "H: activar coordenadas homogéneas   [w = 1]"
        else:
            label_homogeneous.text = f"H: desactivar coordenadas homogéneas   [w = 1 + {k:.2f}·y]"

        axes_str = "activados" if state['show_axes'] else "desactivados"
        wire_str = "activado"  if state['wireframe']  else "desactivado"
        label_extras.text = f"A: ejes {axes_str}   W: wireframe {wire_str}"

        label_homogeneous.draw()
        label_extras.draw()

        tilt_deg  = np.degrees(state['tilt_rad'])
        spin_deg  = np.degrees(state['spin_rad']) % 360
        scale_val = state['scale_val']

        label_param_rx.text    = f"rotationX( {tilt_deg:+7.2f}° )"
        label_param_ry.text    = f"rotationY( {spin_deg:7.2f}° )"
        label_param_scale.text = f"scale(     {scale_val:.4f} )"

        label_param_rx.draw()
        label_param_ry.draw()
        label_param_scale.draw()

    def update_world(dt, window):
        state['total_time'] += dt
        t = state['total_time']

        # Giro principal en Y con balanceo lento en X y pulsación sutil de escala
        tilt_rad  = 0.25 * np.sin(t * 0.7)
        spin_rad  = t * 1.1
        scale_val = 0.88 + 0.06 * np.sin(t * 1.9)

        main_spin    = tr.rotationY(spin_rad)
        slow_tilt    = tr.rotationX(tilt_rad)
        subtle_pulse = tr.uniformScale(scale_val)
        state['transform'] = slow_tilt @ main_spin @ subtle_pulse

        state['tilt_rad']  = tilt_rad
        state['spin_rad']  = spin_rad
        state['scale_val'] = scale_val

        # Interpolación suave hacia el estado objetivo del efecto homogéneo
        alpha = min(1.0, dt * 4.0)
        state['homogeneous_current'] += alpha * (
            state['homogeneous_target'] - state['homogeneous_current']
        )

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.H:
            state['homogeneous_target'] = (
                0.0 if state['homogeneous_target'] > 0.5 else 1.0
            )
        elif symbol == pyglet.window.key.A:
            state['show_axes'] = not state['show_axes']
        elif symbol == pyglet.window.key.W:
            state['wireframe'] = not state['wireframe']

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
