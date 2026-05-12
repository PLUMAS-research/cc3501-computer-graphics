import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline


# Coeficientes del material: rojo opaco con un especular blanco.
# Estos valores no cambian durante la ejecucion. Lo que cambia es
# si se aplican o no (toggles) y el exponente de Phong.
MATERIAL_AMBIENT = np.array([0.20, 0.05, 0.05], dtype=np.float32)
MATERIAL_DIFFUSE = np.array([0.80, 0.20, 0.20], dtype=np.float32)
MATERIAL_SPECULAR = np.array([1.00, 1.00, 1.00], dtype=np.float32)

SHININESS_MIN = 1.0
SHININESS_MAX = 256.0
SHININESS_STEP = 1.5

ZERO = np.zeros(3, dtype=np.float32)


@click.command("phong_basico", short_help="Ecuacion de Phong termino por termino")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
def phong_basico(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    graph = Scenegraph("root")
    graph.load_and_register_mesh("sphere", "assets/sphere.off")

    sphere_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "sphere_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "sphere_fragment_program.glsl",
    )
    bulb_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "bulb_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "bulb_fragment_program.glsl",
    )
    graph.register_pipeline("sphere_pipeline", sphere_pipeline)
    graph.register_pipeline("bulb_pipeline", bulb_pipeline)

    graph.add_object(
        "sphere",
        "sphere",
        "sphere_pipeline",
        parent="root",
        transform=tr.identity(),
        material_ambient=MATERIAL_AMBIENT,
        material_diffuse=MATERIAL_DIFFUSE,
        material_specular=MATERIAL_SPECULAR,
        material_shininess=32.0,
    )

    bulb_scale = 0.08
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    graph.add_object(
        "bulb",
        "sphere",
        "bulb_pipeline",
        parent="root",
        transform=tr.uniformScale(bulb_scale),
        bulb_color=light_color,
    )

    camera_position = np.array([2.2, 2.2, 1.6], dtype=np.float32)
    view = tr.lookAt(
        camera_position,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    projection = tr.perspective(45, width / height, 0.01, 10.0)

    graph.register_view_transform(view)
    graph.set_global_attributes(
        projection=projection,
        view_position=camera_position,
        light_color=light_color,
        ambient_light=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    state = {
        "ambient_on": True,
        "diffuse_on": True,
        "specular_on": True,
        "shininess": 32.0,
    }

    label_ambient = pyglet.text.Label(
        "", font_name="Fira Code", font_size=13,
        x=12, y=height - 28, color=(255, 255, 255, 255),
    )
    label_diffuse = pyglet.text.Label(
        "", font_name="Fira Code", font_size=13,
        x=12, y=height - 50, color=(255, 255, 255, 255),
    )
    label_specular = pyglet.text.Label(
        "", font_name="Fira Code", font_size=13,
        x=12, y=height - 72, color=(255, 255, 255, 255),
    )
    label_shininess = pyglet.text.Label(
        "", font_name="Fira Code", font_size=13,
        x=12, y=height - 94, color=(255, 255, 255, 255),
    )
    instrucciones = pyglet.text.Label(
        "1/2/3 prender o apagar ambient/diffuse/specular   -/= shininess",
        font_name="Fira Code", font_size=11,
        x=12, y=12, color=(220, 220, 220, 255),
    )

    def fmt_toggle(on):
        return "[ON] " if on else "[off]"

    def apply_state():
        graph.apply_instance_attributes(
            "sphere_mesh",
            material_ambient=MATERIAL_AMBIENT if state["ambient_on"] else ZERO,
            material_diffuse=MATERIAL_DIFFUSE if state["diffuse_on"] else ZERO,
            material_specular=MATERIAL_SPECULAR if state["specular_on"] else ZERO,
            material_shininess=float(state["shininess"]),
        )
        label_ambient.text = (
            f"ambient   {fmt_toggle(state['ambient_on'])} k_a = "
            f"({MATERIAL_AMBIENT[0]:.2f}, {MATERIAL_AMBIENT[1]:.2f}, {MATERIAL_AMBIENT[2]:.2f})"
        )
        label_diffuse.text = (
            f"diffuse   {fmt_toggle(state['diffuse_on'])} k_d = "
            f"({MATERIAL_DIFFUSE[0]:.2f}, {MATERIAL_DIFFUSE[1]:.2f}, {MATERIAL_DIFFUSE[2]:.2f})"
        )
        label_specular.text = (
            f"specular  {fmt_toggle(state['specular_on'])} k_s = "
            f"({MATERIAL_SPECULAR[0]:.2f}, {MATERIAL_SPECULAR[1]:.2f}, {MATERIAL_SPECULAR[2]:.2f})"
        )
        label_shininess.text = f"shininess        alpha = {state['shininess']:6.2f}"
        print(
            f"[phong] ambient={state['ambient_on']} diffuse={state['diffuse_on']} "
            f"specular={state['specular_on']} alpha={state['shininess']:.2f}"
        )

    apply_state()

    total_time = 0.0

    @window.event
    def on_draw():
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        graph.set_global_attributes(
            light_position=graph.get_global_position("bulb").astype(np.float32),
        )
        graph.render(recalculate_transforms=False)

        with ui_overlay():
            label_ambient.draw()
            label_diffuse.draw()
            label_specular.draw()
            label_shininess.draw()
            instrucciones.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key._1:
            state["ambient_on"] = not state["ambient_on"]
            apply_state()
        elif symbol == pyglet.window.key._2:
            state["diffuse_on"] = not state["diffuse_on"]
            apply_state()
        elif symbol == pyglet.window.key._3:
            state["specular_on"] = not state["specular_on"]
            apply_state()
        elif symbol == pyglet.window.key.MINUS:
            state["shininess"] = max(SHININESS_MIN, state["shininess"] / SHININESS_STEP)
            apply_state()
        elif symbol == pyglet.window.key.EQUAL:
            state["shininess"] = min(SHININESS_MAX, state["shininess"] * SHININESS_STEP)
            apply_state()

    def update_world(dt, _):
        nonlocal total_time
        total_time += dt

        # La luz orbita en un plano horizontal a velocidad baja.
        # Asi el highlight especular se desplaza sobre la esfera
        # y se nota cuando el termino esta activo.
        radio = 1.6
        velocidad = 0.6
        angulo = total_time * velocidad
        graph.nodes["bulb"]["transform"] = tr.translate(
            radio * np.cos(angulo),
            radio * np.sin(angulo),
            0.6,
        ) @ tr.uniformScale(bulb_scale)

        graph.calculate_global_transforms()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
