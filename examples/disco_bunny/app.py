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


# Cada material es un conjunto de coeficientes que describe cómo la
# superficie reacciona a la luz. Los presets contrastan en tres ejes
# para que las diferencias sean visualmente obvias:
# (1) intensidad de la especular k_s, (2) tinte de la especular
# (cromática vs blanca) y (3) exponente de Phong (highlight ancho
# y borroso vs un punto pequeño y nítido).
MATERIALES = {
    "mate": {
        "label": "1 mate (sin reflejo)",
        "material_ambient": np.array([0.25, 0.05, 0.05], dtype=np.float32),
        "material_diffuse": np.array([0.75, 0.15, 0.15], dtype=np.float32),
        "material_specular": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "material_shininess": 1.0,
    },
    "plastico": {
        "label": "2 plastico (highlight ancho y blanco)",
        "material_ambient": np.array([0.05, 0.12, 0.22], dtype=np.float32),
        "material_diffuse": np.array([0.10, 0.50, 0.90], dtype=np.float32),
        "material_specular": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "material_shininess": 32.0,
    },
    "cromo": {
        "label": "3 cromo (highlight pequeno y filoso)",
        "material_ambient": np.array([0.15, 0.15, 0.15], dtype=np.float32),
        "material_diffuse": np.array([0.30, 0.30, 0.30], dtype=np.float32),
        "material_specular": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "material_shininess": 200.0,
    },
    "oro": {
        "label": "4 oro (highlight tenido)",
        "material_ambient": np.array([0.25, 0.15, 0.0], dtype=np.float32),
        "material_diffuse": np.array([0.85, 0.65, 0.10], dtype=np.float32),
        "material_specular": np.array([1.0, 0.85, 0.40], dtype=np.float32),
        "material_shininess": 80.0,
    },
}

ORDEN_MATERIALES = ["mate", "plastico", "cromo", "oro"]


@click.command("disco_bunny", short_help="Iluminacion de Phong con materiales seleccionables")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
def disco_bunny(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    graph = Scenegraph("root")

    graph.load_and_register_mesh("stanford_bunny", "assets/Stanford_Bunny.stl")
    graph.load_and_register_mesh("sphere", "assets/sphere.off")

    bunny_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "bunny_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "bunny_fragment_program.glsl",
    )
    bulb_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "bulb_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "bulb_fragment_program.glsl",
    )

    graph.register_pipeline("bunny_pipeline", bunny_pipeline)
    graph.register_pipeline("bulb_pipeline", bulb_pipeline)

    bunny_min_z = graph.meshes["stanford_bunny"]["object"].bounds[0][2]
    bunny_base_height = -bunny_min_z / 2

    material_actual = ORDEN_MATERIALES[0]
    graph.add_object(
        "bunny",
        "stanford_bunny",
        "bunny_pipeline",
        parent="root",
        transform=tr.translate(0, 0, bunny_base_height),
        **{k: v for k, v in MATERIALES[material_actual].items() if k != "label"},
    )

    bulb_scale = 0.15
    bulb_1_color = np.array([0.25, 0.85, 1.0], dtype=np.float32)
    bulb_2_color = np.array([1.0, 0.45, 0.20], dtype=np.float32)

    graph.add_object(
        "bulb_1", "sphere", "bulb_pipeline",
        parent="root",
        transform=tr.uniformScale(bulb_scale),
        bulb_color=bulb_1_color,
    )
    graph.add_object(
        "bulb_2", "sphere", "bulb_pipeline",
        parent="root",
        transform=tr.uniformScale(bulb_scale),
        bulb_color=bulb_2_color,
    )

    camera_position = np.array([-2.0, 0, 0.75], dtype=np.float32)
    view = tr.lookAt(
        camera_position,
        np.array([0, 0.0, 0.5]),
        np.array([0.0, 0.0, 1.0]),
    )
    projection = tr.perspective(60, width / height, 0.001, 5.0)

    graph.register_view_transform(view)
    graph.set_global_attributes(
        projection=projection,
        view_position=camera_position,
        light_1_color=bulb_1_color,
        light_2_color=bulb_2_color,
        ambient_light=np.array([0.45, 0.45, 0.45], dtype=np.float32),
    )

    label = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=14,
        x=12,
        y=height - 24,
        color=(255, 255, 255, 255),
    )
    instrucciones = pyglet.text.Label(
        "1: mate    2: plastico    3: cromo    4: oro",
        font_name="Fira Code",
        font_size=11,
        x=12,
        y=12,
        color=(220, 220, 220, 255),
    )

    def aplicar_material(nombre):
        nonlocal material_actual
        material_actual = nombre
        propiedades = {k: v for k, v in MATERIALES[nombre].items() if k != "label"}
        graph.apply_instance_attributes("bunny_mesh", **propiedades)
        label.text = MATERIALES[nombre]["label"]

    aplicar_material(material_actual)

    total_time = 0.0

    @window.event
    def on_draw():
        GL.glClearColor(0.1, 0.0, 0.1, 1.0)
        GL.glLineWidth(1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_DEPTH_TEST)

        window.clear()

        graph.set_global_attributes(
            light_1_position=graph.get_global_position("bulb_1").astype(np.float32),
            light_2_position=graph.get_global_position("bulb_2").astype(np.float32),
        )
        graph.render(recalculate_transforms=False)

        with ui_overlay():
            label.draw()
            instrucciones.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        teclas = {
            pyglet.window.key._1: "mate",
            pyglet.window.key._2: "plastico",
            pyglet.window.key._3: "cromo",
            pyglet.window.key._4: "oro",
        }
        if symbol in teclas:
            aplicar_material(teclas[symbol])

    def update_world(dt, _):
        nonlocal total_time
        total_time += dt

        graph.nodes["bunny"]["transform"] = (
            tr.translate(0, 0, bunny_base_height) @ tr.rotationZ(total_time * 0.3)
        )

        # las dos luces orbitan a velocidad y altura constantes en
        # lados opuestos, así el reflejo especular queda el tiempo
        # suficiente como para comparar materiales sin pestañear
        radio_orbita = 0.9
        velocidad_orbita = 0.6

        angulo_1 = total_time * velocidad_orbita
        graph.nodes["bulb_1"]["transform"] = tr.translate(
            radio_orbita * np.cos(angulo_1),
            radio_orbita * np.sin(angulo_1),
            0.7,
        ) @ tr.uniformScale(bulb_scale)

        angulo_2 = total_time * velocidad_orbita + np.pi
        graph.nodes["bulb_2"]["transform"] = tr.translate(
            radio_orbita * np.cos(angulo_2),
            radio_orbita * np.sin(angulo_2),
            0.5,
        ) @ tr.uniformScale(bulb_scale)

        graph.calculate_global_transforms()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
