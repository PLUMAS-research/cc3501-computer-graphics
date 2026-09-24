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


# Para cel-shading el material sigue describiendo cómo la superficie
# reacciona a la luz, pero el fragment shader cuantiza la componente
# difusa en bandas discretas y umbraliza la especular. Cambiar
# material_diffuse, material_specular o material_shininess sigue
# teniendo el mismo significado que en el modelo de Phong tradicional.
MATERIALES = {
    "rojo": {
        "label": "Rojo",
        "material_diffuse": np.array([0.85, 0.20, 0.20], dtype=np.float32),
        "material_specular": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "material_shininess": 32.0,
    },
    "verde": {
        "label": "Verde",
        "material_diffuse": np.array([0.20, 0.65, 0.30], dtype=np.float32),
        "material_specular": np.array([0.9, 1.0, 0.9], dtype=np.float32),
        "material_shininess": 32.0,
    },
    "azul": {
        "label": "Azul",
        "material_diffuse": np.array([0.20, 0.40, 0.85], dtype=np.float32),
        "material_specular": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "material_shininess": 32.0,
    },
}

ORDEN_MATERIALES = ["rojo", "verde", "azul"]

# Dos modelos para el mismo sombreado. El conejo no tiene mapa difuso, así que
# su color base es un uniform y las teclas 1 a 3 lo cambian. Samus sí lo tiene,
# y ahí el color base sale de la textura: lo que se cuantiza es la iluminación,
# que es como trabajan los juegos con este estilo.
MODELOS = {
    "bunny": {
        "archivo": "assets/Stanford_Bunny.stl",
        "orientacion": tr.identity(),
        "camara": np.array([-2.0, 0.0, 0.75], dtype=np.float32),
        "altura_camara": 0.5,
    },
    "samus": {
        # El modelo viene con el eje Y hacia arriba y la escena trabaja con Z.
        "archivo": "assets/samus/posed.obj",
        "orientacion": tr.rotationX(np.pi / 2),
        "camara": np.array([-2.6, 0.0, 1.1], dtype=np.float32),
        "altura_camara": 0.85,
    },
}


@click.command("cel_bunny", short_help="Cel-shading sobre el conejo de Stanford")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
@click.option(
    "--modelo",
    type=click.Choice(sorted(MODELOS)),
    default="bunny",
    help="Modelo a sombrear: el conejo sin textura o Samus con su mapa difuso",
)
def cel_bunny(width, height, modelo):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    graph = Scenegraph("root")

    configuracion = MODELOS[modelo]
    con_textura = modelo != "bunny"

    graph.load_and_register_mesh("modelo", configuracion["archivo"])
    graph.load_and_register_mesh("sphere", "assets/sphere.off")

    here = Path(os.path.dirname(__file__))
    if con_textura:
        cel_pipeline = load_pipeline(
            here / "cel_textured_vertex_program.glsl",
            here / "cel_textured_fragment_program.glsl",
        )
    else:
        cel_pipeline = load_pipeline(
            here / "cel_vertex_program.glsl",
            here / "cel_fragment_program.glsl",
        )
    bulb_pipeline = load_pipeline(
        here / "bulb_vertex_program.glsl",
        here / "bulb_fragment_program.glsl",
    )

    graph.register_pipeline("cel_pipeline", cel_pipeline)
    graph.register_pipeline("bulb_pipeline", bulb_pipeline)

    # La malla llega centrada en el origen; se sube la mitad de su alto para
    # que quede apoyada sobre el plano z = 0, sea cual sea el modelo.
    esquinas = graph.meshes["modelo"]["object"].bounds
    orientacion = configuracion["orientacion"]
    eje_vertical = 2 if modelo == "bunny" else 1
    altura_base = -esquinas[0][eje_vertical] / 2

    def propiedades_de(nombre):
        """Los uniforms del material que el shader en uso declara.

        El shader con textura no tiene `material_diffuse`, porque su color base
        lo entrega el mapa difuso; pasárselo igual aborta el render, ya que los
        atributos de instancia no se validan contra el shader.
        """
        propiedades = {k: v for k, v in MATERIALES[nombre].items() if k != "label"}
        if con_textura:
            propiedades.pop("material_diffuse")
        return propiedades

    material_actual = ORDEN_MATERIALES[0]
    graph.add_object(
        "modelo",
        "modelo",
        "cel_pipeline",
        parent="root",
        transform=tr.translate(0, 0, altura_base) @ orientacion,
        **propiedades_de(material_actual),
    )

    bulb_scale = 0.15
    bulb_1_color = np.array([0.0, 0.8, 1.0], dtype=np.float32)
    bulb_2_color = np.array([1.0, 0.3, 0.0], dtype=np.float32)

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

    camera_position = configuracion["camara"]
    view = tr.lookAt(
        camera_position,
        np.array([0.0, 0.0, configuracion["altura_camara"]]),
        np.array([0.0, 0.0, 1.0]),
    )
    projection = tr.perspective(60, width / height, 0.001, 5.0)

    graph.register_view_transform(view)

    num_bands = 3
    outline_enabled = 1
    outline_threshold = 0.25
    specular_threshold = 0.5

    graph.set_global_attributes(
        projection=projection,
        view_position=camera_position,
        light_1_color=bulb_1_color,
        light_2_color=bulb_2_color,
        ambient_light=np.array([0.25, 0.25, 0.25], dtype=np.float32),
        num_bands=num_bands,
        outline_enabled=outline_enabled,
        outline_threshold=outline_threshold,
        specular_threshold=specular_threshold,
    )

    label_material = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=14,
        x=12,
        y=height - 24,
        color=(255, 255, 255, 255),
    )
    label_bandas = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=14,
        x=12,
        y=height - 46,
        color=(255, 255, 255, 255),
    )
    label_outline = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=14,
        x=12,
        y=height - 68,
        color=(255, 255, 255, 255),
    )
    instrucciones = pyglet.text.Label(
        "1: rojo  2: verde  3: azul     b: bandas (2/3/4)    o: outline on/off"
        "    --modelo bunny|samus",
        font_name="Fira Code",
        font_size=11,
        x=12,
        y=12,
        color=(220, 220, 220, 255),
    )

    def aplicar_material(nombre):
        nonlocal material_actual
        material_actual = nombre
        propiedades = propiedades_de(nombre)
        for clave in graph.nodes:
            if clave.startswith("modelo_mesh"):
                graph.apply_instance_attributes(clave, **propiedades)
        if con_textura:
            label_material.text = "Color base: mapa difuso del modelo"
        else:
            label_material.text = f"Material: {MATERIALES[nombre]['label']}"

    def actualizar_etiquetas():
        label_bandas.text = f"Bandas de difusa: {num_bands}"
        label_outline.text = f"Outline: {'ON' if outline_enabled else 'OFF'}"

    aplicar_material(material_actual)
    actualizar_etiquetas()

    total_time = 0.0

    @window.event
    def on_draw():
        GL.glClearColor(0.95, 0.93, 0.88, 1.0)
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
            label_material.draw()
            label_bandas.draw()
            label_outline.draw()
            instrucciones.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal num_bands, outline_enabled
        teclas_material = {
            pyglet.window.key._1: "rojo",
            pyglet.window.key._2: "verde",
            pyglet.window.key._3: "azul",
        }
        if symbol in teclas_material:
            aplicar_material(teclas_material[symbol])
            return

        if symbol == pyglet.window.key.B:
            num_bands = 2 + (num_bands - 1) % 3  # cicla 2 -> 3 -> 4 -> 2
            graph.set_global_attributes(num_bands=num_bands)
            actualizar_etiquetas()
        elif symbol == pyglet.window.key.O:
            outline_enabled = 1 - outline_enabled
            graph.set_global_attributes(outline_enabled=outline_enabled)
            actualizar_etiquetas()

    def update_world(dt, _):
        nonlocal total_time
        total_time += dt

        graph.nodes["modelo"]["transform"] = (
            tr.translate(0, 0, altura_base)
            @ tr.rotationZ(total_time * 0.5)
            @ orientacion
        )

        radio_base = 0.8

        radio_1 = radio_base + 0.2 * np.sin(total_time * 5.0)
        angulo_1 = total_time * 2.0 + np.sin(total_time * 0.8) * 0.5
        altura_1 = 0.7 + 0.15 * np.sin(total_time * 3.7)
        graph.nodes["bulb_1"]["transform"] = tr.translate(
            radio_1 * np.cos(angulo_1),
            radio_1 * np.sin(angulo_1),
            altura_1,
        ) @ tr.uniformScale(bulb_scale)

        radio_2 = radio_base + 0.15 * np.cos(total_time * 4.3)
        angulo_2 = -total_time * 3.0 + np.pi + np.sin(total_time * 1.2) * 0.7
        rebote = max(0.0, np.sin(total_time * 2.5))
        altura_2 = 0.4 + 0.25 * rebote * rebote
        graph.nodes["bulb_2"]["transform"] = tr.translate(
            radio_2 * np.cos(angulo_2),
            radio_2 * np.sin(angulo_2),
            altura_2,
        ) @ tr.uniformScale(bulb_scale)

        graph.calculate_global_transforms()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
