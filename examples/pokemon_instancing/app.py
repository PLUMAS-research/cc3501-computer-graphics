import os.path
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.ui import ui_overlay


SPECIES = [
    {
        "name": "bulbasaur",
        "file": "assets/Bulbasaur.STL",
        "color": np.array([0.35, 0.70, 0.45], dtype=np.float32),
    },
    {
        "name": "charmander",
        "file": "assets/Charmander.STL",
        "color": np.array([0.95, 0.55, 0.20], dtype=np.float32),
    },
    {
        "name": "squirtle",
        "file": "assets/Squirtle.STL",
        "color": np.array([0.30, 0.60, 0.85], dtype=np.float32),
    },
]


@click.command("pokemon_instancing", short_help="Instanciamiento con tres mallas Pokémon")
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=768)
@click.option("--columns", type=int, default=6, help="Columnas en la grilla")
@click.option("--rows", type=int, default=5, help="Filas en la grilla")
def pokemon_instancing(width, height, columns, rows):
    window = pyglet.window.Window(width, height)

    # fuente para el overlay informativo
    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    graph = Scenegraph("world")

    # un único pipeline atiende a las tres especies
    graph.load_and_register_pipeline(
        "pokemon_pipeline",
        Path(os.path.dirname(__file__)) / "pokemon_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl",
    )

    # se carga una malla por especie; todas sus instancias la compartirán en GPU
    for species in SPECIES:
        graph.load_and_register_mesh(species["name"], species["file"])

    # grilla de instancias: cada celda recibe un Pokémon distinto rotando su especie
    spacing = 2.6
    x_offset = -(columns - 1) * spacing / 2
    z_offset = -(rows - 1) * spacing / 2

    instance_keys = []
    for row_index in range(rows):
        for column_index in range(columns):
            flat_index = row_index * columns + column_index
            species = SPECIES[flat_index % len(SPECIES)]

            instance_name = f"pokemon_{flat_index:02d}"
            position = tr.translate(
                x_offset + column_index * spacing,
                0.0,
                z_offset + row_index * spacing,
            )
            graph.add_object(
                instance_name,
                species["name"],
                "pokemon_pipeline",
                parent="world",
                transform=position,
                color=species["color"],
            )
            instance_keys.append((instance_name, flat_index))

    # cámara que muestra toda la grilla en diagonal
    camera_position = np.array([0.0, columns * 1.6, rows * spacing])
    view = tr.lookAt(camera_position, np.array([0, 0, 0]), np.array([0, 1, 0]))
    projection = tr.perspective(45, float(width) / float(height), 0.1, 200)
    graph.register_view_transform(view)
    graph.set_global_attributes(projection=projection)

    total_instances = len(instance_keys)
    unique_meshes = len(SPECIES)
    unique_buffers = graph.unique_gpu_buffers()

    stats_label = pyglet.text.Label(
        text=(
            f"mallas únicas: {unique_meshes} | instancias: {total_instances} | "
            f"vertex_lists en GPU: {unique_buffers}"
        ),
        font_name="Fira Code",
        font_size=13,
        x=16,
        y=height - 28,
        color=(230, 230, 230, 255),
    )

    hint_label = pyglet.text.Label(
        text="sin caché habríamos subido "
        f"{total_instances} buffers; con caché basta con {unique_buffers}",
        font_name="Fira Code",
        font_size=11,
        x=16,
        y=height - 52,
        color=(180, 180, 180, 255),
    )

    total_time = 0.0

    @window.event
    def on_draw():
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        window.clear()
        graph.render()

        # el texto se dibuja en 2D encima de la escena
        with ui_overlay():
            stats_label.draw()
            hint_label.draw()

    def update(dt):
        nonlocal total_time
        total_time += dt

        # cada instancia rota alrededor de su propio eje. La posición la define
        # la transformación del nodo padre; aquí actualizamos sólo la rotación
        # reconstruyendo el producto translate * rotate.
        for instance_name, flat_index in instance_keys:
            row_index, column_index = divmod(flat_index, columns)
            position = tr.translate(
                x_offset + column_index * spacing,
                0.0,
                z_offset + row_index * spacing,
            )
            rotation = tr.rotationY(total_time * 0.8 + flat_index * 0.15)
            graph.nodes[instance_name]["transform"] = position @ rotation

    pyglet.clock.schedule_interval(update, 1 / 60.0)
    pyglet.app.run(1 / 60.0)
