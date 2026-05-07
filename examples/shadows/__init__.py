import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.framebuffers import create_depth_framebuffer
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_premade import bounding_box_node, rectangle_2d
from grafica.ui import ui_overlay

# Resolución del shadow map. Independiente del tamaño de la ventana: el
# alumno puede achicar la ventana sin degradar las sombras y, al revés,
# subir la calidad cambiando solo este valor.
SHADOW_MAP_SIZE = 2048

# Estado expuesto al teclado. La función apply_lighting() lo propaga al
# grafo (uniforms), a las etiquetas y al stdout.
SHADOW_DEFAULTS = {
    "shadow_bias_min": 0.005,
    "shadow_bias_max": 0.05,
    "pcf_kernel_radius": 1,  # 0 = 1 muestra, 1 = 3x3, 2 = 5x5, 3 = 7x7
    "light_fov_degrees": 90.0,
    "use_front_face_culling": False,
}

SHADOW_STEPS = {
    "shadow_bias_min": 0.001,
    "shadow_bias_max": 0.005,
    "light_fov_degrees": 5.0,
}

SHADOW_RANGES = {
    "shadow_bias_min": (0.0, 0.05),
    "shadow_bias_max": (0.0, 0.2),
    "light_fov_degrees": (20.0, 140.0),
}

# Modos de visualización ciclados con SPACE.
VIEW_MODES = ("camara", "vista desde la luz", "shadow map")


@click.command(
    "shadow_mapping", short_help="Sombras mediante la técnica shadow mapping."
)
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
def shadow_mapping(width, height):
    """Implementación pedagógica de shadow mapping con perillas interactivas."""

    window = pyglet.window.Window(width, height)
    graph = Scenegraph("root")

    # Cargar la fuente FiraCode antes de crear etiquetas. Las fuentes del
    # sistema (monospace, Arial, etc.) no están disponibles en todos los SO.
    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    # -------------------------------------------------------------------------
    # Mallas y pipelines
    # -------------------------------------------------------------------------

    graph.load_and_register_mesh("cornell_box", "assets/CornellBox_original.obj")
    graph.load_and_register_mesh("sphere", "assets/sphere.off")
    graph.load_and_register_mesh(
        "squirtle", "assets/Squirtle.STL", force_color=np.array([255, 20, 160, 255])
    )

    here = Path(os.path.dirname(__file__))

    # Shader para el pass de profundidad (escribe solo gl_FragCoord.z al
    # depth buffer; el fragment shader es trivial).
    graph.load_and_register_pipeline(
        "depth_shader",
        here / "simple_vertex_program.glsl",
        here / "simple_fragment_program.glsl",
    )

    # Shader principal con iluminación difusa + lectura del shadow map.
    graph.load_and_register_pipeline(
        "basic_shader",
        here / "vertex_program.glsl",
        here / "fragment_program.glsl",
    )

    # Shader para visualizar la bombilla (no proyecta sombra).
    graph.load_and_register_pipeline(
        "bulb_pipeline",
        here / ".." / "disco_bunny" / "bulb_vertex_program.glsl",
        here / ".." / "disco_bunny" / "bulb_fragment_program.glsl",
    )

    # Shader screen-space para visualizar el shadow map en pantalla completa.
    # Su geometría ya está en NDC, así que el shader no usa view/projection
    # y se puede meter en el grafo principal sin interferir con la cámara.
    graph.load_and_register_pipeline(
        "quad_view",
        here / "screen_vertex_program.glsl",
        here / "screen_fragment_program.glsl",
    )

    # -------------------------------------------------------------------------
    # Escena
    # -------------------------------------------------------------------------
    # Las mallas que proyectan sombra declaran su pipeline real (basic_shader).
    # Durante el shadow pass se las redirige a depth_shader vía pipeline_override
    # en render(), sin duplicar instancias en el grafo.

    graph.add_mesh_instance("main", "cornell_box", "basic_shader", parent="root")
    graph.add_mesh_instance(
        "pokemon",
        "squirtle",
        "basic_shader",
        transform=tr.uniformScale(0.5),
        parent="main",
    )

    # -------------------------------------------------------------------------
    # Cámaras y jerarquía de la luz
    # -------------------------------------------------------------------------

    near_plane = 0.1
    far_plane = 3.0

    projection_camera = tr.perspective(
        45, float(width) / float(height), near_plane, far_plane
    )
    view_camera = tr.lookAt(
        np.array([0, 0, 2]),
        np.array([0, 0, 0]),
        np.array([0, 1, 0]),
    )

    # Centro de oscilación de la luz. update_world() le suma una perturbación
    # senoidal en x y z para que la sombra se mueva.
    initial_light_position = np.array([0.01, 0.55, -0.02])

    # Estos tres valores se recalculan en update_light_matrices() al cambiar
    # la posición de la luz o el FOV. Se inicializan acá con valores válidos.
    projection_light = tr.perspective(
        SHADOW_DEFAULTS["light_fov_degrees"], 1.0, near_plane, far_plane
    )
    view_light = tr.lookAt(
        initial_light_position,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    )
    light_transform = projection_light @ view_light

    graph.register_view_transform(view_light, name="light_view")
    graph.register_view_transform(view_camera, name="camera_view")

    # La bombilla es un solo nodo bajo root. update_world() le actualiza el
    # transform en cada frame para animarla. Como solo la bombilla usa
    # bulb_pipeline, queda fuera del shadow pass por only_pipelines.
    graph.add_mesh_instance(
        "bulb_mesh",
        "sphere",
        "bulb_pipeline",
        transform=tr.uniformScale(0.1),
        bulb_color=np.array([1.0, 0.3, 0.0]),
        parent="root",
    )
    graph.nodes["bulb_mesh"]["transform"] = tr.translate(*initial_light_position)

    # -------------------------------------------------------------------------
    # FBO de profundidad: el shadow pass renderiza la escena acá.
    # -------------------------------------------------------------------------
    framebuffer, depth_buffer = create_depth_framebuffer(SHADOW_MAP_SIZE)

    # Asociar el shadow map a los nodos que lo van a samplear.
    graph.add_texture_to_node("main", "shadow_map", depth_buffer.id)
    graph.add_texture_to_node("pokemon", "shadow_map", depth_buffer.id)

    # Quad screen-space que muestra el shadow map. Va en el grafo principal
    # bajo root: en el modo "shadow map" se rendea con only_pipelines={"quad_view"}
    # y como su shader no usa view/projection, las matrices de la cámara 3D
    # no lo afectan.
    graph.add_mesh_instance(
        "screen", rectangle_2d(texture=depth_buffer), "quad_view", parent="root"
    )

    # -------------------------------------------------------------------------
    # Overlay 3D: wireframe del frustum de la luz
    # -------------------------------------------------------------------------
    # El frustum de la luz es el cubo NDC ([-1, 1]^3) llevado a world space
    # por inverse(projection_light @ view_light). Definimos un cubo wireframe
    # en NDC y, en cada frame, asignamos esa matriz inversa como su transform.
    # Va en un grafo paralelo al principal porque usa otra pipeline y solo se
    # dibuja en algunos modos de vista.
    overlay_graph = Scenegraph("root")
    overlay_graph.load_and_register_pipeline(
        "line_pipeline",
        here / ".." / "scene_graphs" / "line_vertex_program.glsl",
        here / ".." / "hello_world" / "fragment_program.glsl",
    )
    overlay_graph.add_mesh_instance(
        "light_frustum_instance",
        bounding_box_node(
            np.array([-1, -1, -1]), np.array([1, 1, 1]), color=[1.0, 0.85, 0.2]
        ),
        "line_pipeline",
        parent="root",
    )
    overlay_graph.register_view_transform(view_camera, name="default")
    overlay_graph.set_global_attributes(projection=projection_camera)

    # -------------------------------------------------------------------------
    # Estado interactivo y etiquetas
    # -------------------------------------------------------------------------

    state = dict(SHADOW_DEFAULTS)
    view_mode_index = 0  # 0 camara, 1 vista desde la luz, 2 shadow map
    show_frustum = False  # F: wireframe del frustum de la luz
    total_time = 0.0

    def make_label(y_offset, color=(220, 220, 220, 255)):
        return pyglet.text.Label(
            "",
            font_name="Fira Code",
            font_size=12,
            x=12,
            y=height - y_offset,
            color=color,
        )

    label_bias = make_label(24)
    label_pcf = make_label(46)
    label_fov = make_label(68)
    label_cull = make_label(90)
    label_view = make_label(112)

    label_instructions = pyglet.text.Label(
        "[1/2] bias_min  [3/4] bias_max  [5/6] FOV luz  [7] PCF  [8] cull-front"
        "  [SPACE] vista  [F] frustum  [R] reset",
        font_name="Fira Code",
        font_size=10,
        x=12,
        y=12,
        color=(200, 200, 200, 255),
    )

    def update_light_matrices():
        nonlocal projection_light, view_light, light_transform
        projection_light = tr.perspective(
            state["light_fov_degrees"], 1.0, near_plane, far_plane
        )
        bulb_position = graph.get_global_position("bulb_mesh")
        view_light = tr.lookAt(
            bulb_position,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
        )
        light_transform = projection_light @ view_light
        graph.views["light_view"] = view_light

    def apply_lighting():
        update_light_matrices()
        graph.set_global_attributes(
            shadow_bias_min=float(state["shadow_bias_min"]),
            shadow_bias_max=float(state["shadow_bias_max"]),
            pcf_kernel_radius=int(state["pcf_kernel_radius"]),
        )
        label_bias.text = f"bias min/max:    {state['shadow_bias_min']:.4f} / {state['shadow_bias_max']:.4f}"
        kernel_size = 2 * state["pcf_kernel_radius"] + 1
        label_pcf.text = f"PCF kernel:      {kernel_size}x{kernel_size} ({kernel_size * kernel_size} muestras)"
        label_fov.text = f"FOV luz:         {state['light_fov_degrees']:.1f} deg"
        label_cull.text = (
            f"cull frontal:    {'on' if state['use_front_face_culling'] else 'off'}"
        )
        label_view.text = (
            f"vista:           {VIEW_MODES[view_mode_index]}"
            f"{'  + frustum' if show_frustum else ''}"
        )
        print(
            f"[shadow]"
            f"  bias=[{state['shadow_bias_min']:.4f}, {state['shadow_bias_max']:.4f}]"
            f"  pcf_radius={state['pcf_kernel_radius']}"
            f"  fov={state['light_fov_degrees']:.1f}"
            f"  cull_front={state['use_front_face_culling']}"
            f"  vista={VIEW_MODES[view_mode_index]}"
            f"  frustum={show_frustum}"
        )

    def adjust_param(name, sign):
        step = SHADOW_STEPS[name]
        low, high = SHADOW_RANGES[name]
        state[name] = float(np.clip(state[name] + sign * step, low, high))
        apply_lighting()

    apply_lighting()

    # -------------------------------------------------------------------------
    # Eventos
    # -------------------------------------------------------------------------

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal view_mode_index, show_frustum
        teclas = {
            pyglet.window.key._1: ("shadow_bias_min", -1),
            pyglet.window.key._2: ("shadow_bias_min", +1),
            pyglet.window.key._3: ("shadow_bias_max", -1),
            pyglet.window.key._4: ("shadow_bias_max", +1),
            pyglet.window.key._5: ("light_fov_degrees", -1),
            pyglet.window.key._6: ("light_fov_degrees", +1),
        }
        if symbol in teclas:
            name, sign = teclas[symbol]
            adjust_param(name, sign)
        elif symbol == pyglet.window.key._7:
            state["pcf_kernel_radius"] = (state["pcf_kernel_radius"] + 1) % 4
            apply_lighting()
        elif symbol == pyglet.window.key._8:
            state["use_front_face_culling"] = not state["use_front_face_culling"]
            apply_lighting()
        elif symbol == pyglet.window.key.R:
            state.update(SHADOW_DEFAULTS)
            view_mode_index = 0
            show_frustum = False
            apply_lighting()
        elif symbol == pyglet.window.key.SPACE:
            view_mode_index = (view_mode_index + 1) % len(VIEW_MODES)
            apply_lighting()
        elif symbol == pyglet.window.key.F:
            show_frustum = not show_frustum
            apply_lighting()

    @window.event
    def on_draw():
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # ---------------------------------------------------------------------
        # Pass 1: shadow map. Renderiza la escena desde el punto de vista de
        # la luz, escribiendo solo profundidad en el FBO.
        # ---------------------------------------------------------------------
        framebuffer.bind()
        GL.glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        if state["use_front_face_culling"]:
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_FRONT)

        graph.current_view = "light_view"
        graph.set_global_attributes(projection=projection_light)
        # Las instancias declaran basic_shader como su pipeline natural;
        # acá las redirigimos a depth_shader para esta pasada. La bombilla,
        # registrada con bulb_pipeline, queda fuera por only_pipelines.
        graph.render(
            recalculate_transforms=False,
            only_pipelines={"basic_shader"},
            pipeline_override="depth_shader",
        )

        if state["use_front_face_culling"]:
            GL.glDisable(GL.GL_CULL_FACE)

        framebuffer.unbind()

        # ---------------------------------------------------------------------
        # Pass 2: render final. Tres modos:
        #   0 camara: vista habitual con sombras.
        #   1 vista desde la luz: la escena renderizada usando view_light y
        #     projection_light. Pedagógicamente útil porque todo lo visible
        #     desde aquí queda iluminado (no hay nada en sombra desde la luz).
        #   2 shadow map: el depth buffer crudo en grayscale.
        # ---------------------------------------------------------------------
        GL.glViewport(0, 0, window.width, window.height)
        window.clear()

        if view_mode_index == 2:
            GL.glDisable(GL.GL_DEPTH_TEST)
            graph.render(recalculate_transforms=False, only_pipelines={"quad_view"})
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            if view_mode_index == 0:
                graph.current_view = "camera_view"
                main_projection = projection_camera
            else:
                graph.current_view = "light_view"
                main_projection = projection_light
            graph.set_global_attributes(
                projection=main_projection,
                light_position=graph.get_global_position("bulb_mesh"),
                light_transform=light_transform,
            )
            # Solo los pipelines 3D: quad_view es screen-space y se rendea
            # solo en el modo "shadow map".
            graph.render(
                recalculate_transforms=False,
                only_pipelines={"basic_shader", "bulb_pipeline"},
            )

            # Overlay del frustum solo tiene sentido en la vista de cámara:
            # desde la luz estaríamos mirando por dentro del propio frustum.
            if view_mode_index == 0 and show_frustum:
                overlay_graph.nodes["light_frustum_instance"]["transform"] = (
                    np.linalg.inv(light_transform).astype(np.float32)
                )
                GL.glLineWidth(2.0)
                overlay_graph.render()
                GL.glLineWidth(1.0)

        with ui_overlay():
            label_bias.draw()
            label_pcf.draw()
            label_fov.draw()
            label_cull.draw()
            label_view.draw()
            label_instructions.draw()

    def update_world(dt, window):
        nonlocal total_time
        total_time += dt

        graph.nodes["pokemon"]["transform"] = tr.rotationY(total_time * 0.5)

        graph.nodes["bulb_mesh"]["transform"] = tr.translate(
            initial_light_position[0] + 0.075 * np.sin(total_time * 5.0),
            initial_light_position[1],
            initial_light_position[2] + 0.05 * np.sin(total_time * 3.7),
        )

        graph.calculate_global_transforms()
        update_light_matrices()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run()
