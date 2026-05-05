import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_premade import rectangle_2d
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
    "pcf_kernel_radius": 1,        # 0 = 1 muestra, 1 = 3x3, 2 = 5x5, 3 = 7x7
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

# Wireframe del frustum de la luz: 8 vertices del cubo NDC + 12 aristas. Al
# multiplicar las posiciones por inverse(projection_light @ view_light) en
# el vertex shader (vía la matriz transform de la instancia), las esquinas
# del cubo NDC quedan ubicadas en world space y dibujan el volumen visible
# desde la luz.
NDC_CUBE_CORNERS = np.array(
    [
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ],
    dtype=np.float32,
)
FRUSTUM_EDGE_INDICES = np.array(
    [
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7,
    ],
    dtype=np.uint32,
)


def _frustum_mesh(color):
    return {
        "mesh": {"n_vertices": 8, "texture": None, "textures": {}},
        "attributes": {
            "position": NDC_CUBE_CORNERS.flatten(),
            "color": np.tile(np.asarray(color, dtype=np.float32), 8),
        },
        "indices": FRUSTUM_EDGE_INDICES,
        "GL_TYPE": GL.GL_LINES,
        "transform": tr.identity(),
        "id": None,
        "children": [],
        "parent": None,
        "object": None,
        "has_texture": False,
    }


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
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
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

    # Pipeline "current_shader" se intercambia en cada pass: apunta al depth
    # shader durante el shadow pass y al basic shader durante el pass final.
    graph.register_pipeline("current_shader", graph.pipelines["basic_shader"])

    # -------------------------------------------------------------------------
    # Escena
    # -------------------------------------------------------------------------

    graph.add_mesh_instance("main", "cornell_box", "current_shader")
    graph.add_edge("root", "main")

    graph.add_mesh_instance(
        "pokemon", "squirtle", "current_shader", transform=tr.uniformScale(0.5)
    )
    graph.add_edge("main", "pokemon")

    graph.add_mesh_instance(
        "bulb_mesh",
        "sphere",
        "bulb_pipeline",
        transform=tr.uniformScale(0.1),
        bulb_color=np.array([1.0, 0.3, 0.0]),
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

    initial_light_position = np.array([0.0, 0.55, -0.03])

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

    graph.add_node("bulb", transform=tr.translate(*initial_light_position))
    graph.add_transform("bulb_perturbation", tr.identity())
    graph.add_edge("root", "bulb")
    graph.add_edge("bulb", "bulb_perturbation")
    graph.add_edge("bulb_perturbation", "bulb_mesh")

    # -------------------------------------------------------------------------
    # FBO con solo depth attachment
    # -------------------------------------------------------------------------

    depth_buffer = pyglet.image.Texture.create(
        SHADOW_MAP_SIZE,
        SHADOW_MAP_SIZE,
        internalformat=GL.GL_DEPTH_COMPONENT32,
        fmt=GL.GL_DEPTH_COMPONENT,
        min_filter=GL.GL_LINEAR,
        mag_filter=GL.GL_LINEAR,
    )

    GL.glBindTexture(GL.GL_TEXTURE_2D, depth_buffer.id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
    # color de borde 1.0: cualquier fragmento fuera del frustum de la luz
    # consulta una "profundidad máxima" y por lo tanto nunca queda en sombra.
    border_color = (GL.GLfloat * 4)(1.0, 1.0, 1.0, 1.0)
    GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, border_color)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    framebuffer = pyglet.image.Framebuffer()
    framebuffer.attach_texture(depth_buffer, attachment=GL.GL_DEPTH_ATTACHMENT)

    # FBO sin color attachment: hay que indicarle al driver que no se va a
    # escribir ni leer color. Si no, el FBO queda incompleto en algunos
    # drivers conformantes.
    framebuffer.bind()
    GL.glDrawBuffer(GL.GL_NONE)
    GL.glReadBuffer(GL.GL_NONE)
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        print(f"[shadows] framebuffer incompleto: {status}")
    framebuffer.unbind()

    # Asociar el shadow map a los nodos que lo van a samplear.
    graph.add_texture_to_node("main", "shadow_map", depth_buffer.id)
    graph.add_texture_to_node("pokemon", "shadow_map", depth_buffer.id)

    # -------------------------------------------------------------------------
    # Vista de debug del shadow map (escena auxiliar)
    # -------------------------------------------------------------------------

    fbo_scene = Scenegraph("root")
    fbo_scene.load_and_register_pipeline(
        "quad_view",
        here / "screen_vertex_program.glsl",
        here / "screen_fragment_program.glsl",
    )
    fbo_scene.register_mesh("quad", rectangle_2d(texture=depth_buffer))
    fbo_scene.add_mesh_instance("screen", "quad", "quad_view")
    fbo_scene.add_edge("root", "screen")

    # -------------------------------------------------------------------------
    # Overlay 3D: wireframe del frustum de la luz
    # -------------------------------------------------------------------------
    # Grafo paralelo al principal (no subarbol) para mantener la escena
    # limpia: el overlay usa su propia pipeline de líneas y se renderiza
    # después del pass principal cuando corresponde.
    overlay_graph = Scenegraph("root")
    overlay_graph.load_and_register_pipeline(
        "line_pipeline",
        here / ".." / "scene_graphs" / "line_vertex_program.glsl",
        here / ".." / "hello_world" / "fragment_program.glsl",
    )
    overlay_graph.register_mesh("light_frustum", _frustum_mesh([1.0, 0.85, 0.2]))
    overlay_graph.add_mesh_instance(
        "light_frustum_instance", "light_frustum", "line_pipeline"
    )
    overlay_graph.add_edge("root", "light_frustum_instance")
    overlay_graph.register_view_transform(view_camera, name="default")
    overlay_graph.set_global_attributes(projection=projection_camera)

    # -------------------------------------------------------------------------
    # Estado interactivo y etiquetas
    # -------------------------------------------------------------------------

    state = dict(SHADOW_DEFAULTS)
    view_mode_index = 0     # 0 camara, 1 vista desde la luz, 2 shadow map
    show_frustum = False    # F: wireframe del frustum de la luz
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
        bulb_position = graph.get_global_position("bulb_perturbation")
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
        label_bias.text = (
            f"bias min/max:    {state['shadow_bias_min']:.4f} / {state['shadow_bias_max']:.4f}"
        )
        kernel_size = 2 * state["pcf_kernel_radius"] + 1
        label_pcf.text = (
            f"PCF kernel:      {kernel_size}x{kernel_size} ({kernel_size * kernel_size} muestras)"
        )
        label_fov.text = (
            f"FOV luz:         {state['light_fov_degrees']:.1f} deg"
        )
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

        graph.pipelines["current_shader"] = graph.pipelines["depth_shader"]
        graph.current_view = "light_view"
        graph.set_global_attributes(projection=projection_light)
        # Solo los objetos de la escena que usan current_shader proyectan
        # sombra. La bombilla, que usa bulb_pipeline, queda fuera.
        graph.render(recalculate_transforms=False, only_pipelines={"current_shader"})

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
            fbo_scene.render()
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            graph.pipelines["current_shader"] = graph.pipelines["basic_shader"]
            if view_mode_index == 0:
                graph.current_view = "camera_view"
                main_projection = projection_camera
            else:
                graph.current_view = "light_view"
                main_projection = projection_light
            graph.set_global_attributes(
                projection=main_projection,
                light_position=graph.get_global_position("bulb_perturbation"),
                light_transform=light_transform,
            )
            graph.render(recalculate_transforms=False)

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

        base = 0.01
        graph.nodes["bulb_perturbation"]["transform"] = tr.translate(
            base + 0.075 * np.sin(total_time * 5.0),
            0,
            base + 0.05 * np.sin(total_time * 3.7),
        )

        graph.calculate_global_transforms()
        update_light_matrices()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run()
