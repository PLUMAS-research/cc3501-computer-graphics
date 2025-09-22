import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
from pathlib import Path
import click


# una función auxiliar para cargar shaders
from grafica.utils import load_pipeline

from grafica.arcball import Arcball
from grafica.textures import texture_2D_setup
from grafica.scenegraph import Scenegraph

# from grafica.scenegraph_nodes import node_from_file
from grafica.scenegraph_premade import rectangle_2d
import grafica.transformations as tr


@click.command("arcball_example", short_help="Visor interactivo de modelos 3D con control de transparencia")
@click.argument("filename", type=str)
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
def arcball_example(filename, width, height):
    window = pyglet.window.Window(width, height)

    graph = Scenegraph("root")
    graph.load_and_register_mesh("object", filename)

    # como no todos los archivos que carguemos tendrán textura,
    # tendremos dos pipelines
    base_path = Path(os.path.dirname(__file__))
    # uno para archivos con textura
    tex_pipeline = load_pipeline(
        base_path / "vertex_program.glsl",
        base_path / "fragment_program.glsl",
    )

    # y otro para archivos sin ella
    notex_pipeline = load_pipeline(
        base_path / "vertex_program_notex.glsl",
        base_path / "fragment_program_notex.glsl",
    )

    # también tenemos un pipeline para ver profundidad!
    depth_pipeline = load_pipeline(
        base_path / "z_vertex_program.glsl",
        base_path / "z_fragment_program.glsl",
    )

    main_pipeline = tex_pipeline if graph.meshes["object"]["has_texture"] else notex_pipeline

    current_pipeline = 0
    pipelines = [main_pipeline, depth_pipeline]
    graph.register_pipeline("pipeline", main_pipeline)

    graph.add_mesh_instance("object", "object", "pipeline")
    graph.add_edge("root", "object")

    # Planos de clipping ajustables
    near_plane = 0.1
    far_plane = 5.0

    projection = tr.perspective(45, float(width) / float(height), near_plane, far_plane)
    view = tr.lookAt(np.array([0, 0, 2]), np.array([0, 0, 0]), np.array([0, 1, 0]))

    graph.register_view_transform(view)

    # instanciamos nuestra Arcball
    arcball = Arcball(
        np.linalg.inv(view),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    # Estados para controlar rendering
    depth_test_enabled = True
    blend_enabled = True
    
    # Modos de blending disponibles
    blend_modes = [
        ("Standard", GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
        ("Premultiplicado", GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA),
        ("Aditivo", GL.GL_SRC_ALPHA, GL.GL_ONE),
        ("Multiplicativo", GL.GL_DST_COLOR, GL.GL_ZERO),
    ]
    current_blend_mode = 0
    
    # Para mostrar texto en pantalla
    batch = pyglet.graphics.Batch()
    status_label = pyglet.text.Label(
        '',
        font_name='Arial',
        font_size=12,
        x=10, y=height - 20,
        anchor_x='left', anchor_y='top',
        color=(255, 255, 255, 255),
        batch=batch
    )
    
    instructions_label = pyglet.text.Label(
        'Controles: [D] Depth test | [B] Blending | [M] Modo blend | [ESPACIO] Pipeline | [R] Reset | [N/F] Near/Far planes',
        font_name='Arial',
        font_size=10,
        x=10, y=20,
        anchor_x='left', anchor_y='bottom',
        color=(200, 200, 200, 255),
        batch=batch
    )
    
    def update_status_text():
        mode_name = blend_modes[current_blend_mode][0]
        pipeline_name = "Profundidad" if current_pipeline else "Normal"
        status_label.text = (
            f'Depth Test: {"ON" if depth_test_enabled else "OFF"} | '
            f'Blending: {"ON" if blend_enabled else "OFF"} | '
            f'Modo: {mode_name} | '
            f'Pipeline: {pipeline_name} | '
            f'Near: {near_plane:.4f} | Far: {far_plane:.1f}'
        )

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        # Botón izquierdo (button=1) para rotación
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        # Botón derecho (button=4) para traslación
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        # Botón central (button=2) para zoom (opcional)
        elif button == pyglet.window.mouse.MIDDLE:
            arcball.set_state(Arcball.STATE_ZOOM)

        arcball.down((x, y))

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        # Opcional: volver al estado de rotación por defecto
        arcball.set_state(Arcball.STATE_ROTATE)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal current_pipeline, depth_test_enabled, blend_enabled, current_blend_mode
        nonlocal near_plane, far_plane, projection
        
        if symbol == pyglet.window.key.SPACE:
            # Cambiar entre pipeline normal y de profundidad
            current_pipeline = not current_pipeline
            graph.register_pipeline('pipeline', pipelines[current_pipeline])
            update_status_text()
            
        elif symbol == pyglet.window.key.D:
            # Activar/desactivar depth test
            depth_test_enabled = not depth_test_enabled
            update_status_text()
            
        elif symbol == pyglet.window.key.B:
            # Activar/desactivar blending
            blend_enabled = not blend_enabled
            update_status_text()
            
        elif symbol == pyglet.window.key.M:
            # Cambiar modo de blending
            if blend_enabled:
                current_blend_mode = (current_blend_mode + 1) % len(blend_modes)
                update_status_text()
                
        elif symbol == pyglet.window.key.R:
            # Reset la vista
            arcball.pose = np.linalg.inv(view)
            
        elif symbol == pyglet.window.key.N:
            # Modificar near plane
            if modifiers & pyglet.window.key.MOD_SHIFT:
                # Shift+N: aumentar near plane (alejar)
                near_plane = min(near_plane * 2.0, 10.0)
            else:
                # N: disminuir near plane (acercar)
                near_plane = max(near_plane * 0.5, 0.00001)
            projection = tr.perspective(45, float(width) / float(height), near_plane, far_plane)
            update_status_text()
            
        elif symbol == pyglet.window.key.F:
            # Modificar far plane
            if modifiers & pyglet.window.key.MOD_SHIFT:
                # Shift+F: disminuir far plane (acercar)
                far_plane = max(far_plane * 0.5, 1.0)
            else:
                # F: aumentar far plane (alejar)
                far_plane = min(far_plane * 2.0, 10000.0)
            projection = tr.perspective(45, float(width) / float(height), near_plane, far_plane)
            update_status_text()

    @window.event
    def on_draw():
        GL.glClearColor(0.2, 0.2, 0.2, 1.0)
        
        # Configurar depth test según el estado
        if depth_test_enabled:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        
        # Configurar blending según el estado
        if blend_enabled:
            GL.glEnable(GL.GL_BLEND)
            # Aplicar el modo de blending actual
            src_factor = blend_modes[current_blend_mode][1]
            dst_factor = blend_modes[current_blend_mode][2]
            GL.glBlendFunc(src_factor, dst_factor)
        else:
            GL.glDisable(GL.GL_BLEND)

        window.clear()

        graph.nodes["root"]["transform"] = np.linalg.inv(arcball.pose)

        graph.set_global_attributes(
            projection=projection, far_plane=far_plane, near_plane=near_plane
        )

        graph.render()
        
        # Dibujar la interfaz de texto
        GL.glDisable(GL.GL_DEPTH_TEST)
        update_status_text()
        batch.draw()

    # Inicializar el texto de estado
    update_status_text()

    pyglet.app.run()