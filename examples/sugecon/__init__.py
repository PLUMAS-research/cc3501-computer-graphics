import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
import click
from pathlib import Path

from grafica.utils import load_pipeline
from grafica.arcball import Arcball
import grafica.transformations as tr

from .curvature import (
    compute_mean_curvature,
    compute_gaussian_curvature,
    compute_principal_curvatures,
    approximate_principal_directions,
    smooth_vertex_attribute
)
from .contours import (
    compute_radial_curvature,
    extract_contour_points,
    extract_silhouette_edges,
    compute_radial_curvature_derivative
)


@click.command("suggestive_contours", short_help="Suggestive Contours con pyglet")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
@click.option("--model", type=str)
def suggestive_contours(width, height, model):
    print(f"Cargando {model}...")
    mesh = tm.load(model)

    print("Normalizando geometría...")
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(2.0 / mesh.scale)
    mesh.fix_normals()

    print(f"Malla: {len(mesh.vertices)} vértices, {len(mesh.faces)} caras")

    print("Calculando curvaturas...")
    H = compute_mean_curvature(mesh)
    K = compute_gaussian_curvature(mesh)
    k1, k2 = compute_principal_curvatures(H, K)
    d1, d2 = approximate_principal_directions(mesh, k1, k2)

    print("Suavizando curvaturas...")
    k1 = smooth_vertex_attribute(mesh, k1, iterations=2, lambda_factor=0.3)
    k2 = smooth_vertex_attribute(mesh, k2, iterations=2, lambda_factor=0.3)

    print(f"  H: min={H.min():.4f}, max={H.max():.4f}")
    print(f"  K: min={K.min():.4f}, max={K.max():.4f}")
    print(f"  k1: min={k1.min():.4f}, max={k1.max():.4f}")
    print(f"  k2: min={k2.min():.4f}, max={k2.max():.4f}")

    print("Precalculando derivadas de curvatura radial (usar posición inicial)...")
    initial_camera_pos = np.array([0, 0, 3])
    kr_initial = compute_radial_curvature(mesh, initial_camera_pos, k1, k2, d1, d2)
    dkr = compute_radial_curvature_derivative(mesh, initial_camera_pos, k1, k2, d1, d2, kr_initial)

    window = pyglet.window.Window(width, height)

    base_path = Path(os.path.dirname(__file__))
    mesh_pipeline = load_pipeline(
        base_path / "mesh_vertex.glsl", base_path / "mesh_fragment.glsl"
    )
    contour_pipeline = load_pipeline(
        base_path / "contour_vertex.glsl", base_path / "contour_fragment.glsl"
    )

    mesh_vertex_list = tm.rendering.mesh_to_vertexlist(mesh)
    num_vertices = len(mesh_vertex_list[4][1]) // 3
    mesh_gpu = mesh_pipeline.vertex_list_indexed(
        num_vertices,
        GL.GL_TRIANGLES,
        mesh_vertex_list[3],
        position=('f', mesh_vertex_list[4][1]),
        normal=('f', mesh_vertex_list[5][1])
    )

    near_plane = 0.1
    far_plane = 10.0
    projection = tr.perspective(60, float(width) / float(height), near_plane, far_plane)
    view = tr.lookAt(np.array([0, 0, 3]), np.array([0, 0, 0]), np.array([0, 1, 0]))

    arcball = Arcball(
        np.linalg.inv(view),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    show_mesh = True
    show_contours = True
    show_silhouettes = True
    contour_threshold = 1e-6

    contour_gpu = None

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        arcball.down((x, y))

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal show_mesh, show_contours, show_silhouettes

        if symbol == pyglet.window.key.M:
            show_mesh = not show_mesh
            print(f"Malla: {'ON' if show_mesh else 'OFF'}")

        elif symbol == pyglet.window.key.C:
            show_contours = not show_contours
            print(f"Contornos: {'ON' if show_contours else 'OFF'}")

        elif symbol == pyglet.window.key.R:
            arcball.reset()
            print("Cámara reiniciada")

        elif symbol == pyglet.window.key.S:
            show_silhouettes = not show_silhouettes
            print(f"Siluetas: {'ON' if show_silhouettes else 'OFF'}")

    @window.event
    def on_draw():
        nonlocal contour_gpu

        GL.glClearColor(0.9, 0.9, 0.9, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        current_view = np.linalg.inv(arcball.pose)

        if show_mesh:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            
            mesh_pipeline.use()
            mesh_pipeline['transform'] = np.eye(4, dtype=np.float32).reshape(16, 1, order="F")
            mesh_pipeline['view'] = current_view.astype(np.float32).reshape(16, 1, order="F")
            mesh_pipeline['projection'] = projection.reshape(16, 1, order="F")
            mesh_pipeline['color'] = (0.8, 0.8, 0.9)
            mesh_pipeline['lightDir'] = (0.0, 0.7, 0.7)
            mesh_gpu.draw(GL.GL_TRIANGLES)

        camera_pos = arcball.pose[:3, 3]

        if show_contours:
            camera_pos = arcball.pose[:3, 3]
            kr = compute_radial_curvature(mesh, camera_pos, k1, k2, d1, d2)
            contour_points, contour_edges, contour_kr_values = extract_contour_points(mesh, kr, dkr, contour_threshold)
            
            if len(contour_points) > 0 and len(contour_edges) > 0:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
                
                max_kr = 0.05
                alphas = 1.0 - np.clip(contour_kr_values / max_kr, 0, 1)
                alphas = alphas.astype(np.float32)
                
                line_indices = np.array(contour_edges, dtype=np.uint32).flatten()
                
                contour_gpu = contour_pipeline.vertex_list_indexed(
                    len(contour_points),
                    GL.GL_LINES,
                    line_indices,
                    position=('f', contour_points.flatten()),
                    alpha=('f', alphas)
                )
                
                GL.glLineWidth(2.5)
                contour_pipeline.use()
                contour_pipeline['transform'] = np.eye(4, dtype=np.float32).reshape(16, 1, order="F")
                contour_pipeline['view'] = current_view.astype(np.float32).reshape(16, 1, order="F")
                contour_pipeline['projection'] = projection.reshape(16, 1, order="F")
                contour_pipeline['color'] = (0.25, 0.25, 0.25)
                contour_gpu.draw(GL.GL_LINES)
                contour_gpu.delete()
                
                GL.glDisable(GL.GL_BLEND)
                GL.glDisable(GL.GL_LINE_SMOOTH)

        if show_silhouettes:
            silhouette_points, silhouette_edges = extract_silhouette_edges(
                mesh, camera_pos
            )

            if len(silhouette_points) > 0 and len(silhouette_edges) > 0:
                line_indices = np.array(silhouette_edges, dtype=np.uint32).flatten()
                silhouette_gpu = contour_pipeline.vertex_list_indexed(
                    len(silhouette_points), GL.GL_LINES, line_indices
                )
                silhouette_gpu.position[:] = silhouette_points.flatten()

                GL.glLineWidth(6.)
                contour_pipeline.use()
                contour_pipeline["transform"] = np.eye(4, dtype=np.float32).reshape(
                    16, 1, order="F"
                )
                contour_pipeline["view"] = current_view.astype(np.float32).reshape(
                    16, 1, order="F"
                )
                contour_pipeline["projection"] = projection.reshape(16, 1, order="F")
                contour_pipeline["color"] = (0.2, 0.2, 0.2)
                silhouette_gpu.draw(GL.GL_LINES)
                silhouette_gpu.delete()

    print("\nControles:")
    print("  Mouse izquierdo: Rotar")
    print("  Mouse derecho: Pan")
    print("  Scroll: Zoom")
    print("  M: Toggle malla")
    print("  C: Toggle contornos")
    print("  R: Reset cámara")
    print("  S: Toggle siluetas")

    pyglet.app.run()

