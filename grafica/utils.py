import numpy as np
import pyglet
import trimesh as tm


def load_pipeline(vertex_path, fragment_path):
    with open(vertex_path) as f:
        vertex_source_code = f.read()

    with open(fragment_path) as f:
        fragment_source_code = f.read()

    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    return pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)


def load_single_mesh(path):
    """Carga el archivo como una sola malla de triángulos, sin texturas.

    Junta todas las partes de la escena con sus transformaciones, suelda los
    vértices que comparten posición y quita las caras degeneradas y repetidas.
    En un OBJ, cada costura de las coordenadas de textura duplica los vértices
    de sus dos lados: sin soldarlos, la malla queda llena de bordes. Las partes
    se juntan sin sus texturas porque, al fundirlas con ellas, trimesh arma un
    atlas con todas sus imágenes.
    """
    partes = [parte for parte in tm.load(path, force="scene").dump() if hasattr(parte, "faces")]
    inicio = np.cumsum([0] + [len(parte.vertices) for parte in partes])
    mesh = tm.Trimesh(
        np.concatenate([parte.vertices for parte in partes]),
        np.concatenate([parte.faces + desde for parte, desde in zip(partes, inicio)]),
        process=False,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh
