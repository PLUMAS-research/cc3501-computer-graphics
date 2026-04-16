import numpy as np
import trimesh as tm
import grafica.transformations as tr
from itertools import chain
import pyglet.gl as GL

def rectangulo():
    vertices = np.array(
        [
            -1,
            -1,
            0.0,  # inf izq
            1,
            -1,
            0.0,  # if der
            1,
            1,
            0.0,  # sup der
            -1,
            1,
            0.0,  # sup izq
        ],
        dtype=np.float32,
    )

    # Gradiente sutil: azul marino en la parte inferior, violeta en la superior
    vertex_colors = np.array(
        [
            0.06, 0.08, 0.25,  # inf izq  — azul marino
            0.06, 0.08, 0.25,  # inf der  — azul marino
            0.26, 0.08, 0.38,  # sup der  — violeta
            0.26, 0.08, 0.38,  # sup izq  — violeta
        ],
        dtype=np.float32,
    )

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    return {
        "position": vertices,
        "color": vertex_colors,
        "indices": indices,
        "n_vertices": 4,
        'gl_type': GL.GL_TRIANGLES
    }


def stanford_bunny():
    bunny = tm.load("assets/Stanford_Bunny.stl")

    # model transform del conejo. la aplicamos directamente en trimesh
    # noten que esta vez solamente escalamos al conejo, ¡no lo estamos rotando!
    bunny_scale = tr.uniformScale(1.0 / bunny.scale)
    bunny_translate = tr.translate(*-bunny.centroid)
    bunny.apply_transform(bunny_scale @ bunny_translate)
    # el conejo ya está transformado. pero lo movimos al origen,
    # cuando en realidad queremos que esté sobre el suelo
    # con esto dejamos la parte baja del conejo en z = 0
    # asumiento que z apunta hacia arriba en nuestro mundo
    bunny.apply_transform(tr.translate(0, 0, -bunny.vertices[:, 2].min()))

    bunny_vertex_list = tm.rendering.mesh_to_vertexlist(bunny)

    # Normales por vértice de cara, expandidas en el mismo orden que las posiciones.
    # bunny.faces tiene forma (n_caras, 3), así que vertex_normals[faces] tiene
    # forma (n_caras, 3, 3) → reshape(-1, 3) da (n_caras*3, 3).
    vertex_normals_expanded = bunny.vertex_normals[bunny.faces].reshape(-1, 3)

    return {
        "mesh": bunny,
        "position": bunny_vertex_list[4][1],
        "n_vertices": len(bunny_vertex_list[4][1]) // 3,
        "indices": bunny_vertex_list[3],
        "normal": vertex_normals_expanded.flatten().astype(np.float32),
        'gl_type': GL.GL_TRIANGLES
    }

def regular_grid(resolution=10):
    # construimos nuestra grilla.
    xv, yv = np.meshgrid(
        np.linspace(0, 1, resolution),
        np.linspace(0, 1, resolution),
        indexing="xy",
    )

    vertices = np.vstack(
        (
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            np.zeros(shape=(1, resolution**2)),
        )
    ).T

    indices = [
        [
            (resolution * row + i, resolution * row + i + 1)
            for i in range(resolution - 1)
        ]
        for row in range(resolution)
    ]
    
    indices.extend(
        [
            [
                (
                    resolution * column + i,
                    resolution * column + i + resolution,
                )
                for i in range(resolution)
            ]
            for column in range(resolution - 1)
        ]
    )
    
    indices = list(chain(*chain(*indices)))

    return {
        'position': vertices.reshape(-1, 1, order="C"),
        'indices': indices,
        'n_vertices': resolution**2,
        'gl_type': GL.GL_LINES
    }