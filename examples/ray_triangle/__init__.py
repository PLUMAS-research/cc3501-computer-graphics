"""Picking por rayos acelerado con un octree, con la estructura a la vista.

Al hacer click, el píxel se des-proyecta a un rayo en el espacio local del
modelo (invirtiendo la matriz MVP completa, ver el apunte de picking). El
rayo se consulta primero contra el octree de los triángulos: las pruebas
rayo-AABB descartan ramas completas y solo los triángulos de las hojas
atravesadas pasan a la prueba exacta de Möller-Trumbore. El panel compara
cuántos triángulos se probaron contra el total de la malla.

La estructura se puede inspeccionar: la tecla O dibuja las cajas de un nivel
del octree (`,` y `.` cambian el nivel; los octantes vacíos no tienen caja,
por eso las cajas abrazan la forma del modelo) y cada click resalta en
amarillo los nodos que el rayo atravesó.

Controles:
- Click izquierdo: lanzar un rayo desde el cursor.
- Arrastre con botón derecho: rotar el modelo.
- O: mostrar/ocultar las cajas del octree.
- , / .: bajar/subir el nivel de octree visible.
- V: modo verificación (compara el resultado contra trimesh).
- C: limpiar el rayo y los resaltados.
- R: reiniciar la rotación.
- ESC: cerrar.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import grafica.transformations as tr
from grafica.intersections import ray_triangle_intersection
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

from .octree import Octree

COLOR_MODELO = (0.7, 0.7, 0.8)
COLOR_NIVEL = np.array([0.3, 0.45, 0.6])
COLOR_VISITADOS = np.array([1.0, 0.8, 0.2])
COLOR_RAYO = np.array([0.2, 0.9, 0.9])
COLOR_TRIANGULO = np.array([1.0, 0.0, 1.0])
COLOR_PUNTO = np.array([1.0, 0.0, 0.0])

# 12 aristas de una caja, como pares de índices sobre sus 8 esquinas.
ARISTAS_CAJA = np.array([
    0, 1, 1, 2, 2, 3, 3, 0,
    4, 5, 5, 6, 6, 7, 7, 4,
    0, 4, 1, 5, 2, 6, 3, 7,
])


def lineas_de_caja(minimo, maximo):
    """Vértices (24, 3) de las aristas de un AABB, listos para GL_LINES."""
    x0, y0, z0 = minimo
    x1, y1, z1 = maximo
    esquinas = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    return esquinas[ARISTAS_CAJA]


@click.command("ray_triangle", short_help="Picking rayo-triángulo acelerado con octree")
@click.argument("filename", type=str, default="assets/bunny.obj", required=False)
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
@click.option("--capacidad", type=int, default=64, help="Triángulos por hoja del octree")
@click.option("--nivel-maximo", type=int, default=6, help="Profundidad máxima del octree")
def ray_triangle_example(filename, width, height, capacidad, nivel_maximo):
    window = pyglet.window.Window(width, height, caption="picking con octree")
    shader_dir = Path(__file__).parent

    # ---- malla ----
    mesh = tm.load(filename)
    if hasattr(mesh, "geometry"):
        mesh = list(mesh.geometry.values())[0]
    mesh.fix_normals()
    # centrar en el origen y escalar a tamaño unitario.
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(2.3 / mesh.scale)

    total_triangulos = len(mesh.faces)
    print(f"[octree] malla: {len(mesh.vertices)} vértices, {total_triangulos} caras")

    # ---- octree ----
    octree = Octree(
        mesh.vertices, mesh.faces, capacidad=capacidad, nivel_maximo=nivel_maximo
    )
    niveles = sorted(octree.nodos_por_nivel)
    total_nodos = sum(len(nodos) for nodos in octree.nodos_por_nivel.values())
    print(
        f"[octree] {total_nodos} nodos, {octree.cantidad_hojas} hojas, "
        f"niveles 0..{niveles[-1]}"
    )

    # ---- shaders y geometría en GPU ----
    model_shader = load_pipeline(shader_dir / "mesh_vp.glsl", shader_dir / "mesh_fp.glsl")
    debug_shader = load_pipeline(shader_dir / "debug_vp.glsl", shader_dir / "debug_fp.glsl")

    model_vertex_list = model_shader.vertex_list_indexed(
        len(mesh.vertices),
        GL.GL_TRIANGLES,
        mesh.faces.ravel(),
        position=("f", mesh.vertices.ravel()),
        normal=("f", mesh.vertex_normals.ravel()),
    )

    def nueva_lista(posiciones, color, primitiva=GL.GL_LINES):
        """vertex_list del shader de debug con un color uniforme por vértice."""
        posiciones = np.asarray(posiciones, dtype=np.float32).reshape(-1, 3)
        return debug_shader.vertex_list(
            len(posiciones),
            primitiva,
            position=("f", posiciones.ravel()),
            color=("f", np.tile(color, len(posiciones)).astype(np.float32)),
        )

    # las cajas de cada nivel del octree no cambian: se suben una sola vez.
    cajas_por_nivel = {
        nivel: nueva_lista(
            np.concatenate([lineas_de_caja(n.minimo, n.maximo) for n in nodos]),
            COLOR_NIVEL,
        )
        for nivel, nodos in octree.nodos_por_nivel.items()
    }

    # ---- cámara ----
    projection = tr.perspective(45.0, width / height, 0.1, 100.0)
    view = tr.lookAt(np.array([0, 0, 3]), np.array([0, 0, 0]), np.array([0, 1, 0]))

    # ---- estado ----
    state = {
        "rotation": np.eye(4),
        "mostrar_cajas": True,
        "nivel": min(2, niveles[-1]),
        "verificar": False,
        # vertex_lists creados por el último click (se borran al reemplazar).
        "rayo": None,
        "triangulo": None,
        "punto": None,
        "visitados": None,
    }

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(20, 20, 20), background_width=560)
        .add("octree").add("nivel").add("consulta").add("resultado")
        .footer("click rayo   arrastre der rotar   O cajas   , . nivel   V verificar   C limpiar   R reset")
    )
    panel["octree"] = (
        f"octree: {total_nodos} nodos, {octree.cantidad_hojas} hojas "
        f"({total_triangulos} triángulos)"
    )
    panel["consulta"] = "haz click para lanzar un rayo"
    panel["resultado"] = ""

    def apply_state():
        if state["mostrar_cajas"]:
            cantidad = len(octree.nodos_por_nivel[state["nivel"]])
            panel["nivel"] = f"nivel visible: {state['nivel']} ({cantidad} cajas)"
        else:
            panel["nivel"] = "cajas del octree ocultas (O las muestra)"
        print(
            f"[octree] cajas={'sí' if state['mostrar_cajas'] else 'no'} "
            f"nivel={state['nivel']} verificar={state['verificar']}"
        )

    def limpiar_consulta():
        for clave in ("rayo", "triangulo", "punto", "visitados"):
            if state[clave] is not None:
                state[clave].delete()
                state[clave] = None

    # ---- picking ----
    def rayo_desde_pantalla(x, y):
        """Convierte un píxel en un rayo en el espacio local del modelo.

        Los dos extremos del rayo en NDC (planos near y far) se transforman
        con la inversa de la MVP completa: así el rayo queda en el mismo
        espacio que los vértices de la malla y el octree, sin transformar
        la geometría.
        """
        ndc_x = (2.0 * x / width) - 1.0
        ndc_y = (2.0 * y / height) - 1.0

        mvp = projection @ view @ state["rotation"]
        inversa = np.linalg.inv(mvp)

        cercano = inversa @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        lejano = inversa @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        cercano /= cercano[3]
        lejano /= lejano[3]

        origen = cercano[:3]
        direccion = lejano[:3] - cercano[:3]
        return origen, direccion / np.linalg.norm(direccion)

    def interseccion_mas_cercana(origen, direccion, candidatos):
        """Möller-Trumbore sobre los candidatos; devuelve la cara más cercana."""
        mejor_t = np.inf
        mejor_cara = -1
        mejor_punto = None
        for indice_cara in candidatos:
            v0, v1, v2 = mesh.vertices[mesh.faces[indice_cara]]
            hay, t, u, v = ray_triangle_intersection(origen, direccion, v0, v1, v2)
            if hay and t < mejor_t:
                mejor_t = t
                mejor_cara = indice_cara
                mejor_punto = (1.0 - u - v) * v0 + u * v1 + v * v2
        return mejor_cara, mejor_punto, mejor_t

    def verificar_con_trimesh(origen, direccion, cara_propia):
        posiciones, _, caras = mesh.ray.intersects_location(
            ray_origins=[origen], ray_directions=[direccion]
        )
        if len(caras) == 0:
            print("[verificación] trimesh tampoco encontró impacto"
                  if cara_propia < 0 else
                  "[verificación] ADVERTENCIA: trimesh no encontró el impacto")
            return
        distancias = np.linalg.norm(posiciones - origen, axis=1)
        cara_trimesh = caras[np.argmin(distancias)]
        coincide = "coincide" if cara_trimesh == cara_propia else "NO COINCIDE"
        print(f"[verificación] trimesh: cara {cara_trimesh} ({coincide})")

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button != pyglet.window.mouse.LEFT:
            return
        origen, direccion = rayo_desde_pantalla(x, y)

        # fase ancha: el octree descarta ramas con pruebas rayo-AABB.
        candidatos, nodos_visitados, cajas_probadas = octree.consultar_rayo(
            origen, direccion
        )
        # fase exacta: Möller-Trumbore solo sobre los candidatos.
        cara, punto, distancia = interseccion_mas_cercana(origen, direccion, candidatos)

        limpiar_consulta()
        state["rayo"] = nueva_lista([origen, origen + direccion * 5.0], COLOR_RAYO)
        if nodos_visitados:
            state["visitados"] = nueva_lista(
                np.concatenate(
                    [lineas_de_caja(n.minimo, n.maximo) for n in nodos_visitados]
                ),
                COLOR_VISITADOS,
            )
        if cara >= 0:
            v0, v1, v2 = mesh.vertices[mesh.faces[cara]]
            state["triangulo"] = nueva_lista([v0, v1, v1, v2, v2, v0], COLOR_TRIANGULO)
            state["punto"] = nueva_lista([punto], COLOR_PUNTO, GL.GL_POINTS)

        porcentaje = 100.0 * len(candidatos) / total_triangulos
        panel["consulta"] = (
            f"rayo: {cajas_probadas} cajas probadas, "
            f"{len(candidatos)} triángulos de {total_triangulos} ({porcentaje:.1f}%)"
        )
        panel["resultado"] = (
            f"impacto en la cara {cara}, distancia {distancia:.3f}"
            if cara >= 0 else "sin impacto"
        )
        print(
            f"[octree] cajas probadas: {cajas_probadas}, "
            f"candidatos: {len(candidatos)} de {total_triangulos} ({porcentaje:.1f}%), "
            + (f"impacto en cara {cara}" if cara >= 0 else "sin impacto")
        )
        if state["verificar"]:
            verificar_con_trimesh(origen, direccion, cara)

    # ---- rotación del modelo ----
    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if not buttons & pyglet.window.mouse.RIGHT:
            return
        rotacion = tr.rotationX(-dy * 0.01) @ tr.rotationY(dx * 0.01) @ state["rotation"]
        # re-ortogonalizar para que el drift numérico no deforme el modelo.
        u, _, vt = np.linalg.svd(rotacion[:3, :3])
        rotacion[:3, :3] = u @ vt
        state["rotation"] = rotacion

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.O:
            state["mostrar_cajas"] = not state["mostrar_cajas"]
            apply_state()
        elif symbol == pyglet.window.key.COMMA:
            state["nivel"] = max(state["nivel"] - 1, niveles[0])
            apply_state()
        elif symbol == pyglet.window.key.PERIOD:
            state["nivel"] = min(state["nivel"] + 1, niveles[-1])
            apply_state()
        elif symbol == pyglet.window.key.V:
            state["verificar"] = not state["verificar"]
            apply_state()
        elif symbol == pyglet.window.key.C:
            limpiar_consulta()
            panel["consulta"] = "haz click para lanzar un rayo"
            panel["resultado"] = ""
        elif symbol == pyglet.window.key.R:
            state["rotation"] = np.eye(4)
            apply_state()
        elif symbol == pyglet.window.key.ESCAPE:
            window.close()

    # ---- render ----
    @window.event
    def on_draw():
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPointSize(10)
        window.clear()

        mvp = projection @ view @ state["rotation"]

        model_shader.use()
        model_shader["mvp"] = mvp.astype(np.float32).flatten("F")
        model_shader["model"] = state["rotation"].astype(np.float32).flatten("F")
        model_shader["color"] = COLOR_MODELO
        model_shader["lightPos"] = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        model_shader["viewPos"] = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        model_vertex_list.draw(GL.GL_TRIANGLES)

        # los overlays están en el espacio local del modelo, igual que el
        # rayo: comparten la MVP de la malla y rotan con ella.
        debug_shader.use()
        debug_shader["mvp"] = mvp.astype(np.float32).flatten("F")
        if state["mostrar_cajas"]:
            cajas_por_nivel[state["nivel"]].draw(GL.GL_LINES)
        if state["visitados"] is not None:
            state["visitados"].draw(GL.GL_LINES)
        if state["rayo"] is not None:
            state["rayo"].draw(GL.GL_LINES)
        if state["triangulo"] is not None:
            state["triangulo"].draw(GL.GL_LINES)
        if state["punto"] is not None:
            state["punto"].draw(GL.GL_POINTS)

        with ui_overlay():
            panel.draw()

    apply_state()
    pyglet.app.run()


if __name__ == "__main__":
    ray_triangle_example()
