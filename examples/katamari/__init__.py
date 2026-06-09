"""Katamari de conejos: detección de colisiones en una escena 3D.

Una bola rueda por un campo lleno de conejos de distintos tamaños. Cada
frame se buscan colisiones entre la bola y los objetos en dos fases:

- Fase ancha: una grilla de hash espacial sobre el plano XZ descarta de
  inmediato los objetos lejanos. Solo se revisan las celdas que toca la
  esfera de la bola (tecla G para ver la grilla y las celdas consultadas,
  tecla B para compararla contra fuerza bruta).
- Fase angosta: prueba esfera-esfera entre la bola y cada candidato. Cada
  objeto usa su esfera contenedora, precalculada de la malla (tecla V para
  verlas: verde si la bola ya puede recogerlo, rojo si todavía no).

La respuesta depende del tamaño relativo: un objeto más chico que la bola
se adhiere (su nodo cambia de padre en el grafo de escena y desde entonces
rueda con ella) y la bola crece conservando el volumen sumado; un objeto
más grande la bloquea (se corrige la penetración empujando la bola hacia
afuera).

Controles:
- Flechas: girar y avanzar/retroceder.
- V: mostrar/ocultar esferas contenedoras.
- G: mostrar/ocultar la grilla de hash y las celdas consultadas.
- B: alternar fase ancha por grilla / fuerza bruta.
- R: reiniciar.
- ESC: cerrar.
"""

import colorsys

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_nodes import _node_from_mesh
from grafica.scenegraph_premade import ring_node
from grafica.ui import InfoPanel, ui_overlay

from .colisiones import GrillaHashEspacial, esferas_se_intersectan

# campo de juego: un cuadrado de lado 2 * MEDIO_CAMPO en el plano XZ.
MEDIO_CAMPO = 16.0
LADO_CELDA = 2.0

# la bola recoge objetos cuyo radio sea a lo más este factor de su radio.
UMBRAL_RECOGIDA = 0.9
# fracción del volumen del objeto recogido que gana la bola.
GANANCIA_VOLUMEN = 0.8
RADIO_INICIAL = 0.4

VELOCIDAD_GIRO = 2.2  # radianes por segundo

# escalas de los conejos, por niveles: muchos chicos, pocos gigantes.
# la progresión importa: recoger todos los chicos hace crecer la bola lo
# suficiente para desbloquear el nivel siguiente.
NIVELES_DE_ESCALA = (
    [(0.35, 0.75)] * 45 + [(1.0, 1.6)] * 18 + [(2.2, 3.0)] * 8 + [(4.0, 5.2)] * 4
)

COLOR_PISO = np.array([0.35, 0.5, 0.3])
COLOR_BOLA = np.array([0.85, 0.85, 0.9])
COLOR_ANILLO_BOLA = np.array([1.0, 1.0, 1.0])
COLOR_RECOGIBLE = np.array([0.3, 0.9, 0.3])
COLOR_BLOQUEANTE = np.array([0.9, 0.25, 0.25])
COLOR_GRILLA = np.array([0.45, 0.45, 0.5])
COLOR_CELDA_CONSULTADA = np.array([0.85, 0.75, 0.2])

# orientaciones de los tres círculos máximos que dibujan cada esfera.
ORIENTACIONES_ANILLO = (
    tr.identity(),
    tr.rotationX(np.pi / 2),
    tr.rotationY(np.pi / 2),
)

# pool de quads para resaltar las celdas consultadas por la fase ancha.
MAXIMO_CELDAS_RESALTADAS = 64
# transform que esconde un nodo del pool (bajo el piso y colapsado).
TRANSFORM_OCULTO = tr.translate(0.0, -50.0, 0.0) @ tr.uniformScale(0.001)

ARRIBA = np.array([0.0, 1.0, 0.0])


def nodo_quad_xz(color=(1.0, 1.0, 1.0)):
    """Nodo con un cuadrado de lado 2 en el plano XZ, centrado en el origen."""
    positions = np.array(
        [-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1], dtype=np.float32
    )
    return {
        "mesh": {"n_vertices": 4, "texture": None, "textures": {}},
        "attributes": {"position": positions},
        "indices": np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32),
        "GL_TYPE": GL.GL_TRIANGLES,
        "transform": tr.identity(),
        "children": [],
        "object": None,
    }


def nodo_linea_unitaria():
    """Nodo con una línea desde el origen hasta (1, 0, 0)."""
    return {
        "mesh": {"n_vertices": 2, "texture": None, "textures": {}},
        "attributes": {
            "position": np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        },
        "indices": np.array([0, 1], dtype=np.uint32),
        "GL_TYPE": GL.GL_LINES,
        "transform": tr.identity(),
        "children": [],
        "object": None,
    }


@click.command("katamari", short_help="Katamari de conejos: fases ancha y angosta de colisión")
@click.option("--width", type=int, default=1000)
@click.option("--height", type=int, default=700)
@click.option("--semilla", type=int, default=7, help="Semilla del campo aleatorio")
def katamari(width, height, semilla):
    window = pyglet.window.Window(width, height, caption="katamari")
    rng = np.random.default_rng(semilla)

    # ---- mallas ----
    bunny = tm.load("assets/bunny.obj")
    bunny.apply_translation(-bunny.centroid)
    bunny.apply_scale(1.0 / bunny.scale)
    # esfera contenedora de la malla, en coordenadas locales: como el conejo
    # está centrado en su centroide, el centro de la esfera es el origen y el
    # radio es la distancia al vértice más lejano.
    radio_base = float(np.linalg.norm(bunny.vertices, axis=1).max())
    # altura del punto más bajo (negativa): sirve para apoyar cada conejo
    # sobre el piso según su escala.
    altura_minima = float(bunny.vertices[:, 1].min())

    cube = tm.load("assets/cube.off")
    cube.apply_translation(-cube.centroid)
    cube.apply_scale(np.sqrt(3) / cube.scale)

    sphere = tm.load("assets/sphere.off")
    sphere.apply_translation(-sphere.centroid)
    sphere.apply_scale(np.sqrt(3) / sphere.scale)

    # ---- grafo de escena principal ----
    graph = Scenegraph("root")
    graph.register_mesh("bunny", _node_from_mesh(bunny))
    graph.register_mesh("cube", _node_from_mesh(cube))
    graph.register_mesh("sphere", _node_from_mesh(sphere))
    shader_dir = __path__[0]
    graph.load_and_register_pipeline(
        "lit", f"{shader_dir}/vertex_program.glsl", f"{shader_dir}/fragment_program.glsl"
    )

    graph.add_object(
        "piso", "cube", "lit", parent="root",
        transform=tr.translate(0, -0.15, 0) @ tr.scale(2 * MEDIO_CAMPO, 0.3, 2 * MEDIO_CAMPO),
        instance_color=COLOR_PISO,
    )
    graph.add_object(
        "bola", "sphere", "lit", parent="root",
        instance_color=COLOR_BOLA,
    )

    # ---- grafos paralelos para los overlays ----
    # un grafo plano por overlay, en vez de subárboles dentro del principal:
    # así cada uno se enciende y apaga con un solo render().
    overlay_esferas = Scenegraph("root")
    overlay_esferas.register_mesh("anillo", ring_node())
    overlay_esferas.load_and_register_pipeline(
        "flat", f"{shader_dir}/flat_vertex_program.glsl", f"{shader_dir}/flat_fragment_program.glsl"
    )

    overlay_grilla = Scenegraph("root")
    overlay_grilla.register_mesh("linea", nodo_linea_unitaria())
    overlay_grilla.register_mesh("celda", nodo_quad_xz())
    overlay_grilla.load_and_register_pipeline(
        "flat", f"{shader_dir}/flat_vertex_program.glsl", f"{shader_dir}/flat_fragment_program.glsl"
    )

    # líneas de la grilla de hash sobre el piso, una por borde de celda.
    bordes = np.arange(-MEDIO_CAMPO, MEDIO_CAMPO + LADO_CELDA / 2, LADO_CELDA)
    for indice_borde, coordenada in enumerate(bordes):
        overlay_grilla.add_mesh_instance(
            f"linea_x_{indice_borde}", "linea", "flat", parent="root",
            transform=tr.translate(-MEDIO_CAMPO, 0.02, coordenada) @ tr.scale(2 * MEDIO_CAMPO, 1, 1),
            instance_color=COLOR_GRILLA,
        )
        overlay_grilla.add_mesh_instance(
            f"linea_z_{indice_borde}", "linea", "flat", parent="root",
            transform=(
                tr.translate(coordenada, 0.02, -MEDIO_CAMPO)
                @ tr.rotationY(-np.pi / 2)
                @ tr.scale(2 * MEDIO_CAMPO, 1, 1)
            ),
            instance_color=COLOR_GRILLA,
        )

    # pool de quads para resaltar las celdas que consulta la fase ancha.
    for indice_celda in range(MAXIMO_CELDAS_RESALTADAS):
        overlay_grilla.add_mesh_instance(
            f"celda_{indice_celda}", "celda", "flat", parent="root",
            transform=TRANSFORM_OCULTO,
            instance_color=COLOR_CELDA_CONSULTADA,
        )

    # ---- estado ----
    state = {
        "posicion": np.zeros(3),       # centro de la bola en el plano (y = radio)
        "angulo": 0.0,                 # rumbo en el plano XZ
        "radio": RADIO_INICIAL,
        "rotacion": tr.identity(),     # rotación acumulada por rodar
        "recogidos": 0,
        "ver_esferas": False,
        "ver_grilla": False,
        "fuerza_bruta": False,
        "celdas_resaltadas": 0,        # cuántos quads del pool están visibles
        "terminado": False,
    }

    grilla = GrillaHashEspacial(LADO_CELDA)
    objetos = {}

    # ---- población del campo ----
    # muestreo con rechazo: se sortea una posición y se acepta solo si no se
    # superpone con los objetos ya puestos. la consulta de superposición usa
    # la misma grilla de hash que después usa el juego.
    for indice, (escala_minima, escala_maxima) in enumerate(NIVELES_DE_ESCALA):
        escala = rng.uniform(escala_minima, escala_maxima)
        radio = escala * radio_base
        for _ in range(80):
            x = rng.uniform(-MEDIO_CAMPO + radio + 0.5, MEDIO_CAMPO - radio - 0.5)
            z = rng.uniform(-MEDIO_CAMPO + radio + 0.5, MEDIO_CAMPO - radio - 0.5)
            if np.hypot(x, z) < 3.0 + radio:
                continue  # despeja la zona donde parte la bola
            centro = np.array([x, -escala * altura_minima, z])
            candidatos, _ = grilla.consultar((x, z), radio + 0.3)
            if any(
                esferas_se_intersectan(
                    centro, radio + 0.2, objetos[otro]["centro"], objetos[otro]["radio"]
                )
                for otro in candidatos
            ):
                continue
            break
        else:
            continue  # no hubo lugar para este conejo; el campo sigue válido

        nombre = f"bunny_{indice}"
        transform = (
            tr.translate(*centro)
            @ tr.rotationY(rng.uniform(0, 2 * np.pi))
            @ tr.uniformScale(escala)
        )
        tono = rng.uniform(0, 1)
        color = np.array(colorsys.hsv_to_rgb(tono, 0.55, 0.85))
        graph.add_object(
            nombre, "bunny", "lit", parent="root",
            transform=transform, instance_color=color,
        )
        anillos = []
        for indice_anillo, orientacion in enumerate(ORIENTACIONES_ANILLO):
            nombre_anillo = f"{nombre}_anillo_{indice_anillo}"
            overlay_esferas.add_mesh_instance(
                nombre_anillo, "anillo", "flat", parent="root",
                transform=tr.translate(*centro) @ orientacion @ tr.uniformScale(radio),
                instance_color=COLOR_RECOGIBLE,
            )
            anillos.append(nombre_anillo)
        objetos[nombre] = {
            "centro": centro,
            "radio": radio,
            "transform": transform,
            "anillos": anillos,
            "vivo": True,
        }
        grilla.insertar(nombre, (centro[0], centro[2]), radio)

    total_objetos = len(objetos)

    # anillos de la esfera de la bola (se mueven con ella cada frame).
    for indice_anillo in range(3):
        overlay_esferas.add_mesh_instance(
            f"bola_anillo_{indice_anillo}", "anillo", "flat", parent="root",
            transform=tr.identity(), instance_color=COLOR_ANILLO_BOLA,
        )

    # ---- panel de información ----
    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(20, 20, 20), background_width=480)
        .add("radio").add("recogidos").add("ancha").add("angosta")
        .footer("flechas mover   V esferas   G grilla   B fuerza bruta   R reset")
    )

    # ---- helpers de estado ----
    def transform_bola():
        x, _, z = state["posicion"]
        return tr.translate(x, state["radio"], z) @ state["rotacion"]

    def actualizar_anillos_objetos():
        """Recolorea las esferas contenedoras según el umbral actual.

        Verde si la bola ya puede recoger al objeto, rojo si la bloquea.
        Se llama cuando cambia el radio, no cada frame.
        """
        limite = UMBRAL_RECOGIDA * state["radio"]
        for objeto in objetos.values():
            if not objeto["vivo"]:
                continue
            color = COLOR_RECOGIBLE if objeto["radio"] <= limite else COLOR_BLOQUEANTE
            for nombre_anillo in objeto["anillos"]:
                overlay_esferas.apply_instance_attributes(
                    nombre_anillo, instance_color=color
                )

    def apply_state():
        graph.apply_instance_attributes(
            "bola_mesh", transform=tr.uniformScale(2 * state["radio"])
        )
        actualizar_anillos_objetos()
        modo = "fuerza bruta" if state["fuerza_bruta"] else "grilla de hash"
        print(
            f"[katamari] radio={state['radio']:.2f} "
            f"recogidos={state['recogidos']}/{total_objetos} "
            f"fase_ancha={modo} esferas={state['ver_esferas']} grilla={state['ver_grilla']}"
        )

    def recoger(nombre):
        """Adhiere un objeto a la bola: reparenting en el grafo de escena.

        El nodo deja de colgar de la raíz y pasa a ser hijo del nodo de la
        bola. Su nueva transformación local es la relativa a la bola en este
        instante, así conserva su pose mundial y desde ahora rueda con ella.
        """
        objeto = objetos[nombre]
        objeto["vivo"] = False
        grilla.remover(nombre, (objeto["centro"][0], objeto["centro"][2]), objeto["radio"])

        relativa = np.linalg.inv(transform_bola()) @ objeto["transform"]
        graph.nodes[nombre]["transform"] = relativa
        graph.remove_edge("root", nombre)
        graph.add_edge("bola", nombre)

        # la esfera contenedora del objeto deja de existir por separado.
        for nombre_anillo in objeto["anillos"]:
            overlay_esferas.apply_instance_attributes(
                nombre_anillo, transform=TRANSFORM_OCULTO
            )

        # crecimiento conservando una fracción del volumen recogido:
        # r^3 nuevo = r^3 + ganancia * r_objeto^3.
        state["radio"] = float(np.cbrt(
            state["radio"] ** 3 + GANANCIA_VOLUMEN * objeto["radio"] ** 3
        ))
        state["recogidos"] += 1
        print(
            f"[katamari] recogido {nombre} (radio {objeto['radio']:.2f}); "
            f"la bola ahora tiene radio {state['radio']:.2f}"
        )
        if state["recogidos"] == total_objetos:
            state["terminado"] = True
            print("[katamari] ¡campo despejado!")

    def reset():
        for nombre, objeto in objetos.items():
            if not objeto["vivo"]:
                objeto["vivo"] = True
                graph.remove_edge("bola", nombre)
                graph.add_edge("root", nombre)
                graph.nodes[nombre]["transform"] = objeto["transform"]
                grilla.insertar(nombre, (objeto["centro"][0], objeto["centro"][2]), objeto["radio"])
                for nombre_anillo, orientacion in zip(objeto["anillos"], ORIENTACIONES_ANILLO):
                    overlay_esferas.apply_instance_attributes(
                        nombre_anillo,
                        transform=(
                            tr.translate(*objeto["centro"])
                            @ orientacion
                            @ tr.uniformScale(objeto["radio"])
                        ),
                    )
        state["posicion"] = np.zeros(3)
        state["angulo"] = 0.0
        state["radio"] = RADIO_INICIAL
        state["rotacion"] = tr.identity()
        state["recogidos"] = 0
        state["terminado"] = False
        apply_state()

    # ---- simulación ----
    # estado de las flechas, para leerlas mientras están presionadas.
    # se empuja al stack de handlers DESPUÉS de definir on_key_press: el
    # decorador @window.event escribe en el frame superior del stack, así
    # que si el KeyStateHandler ya está ahí, lo pisa y deja de recibir teclas.
    keys = pyglet.window.key.KeyStateHandler()

    def actualizar(dt):
        # movimiento: girar cambia el rumbo, avanzar desplaza a la bola.
        if keys[pyglet.window.key.LEFT]:
            state["angulo"] -= VELOCIDAD_GIRO * dt
        if keys[pyglet.window.key.RIGHT]:
            state["angulo"] += VELOCIDAD_GIRO * dt
        direccion = np.array([np.sin(state["angulo"]), 0.0, -np.cos(state["angulo"])])
        avance = (1 if keys[pyglet.window.key.UP] else 0) - (
            1 if keys[pyglet.window.key.DOWN] else 0
        )
        rapidez = 4.0 + 2.5 * state["radio"]  # una bola grande avanza más rápido
        if avance != 0:
            state["posicion"] = state["posicion"] + direccion * (avance * rapidez * dt)
            # rodadura sin deslizamiento: el ángulo girado es la distancia
            # recorrida dividida por el radio, en torno al eje horizontal
            # perpendicular a la dirección de movimiento.
            eje = np.cross(ARRIBA, direccion * avance)
            state["rotacion"] = (
                tr.rotationA(rapidez * dt / state["radio"], eje) @ state["rotacion"]
            )

        # la bola no sale del campo.
        limite = MEDIO_CAMPO - state["radio"]
        state["posicion"][0] = np.clip(state["posicion"][0], -limite, limite)
        state["posicion"][2] = np.clip(state["posicion"][2], -limite, limite)
        graph.nodes["bola"]["transform"] = transform_bola()

        # ---- fase ancha ----
        radio_bola = state["radio"]
        posicion = state["posicion"]
        if state["fuerza_bruta"]:
            candidatos = [n for n, o in objetos.items() if o["vivo"]]
            celdas_consultadas = []
        else:
            candidatos, celdas_consultadas = grilla.consultar(
                (posicion[0], posicion[2]), radio_bola
            )

        # ---- fase angosta y respuesta ----
        centro_bola = np.array([posicion[0], radio_bola, posicion[2]])
        pruebas = 0
        for nombre in sorted(candidatos):
            objeto = objetos[nombre]
            pruebas += 1
            if not esferas_se_intersectan(
                centro_bola, radio_bola, objeto["centro"], objeto["radio"]
            ):
                continue
            if objeto["radio"] <= UMBRAL_RECOGIDA * radio_bola:
                recoger(nombre)
            else:
                # objeto bloqueante: empuja la bola hacia afuera en el plano,
                # lo justo para deshacer la penetración.
                diferencia = centro_bola - objeto["centro"]
                distancia = float(np.linalg.norm(diferencia))
                penetracion = radio_bola + objeto["radio"] - distancia
                horizontal = np.array([diferencia[0], 0.0, diferencia[2]])
                norma = float(np.linalg.norm(horizontal))
                if norma > 1e-6:
                    state["posicion"] = state["posicion"] + horizontal / norma * (
                        penetracion + 1e-3
                    )

        if state["radio"] != radio_bola:
            # hubo recogidas en este frame: el radio cambió y la escala de la
            # malla y los colores de los anillos deben reflejarlo.
            apply_state()
        graph.nodes["bola"]["transform"] = transform_bola()

        # ---- conteos para el panel ----
        vivos = total_objetos - state["recogidos"]
        panel["radio"] = (
            f"radio de la bola: {state['radio']:.2f} "
            f"(recoge hasta {UMBRAL_RECOGIDA * state['radio']:.2f})"
        )
        if state["terminado"]:
            panel["recogidos"] = "¡campo despejado!"
            panel.color("recogidos", (120, 255, 120, 255))
        else:
            panel["recogidos"] = f"objetos recogidos: {state['recogidos']} de {total_objetos}"
        if state["fuerza_bruta"]:
            panel["ancha"] = f"fase ancha: desactivada ({vivos} candidatos)"
        else:
            panel["ancha"] = (
                f"fase ancha: {len(celdas_consultadas)} celdas, "
                f"{len(candidatos)} candidatos de {vivos}"
            )
        panel["angosta"] = f"pruebas esfera-esfera por frame: {pruebas}"

        # ---- resaltado de celdas consultadas ----
        if state["ver_grilla"]:
            visibles = celdas_consultadas[:MAXIMO_CELDAS_RESALTADAS]
            for indice_celda, (i, j) in enumerate(visibles):
                centro_celda_x = (i + 0.5) * LADO_CELDA
                centro_celda_z = (j + 0.5) * LADO_CELDA
                overlay_grilla.apply_instance_attributes(
                    f"celda_{indice_celda}",
                    transform=(
                        tr.translate(centro_celda_x, 0.01, centro_celda_z)
                        @ tr.scale(LADO_CELDA / 2, 1, LADO_CELDA / 2)
                    ),
                )
            for indice_celda in range(len(visibles), state["celdas_resaltadas"]):
                overlay_grilla.apply_instance_attributes(
                    f"celda_{indice_celda}", transform=TRANSFORM_OCULTO
                )
            state["celdas_resaltadas"] = len(visibles)

        # anillos de la bola.
        if state["ver_esferas"]:
            for indice_anillo, orientacion in enumerate(ORIENTACIONES_ANILLO):
                overlay_esferas.apply_instance_attributes(
                    f"bola_anillo_{indice_anillo}",
                    transform=(
                        tr.translate(*centro_bola)
                        @ orientacion
                        @ tr.uniformScale(state["radio"])
                    ),
                )

    pyglet.clock.schedule_interval(actualizar, 1.0 / 60.0)

    # ---- entrada ----
    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.V:
            state["ver_esferas"] = not state["ver_esferas"]
            apply_state()
        elif symbol == pyglet.window.key.G:
            state["ver_grilla"] = not state["ver_grilla"]
            apply_state()
        elif symbol == pyglet.window.key.B:
            state["fuerza_bruta"] = not state["fuerza_bruta"]
            apply_state()
        elif symbol == pyglet.window.key.R:
            reset()
        elif symbol == pyglet.window.key.ESCAPE:
            window.close()

    window.push_handlers(keys)

    # ---- cámara y proyección ----
    projection = tr.perspective(60, width / height, 0.1, 300.0)
    for grafo in (graph, overlay_esferas, overlay_grilla):
        grafo.set_global_attributes(projection=projection)
        grafo.register_view_transform(tr.identity())

    def actualizar_camara():
        # cámara de seguimiento: detrás de la bola según el rumbo, y más
        # lejos mientras más grande es la bola.
        direccion = np.array([np.sin(state["angulo"]), 0.0, -np.cos(state["angulo"])])
        distancia = 2.5 + 4.5 * state["radio"]
        altura = 1.4 + 2.2 * state["radio"]
        objetivo = state["posicion"] + np.array([0.0, state["radio"], 0.0])
        ojo = objetivo - direccion * distancia + ARRIBA * altura
        view = tr.lookAt(ojo, objetivo, ARRIBA)
        for grafo in (graph, overlay_esferas, overlay_grilla):
            grafo.views[grafo.current_view] = view

    # ---- render ----
    @window.event
    def on_draw():
        GL.glClearColor(0.45, 0.6, 0.75, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        actualizar_camara()
        graph.render()
        GL.glLineWidth(2)
        if state["ver_grilla"]:
            overlay_grilla.render()
        if state["ver_esferas"]:
            overlay_esferas.render()
        GL.glLineWidth(1)

        with ui_overlay():
            panel.draw()

    apply_state()
    print(f"[katamari] {total_objetos} conejos en el campo. ¡A rodar!")
    pyglet.app.run()


if __name__ == "__main__":
    katamari()
