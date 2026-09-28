"""Sombras en un campo abierto: un sol con cascaded shadow maps y focos móviles.

El sol es una luz direccional, así que su proyección es ortográfica. El
volumen de esa proyección no es fijo: en cada cuadro se ajusta al frustum de
la cámara, de modo que todo lo visible tiene sombra, esté donde esté la
cámara. Ese frustum se parte por distancia en tramos, las cascadas, y cada
tramo tiene su propio shadow map: la cascada cercana cubre poco terreno con
toda la resolución del mapa y la lejana cubre mucho con la misma.

Todas las cascadas comparten la vista del sol y difieren en la proyección.
Las esquinas de cada tramo salen del cubo NDC llevado al mundo por la inversa
de su matriz de cámara, y la caja envolvente de esas esquinas, expresada en
el sistema del sol, da los seis planos de la proyección ortográfica.

Los focos son luces dinámicas: se mueven, así que su matriz de vista se
recalcula en cada cuadro. Iluminan como el spotlight de la unidad de
iluminación, y su shadow map usa una proyección en perspectiva, igual que la
luz del ejemplo shadow_mapping. Cada foco agrega una pasada de profundidad
por cuadro.

Controles:
- W/S/A/D: moverse. Q/E: subir y bajar.
- Flechas: girar la cámara.
- C: cambiar entre 1, 2 y 3 cascadas.
- V: teñir cada píxel con el color de su cascada.
- T: alternar el ajuste estable (esfera y redondeo al téxel) y el ajuste justo.
- L: encender y apagar los focos.
- N: alternar el día y la noche.
- F: vista desde arriba, con el frustum de la cámara, las cajas del sol y los
  frustums de los focos.
- 1/2: azimut del sol. 3/4: altura del sol.
- R: reiniciar.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import grafica.transformations as tr
from grafica.framebuffers import create_depth_framebuffer
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_nodes import _node_from_mesh
from grafica.scenegraph_premade import bounding_box_node
from grafica.ui import InfoPanel, ui_overlay

SHADOW_MAP_SIZE = 2048
MAX_CASCADES = 3

CAMERA_FOV = 60.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 150.0

# Peso del reparto logarítmico en los cortes entre cascadas; el resto es
# reparto uniforme. Con 1 los tramos cercanos quedan muy cortos.
SPLIT_LAMBDA = 0.75

# Cuánto se estira cada caja hacia el sol. Un objeto fuera del frustum de la
# cámara puede proyectar sombra dentro de él, y tiene que caber en la caja.
CASTER_MARGIN = 60.0

# Bias base, medido en téxeles del shadow map.
BIAS_TEXELS = 1.5

FIELD_HALF_SIZE = 90.0
GROUND_HALF_SIZE = 250.0

MOVE_SPEED = 12.0  # unidades por segundo
TURN_SPEED = 1.2  # radianes por segundo

CASCADE_COLORS = ([1.0, 0.35, 0.35], [0.35, 1.0, 0.35], [0.4, 0.5, 1.0])

# Los focos: centro de su órbita en el piso, radio, altura, velocidad angular
# y color. En el centro de cada órbita hay un Pokémon grande, y el foco apunta
# a un punto que da vueltas cerca de él, así que su sombra gira con la luz.
SPOTS = (
    (np.array([-10.0, 0.0, 18.0]), 12.0, 9.0, 0.35, [1.0, 0.75, 0.35]),
    (np.array([12.0, 0.0, 8.0]), 9.0, 7.0, -0.5, [0.35, 0.8, 1.0]),
    (np.array([0.0, 0.0, -8.0]), 15.0, 11.0, 0.25, [1.0, 0.4, 0.85]),
)
SPOT_MAP_SIZE = 1024
SPOT_FOV = 75.0
SPOT_NEAR, SPOT_FAR = 0.5, 60.0
SPOT_INTENSITY = 3.0

# Luz ambiente, intensidad del sol y color del cielo.
DAY = (0.25, 0.6, (0.62, 0.78, 0.92, 1.0))
NIGHT = (0.05, 0.06, (0.03, 0.04, 0.09, 1.0))

GROUND_COLOR = [0.55, 0.65, 0.45]
PILLAR_COLOR = [0.80, 0.78, 0.74]

# Las ocho esquinas del cubo NDC, en coordenadas homogéneas.
NDC_CORNERS = np.array(
    [[x, y, z, 1.0] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
)

SPECIES = (
    ("assets/Bulbasaur.STL", [0.35, 0.70, 0.45]),
    ("assets/Charmander.STL", [0.95, 0.55, 0.20]),
    ("assets/Squirtle.STL", [0.30, 0.60, 0.85]),
)

DEFAULTS = {
    "eye": np.array([0.0, 2.0, 40.0]),
    "yaw": 0.0,  # 0 mira hacia -z
    "pitch": -0.1,
    "cascade_count": 3,
    "stable": True,
    "show_cascades": False,
    "top_view": False,
    "sun_azimuth": np.radians(35.0),
    "sun_elevation": np.radians(35.0),
    "spots_on": True,
    "night": False,
    "time": 0.0,
}


def normalized_mesh(path):
    """La malla centrada en x y z, apoyada en y = 0 y con altura 1."""
    mesh = tm.load(path)
    low, high = mesh.bounds
    mesh.apply_translation([-(low[0] + high[0]) / 2, -low[1], -(low[2] + high[2]) / 2])
    mesh.apply_scale(1.0 / (high[1] - low[1]))
    return mesh


def field_mesh(rng):
    """Toda la geometría del campo en una sola malla, con color por cara.

    Nada de esto se mueve, así que se fusiona: cada pasada de profundidad es
    una sola llamada de dibujo. Con un nodo por objeto, cada pasada haría unas
    180 llamadas desde Python y el ejemplo no alcanzaría a dibujar una pasada
    por foco.
    """
    cube = normalized_mesh("assets/cube.off")
    species = [(normalized_mesh(path), color) for path, color in SPECIES]
    pieces = []

    def place(mesh, transform, color):
        piece = mesh.copy()
        piece.apply_transform(transform)
        piece.visual.face_colors = [*(np.array(color) * 255).astype(np.uint8), 255]
        pieces.append(piece)

    place(cube, tr.translate(0, -0.2, 0) @ tr.scale(2 * GROUND_HALF_SIZE, 0.2, 2 * GROUND_HALF_SIZE),
          GROUND_COLOR)
    for index in range(150):
        mesh, color = species[index % len(species)]
        x, z = rng.uniform(-FIELD_HALF_SIZE, FIELD_HALF_SIZE, 2)
        place(mesh, tr.translate(x, 0, z) @ tr.rotationY(rng.uniform(0, 2 * np.pi))
              @ tr.uniformScale(rng.uniform(1.5, 4.0)), color)
    for index in range(30):
        x, z = rng.uniform(-FIELD_HALF_SIZE, FIELD_HALF_SIZE, 2)
        place(cube, tr.translate(x, 0, z) @ tr.scale(1.5, rng.uniform(6.0, 18.0), 1.5), PILLAR_COLOR)
    for index, (center, *_) in enumerate(SPOTS):
        mesh, color = species[index % len(species)]
        place(mesh, tr.translate(*center) @ tr.uniformScale(5.0), color)
    return tm.util.concatenate(pieces)


def spot_matrices(center, radius, height, speed, time):
    """Posición, eje y matriz de vista de un foco en el instante `time`.

    El punto al que apunta queda siempre a más de 0,8 radios de la vertical
    del foco, así que up = (0, 1, 0) nunca es paralelo a la dirección de
    visión.
    """
    angle = speed * time
    position = center + np.array([radius * np.cos(angle), height, radius * np.sin(angle)])
    target = center + 0.2 * radius * np.array([np.cos(1.7 * angle), 0.0, np.sin(1.7 * angle)])
    axis = (target - position) / np.linalg.norm(target - position)
    return position, axis, tr.lookAt(position, target, np.array([0.0, 1.0, 0.0]))


def cascade_ends(count):
    """Distancia a la cámara donde termina cada cascada."""
    fractions = np.arange(1, count + 1) / count
    logarithmic = CAMERA_NEAR * (CAMERA_FAR / CAMERA_NEAR) ** fractions
    uniform = CAMERA_NEAR + (CAMERA_FAR - CAMERA_NEAR) * fractions
    return SPLIT_LAMBDA * logarithmic + (1 - SPLIT_LAMBDA) * uniform


def sun_matrices(azimuth, elevation):
    """Dirección hacia el sol y matriz de vista del sol.

    El sol está en el infinito, así que su vista solo fija la orientación: el
    ojo se ubica a una unidad del origen en la dirección del sol, y las cajas
    de las cascadas se miden desde ahí. La altura se limita a 80 grados porque
    con el sol en el cenit up = (0, 1, 0) quedaría paralelo a la dirección de
    visión.
    """
    direction = np.array([
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.cos(azimuth),
    ])
    view = tr.lookAt(direction, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    return direction, view


def fit_cascade(view_camera, aspect, near, far, view_light, stable):
    """Proyección ortográfica del sol que contiene un tramo del frustum.

    Devuelve la proyección y el tamaño de un téxel expresado en unidades de
    profundidad de esa proyección, que es lo que necesita el bias.
    """
    # Las esquinas del tramo en el mundo: el cubo NDC llevado por la inversa
    # de la matriz de cámara de ese tramo. Después, en el sistema del sol.
    slice_projection = tr.perspective(CAMERA_FOV, aspect, near, far)
    corners = NDC_CORNERS @ np.linalg.inv(slice_projection @ view_camera).T
    corners = corners / corners[:, 3:]
    corners_light = (corners @ view_light.T)[:, :3]

    if stable:
        # Una esfera que contiene al tramo no cambia de tamaño al girar la
        # cámara. Además el borde de la caja avanza de a un téxel, así que la
        # cuadrícula del shadow map no se desliza sobre el terreno cuando la
        # cámara se traslada.
        center = corners_light.mean(axis=0)
        radius = np.linalg.norm(corners_light - center, axis=1).max()
        texel = 2 * radius / SHADOW_MAP_SIZE
        low = center - radius
        low[:2] = np.floor(low[:2] / texel) * texel
        high = low + 2 * radius
    else:
        low, high = corners_light.min(axis=0), corners_light.max(axis=0)

    # El sol mira hacia su -z, así que near y far son las z cambiadas de signo.
    near_plane = -high[2] - CASTER_MARGIN
    far_plane = -low[2]
    projection = tr.ortho(low[0], high[0], low[1], high[1], near_plane, far_plane)
    texel_depth = (high[0] - low[0]) / SHADOW_MAP_SIZE / (far_plane - near_plane)
    return projection, texel_depth


@click.command("cascaded_shadows", short_help="Sombras en un campo abierto: un sol con cascadas y focos móviles.")
@click.option("--width", type=int, default=1100)
@click.option("--height", type=int, default=720)
@click.option("--semilla", type=int, default=3, help="Semilla del campo aleatorio")
def cascaded_shadows(width, height, semilla):
    window = pyglet.window.Window(width, height, caption="cascaded shadows")
    aspect = width / height
    rng = np.random.default_rng(semilla)
    here = Path(__file__).parent

    # -------------------------------------------------------------------------
    # Escena: el campo en una sola malla y una ampolleta por foco
    # -------------------------------------------------------------------------
    graph = Scenegraph("root")
    graph.load_and_register_pipeline(
        "lit", here / "vertex_program.glsl", here / "fragment_program.glsl"
    )
    graph.load_and_register_pipeline(
        "depth_shader",
        here / ".." / "shadows" / "simple_vertex_program.glsl",
        here / ".." / "shadows" / "simple_fragment_program.glsl",
    )
    graph.load_and_register_pipeline(
        "bulb",
        here / ".." / "katamari" / "flat_vertex_program.glsl",
        here / ".." / "katamari" / "flat_fragment_program.glsl",
    )

    graph.register_mesh("field", _node_from_mesh(field_mesh(rng)))
    graph.add_mesh_instance("field", "field", "lit", parent="root")

    # Una ampolleta por foco, sin iluminación y fuera de las pasadas de
    # profundidad, porque su pipeline no es "lit".
    graph.register_mesh("sphere", _node_from_mesh(normalized_mesh("assets/sphere.off")))
    for index, (_, _, _, _, color) in enumerate(SPOTS):
        graph.add_object(f"bulb_{index}", "sphere", "bulb", parent="root",
                         instance_color=np.array(color))

    # Un FBO de profundidad por cascada y otro por foco.
    depth_framebuffers = [create_depth_framebuffer(SHADOW_MAP_SIZE) for _ in range(MAX_CASCADES)]
    spot_framebuffers = [create_depth_framebuffer(SPOT_MAP_SIZE) for _ in SPOTS]
    for index, (_, depth_buffer) in enumerate(depth_framebuffers):
        graph.add_texture_to_node("field", f"shadow_map_{index}", depth_buffer.id)
    for index, (_, depth_buffer) in enumerate(spot_framebuffers):
        graph.add_texture_to_node("field", f"spot_shadow_map_{index}", depth_buffer.id)

    spot_projection = tr.perspective(SPOT_FOV, 1.0, SPOT_NEAR, SPOT_FAR)
    graph.set_global_attributes(
        spot_near=SPOT_NEAR,
        spot_far=SPOT_FAR,
        # el téxel de un foco mide 2 d tan(fov / 2) / tamaño a distancia d
        spot_bias_per_distance=float(BIAS_TEXELS * 2 * np.tan(np.radians(SPOT_FOV) / 2) / SPOT_MAP_SIZE),
        **{f"spot_color_{index}": np.array(color, dtype=np.float32)
           for index, (_, _, _, _, color) in enumerate(SPOTS)},
    )

    # -------------------------------------------------------------------------
    # Overlay de la vista desde arriba: por cada cascada, su tramo del frustum
    # de la cámara y su caja del sol, y el frustum de cada foco. Todas las
    # piezas son el cubo NDC llevado al mundo por una inversa.
    # -------------------------------------------------------------------------
    overlay_graph = Scenegraph("root")
    overlay_graph.load_and_register_pipeline(
        "line_pipeline",
        here / ".." / "scene_graphs" / "line_vertex_program.glsl",
        here / ".." / "hello_world" / "fragment_program.glsl",
    )
    for index, color in enumerate(CASCADE_COLORS):
        for piece in ("slice", "box"):
            overlay_graph.add_mesh_instance(
                f"{piece}_{index}",
                bounding_box_node(np.array([-1, -1, -1]), np.array([1, 1, 1]), color=color),
                "line_pipeline",
                parent="root",
            )
    for index, (_, _, _, _, color) in enumerate(SPOTS):
        overlay_graph.add_mesh_instance(
            f"spot_{index}",
            bounding_box_node(np.array([-1, -1, -1]), np.array([1, 1, 1]), color=color),
            "line_pipeline",
            parent="root",
        )

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(20, 20, 20), background_width=560)
        .add("cascadas").add("cortes").add("ajuste").add("sol").add("focos").add("pasadas").add("vista")
        .footer("WASD mover  QE altura  flechas mirar  C cascadas  V colores  T ajuste  "
                "L focos  N noche  F arriba  1/2 3/4 sol  R reset")
    )

    state = {key: (value.copy() if isinstance(value, np.ndarray) else value)
             for key, value in DEFAULTS.items()}
    keys = pyglet.window.key.KeyStateHandler()

    def camera_view():
        forward = np.array([
            np.sin(state["yaw"]) * np.cos(state["pitch"]),
            np.sin(state["pitch"]),
            -np.cos(state["yaw"]) * np.cos(state["pitch"]),
        ])
        return tr.lookAt(state["eye"], state["eye"] + forward, np.array([0.0, 1.0, 0.0]))

    def update_panel():
        ends = cascade_ends(state["cascade_count"])
        panel["cascadas"] = f"cascadas:  {state['cascade_count']}"
        panel["cortes"] = "cortes:    " + "  ".join(f"{end:.1f}" for end in ends)
        panel["ajuste"] = f"ajuste:    {'estable' if state['stable'] else 'justo'}"
        panel["sol"] = (f"sol:       azimut {np.degrees(state['sun_azimuth']):.0f}  "
                        f"altura {np.degrees(state['sun_elevation']):.0f}  "
                        f"({'noche' if state['night'] else 'día'})")
        spot_count = len(SPOTS) if state["spots_on"] else 0
        panel["focos"] = f"focos:     {'encendidos' if state['spots_on'] else 'apagados'}"
        panel["pasadas"] = (f"pasadas:   {state['cascade_count']} del sol + {spot_count} de focos"
                            f" + 1 final")
        panel["vista"] = f"vista:     {'desde arriba' if state['top_view'] else 'cámara'}"

    update_panel()

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.C:
            state["cascade_count"] = state["cascade_count"] % MAX_CASCADES + 1
        elif symbol == key.V:
            state["show_cascades"] = not state["show_cascades"]
        elif symbol == key.T:
            state["stable"] = not state["stable"]
        elif symbol == key.F:
            state["top_view"] = not state["top_view"]
        elif symbol == key.L:
            state["spots_on"] = not state["spots_on"]
        elif symbol == key.N:
            state["night"] = not state["night"]
        elif symbol in (key._1, key._2):
            state["sun_azimuth"] += np.radians(10.0) * (1 if symbol == key._2 else -1)
        elif symbol in (key._3, key._4):
            state["sun_elevation"] = float(np.clip(
                state["sun_elevation"] + np.radians(5.0) * (1 if symbol == key._4 else -1),
                np.radians(10.0), np.radians(80.0),
            ))
        elif symbol == key.R:
            state.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in DEFAULTS.items()})
        update_panel()

    # Se empuja después de definir on_key_press: @window.event escribe en el
    # frame superior del stack y reemplazaría el on_key_press de `keys`.
    window.push_handlers(keys)

    def update_world(dt):
        key = pyglet.window.key
        state["yaw"] += TURN_SPEED * dt * (keys[key.RIGHT] - keys[key.LEFT])
        state["pitch"] = float(np.clip(
            state["pitch"] + TURN_SPEED * dt * (keys[key.UP] - keys[key.DOWN]), -1.4, 1.4
        ))
        ahead = np.array([np.sin(state["yaw"]), 0.0, -np.cos(state["yaw"])])
        right = np.array([np.cos(state["yaw"]), 0.0, np.sin(state["yaw"])])
        state["eye"] += MOVE_SPEED * dt * (
            ahead * (keys[key.W] - keys[key.S])
            + right * (keys[key.D] - keys[key.A])
            + np.array([0.0, 1.0, 0.0]) * (keys[key.E] - keys[key.Q])
        )
        state["eye"][1] = max(state["eye"][1], 0.5)

        state["time"] += dt
        for index, spot in enumerate(SPOTS):
            position, _, _ = spot_matrices(*spot[:4], state["time"])
            graph.nodes[f"bulb_{index}"]["transform"] = (
                tr.translate(*position) @ tr.uniformScale(0.8) @ tr.translate(0.0, -0.5, 0.0)
            )
        graph.calculate_global_transforms()

    @window.event
    def on_draw():
        GL.glEnable(GL.GL_DEPTH_TEST)
        ambient, sun_intensity, sky = NIGHT if state["night"] else DAY
        GL.glClearColor(*sky)

        view_camera = camera_view()
        sun_direction, view_light = sun_matrices(state["sun_azimuth"], state["sun_elevation"])
        graph.register_view_transform(view_light, name="light_view")

        # ---------------------------------------------------------------------
        # Una pasada de profundidad por cascada, con la vista del sol y la
        # proyección ajustada a su tramo.
        # ---------------------------------------------------------------------
        ends = cascade_ends(state["cascade_count"])
        starts = np.concatenate([[CAMERA_NEAR], ends[:-1]])
        projections, texel_depths = [], []
        for index in range(state["cascade_count"]):
            projection, texel_depth = fit_cascade(
                view_camera, aspect, starts[index], ends[index], view_light, state["stable"]
            )
            projections.append(projection)
            texel_depths.append(texel_depth)

            framebuffer, _ = depth_framebuffers[index]
            framebuffer.bind()
            GL.glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            graph.current_view = "light_view"
            graph.set_global_attributes(projection=projection)
            graph.render(
                recalculate_transforms=False,
                only_pipelines={"lit"},
                pipeline_override="depth_shader",
            )
            framebuffer.unbind()

        # Las cascadas que no se usan repiten la última, para que los
        # uniforms siempre tengan un valor válido.
        while len(projections) < MAX_CASCADES:
            projections.append(projections[-1])
            texel_depths.append(texel_depths[-1])
        ndc_to_texture = tr.translate(0.5, 0.5, 0.5) @ tr.uniformScale(0.5)
        shadow_transforms = {
            f"shadow_transform_{index}": ndc_to_texture @ projection @ view_light
            for index, projection in enumerate(projections)
        }

        # ---------------------------------------------------------------------
        # Una pasada de profundidad por foco. Cada foco se movió desde el
        # cuadro anterior, así que su vista se vuelve a calcular.
        # ---------------------------------------------------------------------
        spot_views, spot_uniforms = [], {}
        for index, spot in enumerate(SPOTS):
            position, axis, view_spot = spot_matrices(*spot[:4], state["time"])
            spot_views.append(view_spot)
            spot_uniforms[f"spot_position_{index}"] = position.astype(np.float32)
            spot_uniforms[f"spot_direction_{index}"] = axis.astype(np.float32)
            spot_uniforms[f"spot_shadow_transform_{index}"] = (
                ndc_to_texture @ spot_projection @ view_spot
            ).astype(np.float32)
            if not state["spots_on"]:
                continue

            framebuffer, _ = spot_framebuffers[index]
            framebuffer.bind()
            GL.glViewport(0, 0, SPOT_MAP_SIZE, SPOT_MAP_SIZE)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            graph.register_view_transform(view_spot, name="spot_view")
            graph.set_global_attributes(projection=spot_projection)
            graph.render(
                recalculate_transforms=False,
                only_pipelines={"lit"},
                pipeline_override="depth_shader",
            )
            framebuffer.unbind()

        # ---------------------------------------------------------------------
        # La pasada final, desde la cámara o desde arriba.
        # ---------------------------------------------------------------------
        GL.glViewport(0, 0, window.width, window.height)
        window.clear()

        if state["top_view"]:
            ahead = np.array([np.sin(state["yaw"]), 0.0, -np.cos(state["yaw"])])
            above = state["eye"] + np.array([0.0, 230.0, 0.0])
            graph.register_view_transform(tr.lookAt(above, state["eye"], ahead), name="top")
            main_projection = tr.perspective(60.0, aspect, 1.0, 600.0)
        else:
            graph.register_view_transform(view_camera, name="camera")
            main_projection = tr.perspective(CAMERA_FOV, aspect, CAMERA_NEAR, CAMERA_FAR)

        padded_ends = np.concatenate([ends, np.full(MAX_CASCADES - len(ends), CAMERA_FAR)])
        graph.set_global_attributes(
            projection=main_projection,
            camera_view=view_camera.astype(np.float32),
            sun_direction=sun_direction.astype(np.float32),
            cascade_ends=padded_ends.astype(np.float32),
            cascade_count=int(state["cascade_count"]),
            cascade_biases=(BIAS_TEXELS * np.array(texel_depths)).astype(np.float32),
            show_cascades=int(state["show_cascades"]),
            ambient=ambient,
            sun_intensity=sun_intensity,
            spot_intensity=SPOT_INTENSITY if state["spots_on"] else 0.0,
            **{name: matrix.astype(np.float32) for name, matrix in shadow_transforms.items()},
            **spot_uniforms,
        )
        graph.render(recalculate_transforms=False, only_pipelines={"lit", "bulb"})

        if state["top_view"]:
            for index in range(MAX_CASCADES):
                if index < state["cascade_count"]:
                    slice_projection = tr.perspective(CAMERA_FOV, aspect, starts[index], ends[index])
                    slice_transform = np.linalg.inv(slice_projection @ view_camera)
                    box_transform = np.linalg.inv(projections[index] @ view_light)
                else:
                    # escala cero: la cascada que no se usa colapsa a un punto
                    slice_transform = box_transform = tr.uniformScale(0.0)
                overlay_graph.nodes[f"slice_{index}"]["transform"] = slice_transform.astype(np.float32)
                overlay_graph.nodes[f"box_{index}"]["transform"] = box_transform.astype(np.float32)
            # el frustum de cada foco, con la misma inversa del ejemplo shadow_mapping
            for index, view_spot in enumerate(spot_views):
                spot_transform = (np.linalg.inv(spot_projection @ view_spot) if state["spots_on"]
                                  else tr.uniformScale(0.0))
                overlay_graph.nodes[f"spot_{index}"]["transform"] = spot_transform.astype(np.float32)
            overlay_graph.register_view_transform(graph.views["top"])
            overlay_graph.set_global_attributes(projection=main_projection)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glLineWidth(2.0)
            overlay_graph.render()
            GL.glLineWidth(1.0)
            GL.glEnable(GL.GL_DEPTH_TEST)

        with ui_overlay():
            panel.draw()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0)
    pyglet.app.run()
