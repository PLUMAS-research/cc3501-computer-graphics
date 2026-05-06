"""Sistema Tierra-Luna iluminado por el Sol con composición tipo eclipse.

La cámara sigue a la Tierra desde el lado opuesto al Sol, mirando hacia ella.
Como resultado, la Tierra ocupa el centro del frame con su lado nocturno
hacia la cámara, y el Sol queda detrás de ella. El halo del Sol se dibuja
como un billboard 3D ubicado en la posición del Sol y orientado a la cámara,
así el depth test se encarga de que solo se vea el halo que asoma alrededor
de la silueta del planeta.

Para que las texturas del lado nocturno sean visibles (la Tierra real estaría
casi negra) subimos artificialmente la luz ambiente. Es una concesión
pedagógica: el alumno tiene que poder ver continentes y océanos.

Compone tres efectos:
- Tierra y Luna iluminadas por Phong con textura difusa.
- Sol como esfera emisiva pequeña + halo 3D que el depth test ocluye.
- Lens flare como cadena de billboards aditivos en NDC, dibujada solo
  cuando el Sol está delante de la cámara y dentro del frame.
"""

import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from PIL import Image

import grafica.transformations as tr
from grafica.scenegraph import Scenegraph
from grafica.textures import texture_2D_setup
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline


def uv_sphere_node(radius=1.0, lat_segments=48, lon_segments=96):
    """Esfera con normales por vértice y UV equirectangular.

    El polo norte queda en +Z (consistente con el resto del proyecto, que
    usa Z arriba). La coordenada u recorre la longitud de 0 a 1 sobre el
    ecuador y v recorre la latitud de 0 (polo norte) a 1 (polo sur). Esto
    coincide con el formato de los mapas equirectangulares de NASA.
    """
    lat_indices = np.arange(lat_segments + 1)
    lon_indices = np.arange(lon_segments + 1)
    lat_grid, lon_grid = np.meshgrid(lat_indices, lon_indices, indexing="ij")

    v = lat_grid / lat_segments
    u = lon_grid / lon_segments

    theta = v * np.pi          # 0 polo norte, pi polo sur
    phi = u * 2.0 * np.pi      # 0 a 2pi alrededor de Z

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    positions = np.stack([radius * x, radius * y, radius * z], axis=-1).astype(np.float32)
    normals = np.stack([x, y, z], axis=-1).astype(np.float32)
    uvs = np.stack([u, v], axis=-1).astype(np.float32)

    indices = []
    stride = lon_segments + 1
    for lat in range(lat_segments):
        for lon in range(lon_segments):
            current = lat * stride + lon
            next_row = current + stride
            indices.extend([current, next_row, current + 1])
            indices.extend([current + 1, next_row, next_row + 1])
    indices = np.array(indices, dtype=np.uint32)

    n_vertices = (lat_segments + 1) * (lon_segments + 1)

    return {
        "mesh": {
            "n_vertices": n_vertices,
            "texture": None,
            "textures": {},
        },
        "attributes": {
            "position": positions.flatten(),
            "normal": normals.flatten(),
            "uv": uvs.flatten(),
            "color": None,
        },
        "indices": indices,
        "GL_TYPE": GL.GL_TRIANGLES,
        "transform": tr.identity(),
        "id": None,
        "children": [],
        "parent": None,
        "object": None,
        "has_texture": False,
    }


def make_radial_falloff(size=256, power=2.0):
    """PNG en memoria con caída radial blanca, para halo y lens flare.

    El RGB es blanco constante; la información del disco va en el canal
    alfa con caída (1 - r)^power, donde r es la distancia al centro
    normalizada. Sirve como textura única para todos los billboards
    aditivos del ejemplo.
    """
    coords = np.linspace(-1.0, 1.0, size)
    grid_x, grid_y = np.meshgrid(coords, coords)
    distance = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    intensity = np.clip(1.0 - distance, 0.0, 1.0) ** power

    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = (intensity * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


# Cadena de "fantasmas" del lens flare. Cada elemento se ubica
# interpolando t entre la posición del Sol en pantalla (t=0) y el
# centro (t=1). El halo del Sol no va acá: lo dibujamos aparte como
# billboard 3D para que se beneficie del depth test (la Tierra lo
# ocluye correctamente). La cadena entera solo se dibuja cuando el
# Sol está delante de la cámara y su NDC está dentro del frame.
LENS_FLARE_CHAIN = [
    {"t": 0.30, "size": 0.10, "intensity": 0.45, "color": (0.95, 0.80, 0.60)},
    {"t": 0.55, "size": 0.06, "intensity": 0.30, "color": (0.55, 0.75, 1.00)},
    {"t": 0.70, "size": 0.18, "intensity": 0.25, "color": (1.00, 0.65, 0.50)},
    {"t": 1.00, "size": 0.04, "intensity": 0.50, "color": (1.00, 1.00, 1.00)},  # rim al centro
]

# Si el NDC del Sol está más lejos del centro que este umbral, ya
# no dibujamos lens flare: el Sol está demasiado fuera del frame
# como para que tenga sentido óptico
LENS_FLARE_NDC_LIMIT = 1.3

SUN_HALO_WORLD_SIZE = 8.0  # diámetro del billboard 3D del halo en unidades de mundo

# Tintes fijos: lo que cambia con los controles es la intensidad,
# no la cromaticidad. El ambient global tiene una leve dominancia azul
# (recuerda el azul del cielo terrestre); el Sol tiene tinte cálido
AMBIENT_TINT = np.array([1.00, 1.00, 1.15], dtype=np.float32)
SUN_TINT = np.array([1.00, 0.95, 0.80], dtype=np.float32)

LIGHTING_DEFAULTS = {
    "ambient_intensity": 0.20,
    "diffuse_intensity": 1.00,
    "earth_ambient_factor": 1.00,
    "moon_ambient_factor": 0.10,
}

LIGHTING_STEPS = {
    "ambient_intensity": 0.02,
    "diffuse_intensity": 0.05,
    "earth_ambient_factor": 0.05,
    "moon_ambient_factor": 0.02,
}

LIGHTING_RANGES = {
    "ambient_intensity": (0.0, 1.0),
    "diffuse_intensity": (0.0, 2.0),
    "earth_ambient_factor": (0.0, 2.0),
    "moon_ambient_factor": (0.0, 1.0),
}


@click.command("sistema_tierra", short_help="Tierra y Luna iluminadas por el Sol con halo y lens flare")
@click.option("--width", type=int, default=1280)
@click.option("--height", type=int, default=720)
@click.option(
    "--earth-texture",
    type=click.Path(exists=False),
    default="assets/earth_diffuse.jpg",
    help="Mapa difuso equirectangular de la Tierra",
)
@click.option(
    "--moon-texture",
    type=click.Path(exists=False),
    default="assets/moon_diffuse.jpg",
    help="Mapa difuso equirectangular de la Luna",
)
def sistema_tierra(width, height, earth_texture, moon_texture):
    earth_path = Path(earth_texture)
    moon_path = Path(moon_texture)
    if not earth_path.exists():
        click.echo(
            f"Falta {earth_path}. Descarga la textura desde "
            f"https://visibleearth.nasa.gov/collection/1484/blue-marble"
        )
        return
    if not moon_path.exists():
        click.echo(
            f"Falta {moon_path}. Descarga la textura desde "
            f"https://svs.gsfc.nasa.gov/4720/"
        )
        return

    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )

    earth_image = Image.open(earth_path).convert("RGB")
    moon_image = Image.open(moon_path).convert("RGB")
    # los mapas equirectangulares de NASA tienen el polo norte en la fila
    # superior. PIL conserva esa orientación. Si dejaramos el flip vertical
    # por defecto de texture_2D_setup, sample con v=0 leería el polo sur
    # y la Antártida aparecería en el polo norte de la esfera
    earth_texture_id = texture_2D_setup(earth_image, flip_top_bottom=False)
    moon_texture_id = texture_2D_setup(moon_image, flip_top_bottom=False)
    halo_image = make_radial_falloff(size=256, power=2.0)
    halo_texture_id = texture_2D_setup(halo_image)

    base_path = Path(os.path.dirname(__file__))

    earth_pipeline = load_pipeline(
        base_path / "earth_vertex_program.glsl",
        base_path / "earth_fragment_program.glsl",
    )
    sun_pipeline = load_pipeline(
        base_path / "sun_vertex_program.glsl",
        base_path / "sun_fragment_program.glsl",
    )
    flare_pipeline = load_pipeline(
        base_path / "flare_vertex_program.glsl",
        base_path / "flare_fragment_program.glsl",
    )
    halo_pipeline = load_pipeline(
        base_path / "halo_vertex_program.glsl",
        base_path / "flare_fragment_program.glsl",
    )

    graph = Scenegraph("root")
    graph.register_pipeline("earth_pipeline", earth_pipeline)
    graph.register_pipeline("sun_pipeline", sun_pipeline)

    earth_sphere = uv_sphere_node(radius=1.0, lat_segments=64, lon_segments=128)
    earth_sphere["mesh"]["texture"] = earth_texture_id
    graph.register_mesh("earth_sphere", earth_sphere)

    moon_sphere = uv_sphere_node(radius=1.0, lat_segments=48, lon_segments=96)
    moon_sphere["mesh"]["texture"] = moon_texture_id
    graph.register_mesh("moon_sphere", moon_sphere)

    sun_sphere = uv_sphere_node(radius=1.0, lat_segments=24, lon_segments=48)
    graph.register_mesh("sun_sphere", sun_sphere)

    # constantes "narrativas". No son a escala real (en escala el sistema
    # solar es mayoritariamente vacío) sino las que producen una imagen
    # legible. Sol pequeño y órbita compacta para que la cámara pueda
    # acercarse a la Tierra (el sujeto principal) sin perder al Sol del
    # encuadre. El halo aditivo le devuelve presencia al Sol aunque el
    # disco sea pequeño
    SUN_RADIUS = 0.8
    EARTH_RADIUS = 1.0
    MOON_RADIUS = 0.27
    EARTH_ORBIT = 8.0
    MOON_ORBIT = 2.5

    sun_color = np.array([1.0, 0.95, 0.80], dtype=np.float32)

    # Sol
    graph.add_object(
        "sun", "sun_sphere", "sun_pipeline",
        parent="root",
        transform=tr.uniformScale(SUN_RADIUS),
        emission_color=sun_color,
    )

    # cadena Sol -> earth_orbit (rotación) -> earth_position (translación)
    # -> earth (rotación de eje + escala). La translación va separada de
    # la escala porque la Luna también cuelga de earth_position y no debe
    # heredar el escalado de la Tierra
    graph.add_transform("earth_orbit", tr.identity())
    graph.add_edge("root", "earth_orbit")

    graph.add_transform("earth_position", tr.translate(EARTH_ORBIT, 0, 0))
    graph.add_edge("earth_orbit", "earth_position")

    graph.add_object(
        "earth", "earth_sphere", "earth_pipeline",
        parent="earth_position",
        transform=tr.uniformScale(EARTH_RADIUS),
        ambient_factor=LIGHTING_DEFAULTS["earth_ambient_factor"],
    )

    graph.add_transform("moon_orbit", tr.identity())
    graph.add_edge("earth_position", "moon_orbit")

    graph.add_transform("moon_position", tr.translate(MOON_ORBIT, 0, 0))
    graph.add_edge("moon_orbit", "moon_position")

    graph.add_object(
        "moon", "moon_sphere", "earth_pipeline",
        parent="moon_position",
        transform=tr.uniformScale(MOON_RADIUS),
        ambient_factor=LIGHTING_DEFAULTS["moon_ambient_factor"],
    )

    # cámara tipo "eclipse parcial" sesgada al lado del Sol. Apenas un
    # poquito detrás de la Tierra (k_radial pequeño y positivo) con un
    # offset lateral más grande, así el ~40% del disco visible queda
    # iluminado: el lado de día domina visualmente uno de los limbos y
    # se aprecia la transición del terminador. El Sol queda apenas
    # fuera del frame en el costado opuesto, pero su halo 3D asoma de
    # vuelta hacia adentro y la cadena de lens flare cruza la pantalla.
    CAMERA_OFFSET_RADIAL = 1.0
    CAMERA_OFFSET_TANGENT = 2.0
    CAMERA_OFFSET_UP = 1.5

    world_up = np.array([0.0, 0.0, 1.0])
    projection = tr.perspective(65, width / height, 0.5, 200.0)

    def compute_view():
        """Recalcula la matriz de vista a partir de la posición actual de la Tierra."""
        earth_world = graph.get_global_position("earth_position")
        earth_distance = float(np.linalg.norm(earth_world))
        if earth_distance < 1e-6:
            return tr.lookAt(np.array([0.0, 1.0, 1.0]), np.zeros(3), world_up), np.zeros(3)

        radial = earth_world / earth_distance
        tangent = np.cross(world_up, radial)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-6:
            tangent = np.array([0.0, 1.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        camera = (
            earth_world
            + CAMERA_OFFSET_RADIAL * radial
            + CAMERA_OFFSET_TANGENT * tangent
            + CAMERA_OFFSET_UP * world_up
        )

        return tr.lookAt(camera, earth_world, world_up), camera

    initial_view, _ = compute_view()

    graph.register_view_transform(initial_view)
    graph.set_global_attributes(
        projection=projection,
        light_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    # estado de iluminación manipulable por teclado. Los valores se
    # propagan al grafo y a los uniforms de instancia cada vez que
    # cambian. Las teclas están definidas más abajo en on_key_press
    lighting_state = dict(LIGHTING_DEFAULTS)

    # quad unitario en NDC para los billboards aditivos del lens flare
    flare_vertex_list = flare_pipeline.vertex_list_indexed(
        4, GL.GL_TRIANGLES,
        np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32),
    )
    flare_vertex_list.position[:] = np.array(
        [-0.5, -0.5,  0.5, -0.5,  0.5,  0.5,  -0.5,  0.5],
        dtype=np.float32,
    )
    flare_vertex_list.uv[:] = np.array(
        [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0],
        dtype=np.float32,
    )

    # quad para el halo 3D del Sol; el vertex shader lo expande en
    # coordenadas de mundo usando los ejes de la cámara
    halo_vertex_list = halo_pipeline.vertex_list_indexed(
        4, GL.GL_TRIANGLES,
        np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32),
    )
    halo_vertex_list.position[:] = np.array(
        [-0.5, -0.5,  0.5, -0.5,  0.5,  0.5,  -0.5,  0.5],
        dtype=np.float32,
    )
    halo_vertex_list.uv[:] = np.array(
        [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0],
        dtype=np.float32,
    )

    def make_label(y_offset, color=(220, 220, 220, 255)):
        return pyglet.text.Label(
            "",
            font_name="Fira Code",
            font_size=12,
            x=12,
            y=height - y_offset,
            color=color,
        )

    label_ambient = make_label(24)
    label_diffuse = make_label(46)
    label_earth_amb = make_label(68)
    label_moon_amb = make_label(90)

    label_instructions = pyglet.text.Label(
        "[1/2] luz ambiente   [3/4] luz solar   [5/6] amb Tierra   [7/8] amb Luna   [R] reset",
        font_name="Fira Code",
        font_size=11,
        x=12,
        y=12,
        color=(200, 200, 200, 255),
    )

    def apply_lighting():
        """Propaga lighting_state al grafo y a las etiquetas; imprime al stdout."""
        ambient_color = (lighting_state["ambient_intensity"] * AMBIENT_TINT).astype(np.float32)
        diffuse_color = (lighting_state["diffuse_intensity"] * SUN_TINT).astype(np.float32)
        graph.set_global_attributes(
            ambient_light=ambient_color,
            light_color=diffuse_color,
        )
        graph.apply_instance_attributes(
            "earth_mesh", ambient_factor=float(lighting_state["earth_ambient_factor"])
        )
        graph.apply_instance_attributes(
            "moon_mesh", ambient_factor=float(lighting_state["moon_ambient_factor"])
        )

        label_ambient.text = (
            f"luz ambiente:    {lighting_state['ambient_intensity']:.2f}"
        )
        label_diffuse.text = (
            f"luz solar:       {lighting_state['diffuse_intensity']:.2f}"
        )
        label_earth_amb.text = (
            f"amb Tierra:      {lighting_state['earth_ambient_factor']:.2f}"
        )
        label_moon_amb.text = (
            f"amb Luna:        {lighting_state['moon_ambient_factor']:.2f}"
        )

        print(
            f"[iluminacion]"
            f"  ambient={lighting_state['ambient_intensity']:.2f}"
            f"  diffuse={lighting_state['diffuse_intensity']:.2f}"
            f"  amb_tierra={lighting_state['earth_ambient_factor']:.2f}"
            f"  amb_luna={lighting_state['moon_ambient_factor']:.2f}"
        )

    apply_lighting()

    def adjust_param(name, sign):
        step = LIGHTING_STEPS[name]
        low, high = LIGHTING_RANGES[name]
        lighting_state[name] = float(np.clip(lighting_state[name] + sign * step, low, high))
        apply_lighting()

    total_time = 0.0

    def project_to_ndc(world_point):
        clip = projection @ current_view @ np.array([*world_point, 1.0])
        if clip[3] <= 0.0:
            return None  # detrás de la cámara
        return clip[:2] / clip[3]

    def draw_sun_halo(camera_position):
        """Halo 3D del Sol: billboard en la posición del Sol.

        El depth test está activo, así que el halo solo se dibuja en
        los píxeles donde no hay nada más cercano. La Tierra (más cerca
        de la cámara) lo ocluye en la región donde se solapan, dejando
        que solo asome el "borde" del halo alrededor de la silueta.
        """
        # camera_right y camera_up en coordenadas de mundo: las dos
        # primeras filas de la matriz de vista son justamente esos
        # ejes expresados en mundo (lookAt los pone allí)
        camera_right_world = current_view[0, 0:3].astype(np.float32)
        camera_up_world = current_view[1, 0:3].astype(np.float32)

        GL.glDepthMask(GL.GL_FALSE)  # no escribir depth: el halo no debe ocluir cosas detrás
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, halo_texture_id)

        halo_pipeline.use()
        halo_pipeline["halo_world_position"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        halo_pipeline["halo_size"] = float(SUN_HALO_WORLD_SIZE)
        halo_pipeline["camera_right"] = camera_right_world
        halo_pipeline["camera_up"] = camera_up_world
        halo_pipeline["view"] = current_view.reshape(16, 1, order="F")
        halo_pipeline["projection"] = projection.reshape(16, 1, order="F")
        halo_pipeline["flare_color"] = sun_color
        halo_pipeline["flare_intensity"] = 1.0
        halo_vertex_list.draw(GL.GL_TRIANGLES)
        halo_pipeline.stop()

        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)

    def draw_lens_flare(sun_ndc):
        # solo dibujamos lens flare si el Sol está dentro del frame
        # (o muy cerca del borde). Esto evita que aparezcan los
        # fantasmas cuando el Sol está fuera de la imagen, lo cual
        # no tiene sentido óptico (el flare es un artefacto de la
        # luz que entra al lente)
        if sun_ndc is None:
            return
        if np.linalg.norm(sun_ndc) > LENS_FLARE_NDC_LIMIT:
            return

        aspect_ratio = width / height

        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, halo_texture_id)

        flare_pipeline.use()
        flare_pipeline["aspect_ratio"] = aspect_ratio

        for element in LENS_FLARE_CHAIN:
            t = element["t"]
            center = (1.0 - t) * sun_ndc + t * np.array([0.0, 0.0])
            flare_pipeline["flare_center_ndc"] = center.astype(np.float32)
            flare_pipeline["flare_size"] = float(element["size"])
            flare_pipeline["flare_color"] = np.array(element["color"], dtype=np.float32)
            flare_pipeline["flare_intensity"] = float(element["intensity"])
            flare_vertex_list.draw(GL.GL_TRIANGLES)

        flare_pipeline.stop()

        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)

    current_view = initial_view

    @window.event
    def on_draw():
        nonlocal current_view

        GL.glClearColor(0.01, 0.01, 0.03, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        current_view, camera_position = compute_view()
        graph.views[graph.current_view] = current_view

        graph.render(recalculate_transforms=False)

        # el halo 3D va antes que el lens flare, en el mismo bloque
        # de blending aditivo. Tiene depth test on porque queremos que
        # la Tierra lo ocluya
        draw_sun_halo(camera_position)

        sun_ndc = project_to_ndc(np.array([0.0, 0.0, 0.0]))
        draw_lens_flare(sun_ndc)

        with ui_overlay():
            label_ambient.draw()
            label_diffuse.draw()
            label_earth_amb.draw()
            label_moon_amb.draw()
            label_instructions.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        teclas = {
            pyglet.window.key._1: ("ambient_intensity", -1),
            pyglet.window.key._2: ("ambient_intensity", +1),
            pyglet.window.key._3: ("diffuse_intensity", -1),
            pyglet.window.key._4: ("diffuse_intensity", +1),
            pyglet.window.key._5: ("earth_ambient_factor", -1),
            pyglet.window.key._6: ("earth_ambient_factor", +1),
            pyglet.window.key._7: ("moon_ambient_factor", -1),
            pyglet.window.key._8: ("moon_ambient_factor", +1),
        }
        if symbol in teclas:
            name, sign = teclas[symbol]
            adjust_param(name, sign)
        elif symbol == pyglet.window.key.R:
            lighting_state.update(LIGHTING_DEFAULTS)
            apply_lighting()

    def update_world(dt, _):
        nonlocal total_time
        total_time += dt

        # frecuencias arbitrarias para que la Luna dé varias vueltas
        # mientras la Tierra recorre su órbita
        earth_orbit_speed = 0.05
        earth_spin_speed = 0.5
        moon_orbit_speed = 0.6
        moon_spin_speed = 0.6  # Luna en rotación sincrónica con su órbita

        graph.nodes["earth_orbit"]["transform"] = tr.rotationZ(total_time * earth_orbit_speed)
        graph.nodes["earth"]["transform"] = (
            tr.rotationZ(total_time * earth_spin_speed) @ tr.uniformScale(EARTH_RADIUS)
        )
        graph.nodes["moon_orbit"]["transform"] = tr.rotationZ(total_time * moon_orbit_speed)
        graph.nodes["moon"]["transform"] = (
            tr.rotationZ(total_time * moon_spin_speed) @ tr.uniformScale(MOON_RADIUS)
        )

        # las transformaciones globales se usan en compute_view (para ubicar
        # la cámara en el marco local de la Tierra), así que las recalculamos
        # acá una sola vez por cuadro
        graph.calculate_global_transforms()

    pyglet.clock.schedule_interval(update_world, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)
