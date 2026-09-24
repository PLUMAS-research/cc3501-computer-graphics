"""Oclusión ambiental en espacio de pantalla (SSAO).

La luz ambiental de Phong es una constante: el mismo valor en todo punto de la
escena, así que las esquinas interiores y los puntos de contacto reciben tanta
luz indirecta como una pared despejada. SSAO estima, para cada píxel, qué
fracción del hemisferio le tapa la geometría cercana y atenúa la ambiental con
ese factor.

El ejemplo son cuatro pasadas por cuadro:

1. Geometría. La escena se dibuja en un framebuffer con dos adjuntos, que
   guardan la posición y la normal de cada píxel en espacio de vista.
2. Oclusión. Un rectángulo de pantalla completa toma muestras alrededor de
   cada píxel y cuenta cuántas quedan tapadas.
3. Desenfoque. Promedia el factor para borrar el patrón que deja el ruido.
4. Iluminación. La escena otra vez, con el término ambiental multiplicado por
   el factor.

La tecla ESPACIO muestra las texturas intermedias y las flechas giran la
cámara, que es la forma de ver la limitación de la técnica: el factor se
calcula desde el buffer de profundidad, que solo contiene lo que la cámara ve,
así que la oclusión cambia al mover el punto de vista.
"""

import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL

import grafica.transformations as tr
from grafica.framebuffers import (
    create_geometry_framebuffer,
    create_single_channel_framebuffer,
)
from grafica.scenegraph import Scenegraph
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

# Tiene que coincidir con el #define del shader de oclusión: el arreglo de
# muestras se declara de este largo y sample_count decide cuántas se usan.
MAX_SAMPLES = 64

# Lado de la textura de ruido. El mismo número es el radio del desenfoque que
# la cancela, porque el patrón se repite cada NOISE_SIZE píxeles.
NOISE_SIZE = 4

SSAO_DEFAULTS = {
    "sample_radius": 0.20,
    "depth_bias": 0.005,
    "occlusion_strength": 2.0,
    "sample_count": 32,
    "use_occlusion": True,
    "use_blur": True,
}

SSAO_STEPS = {"sample_radius": 0.04, "depth_bias": 0.0025, "occlusion_strength": 0.25}
SSAO_RANGES = {
    "sample_radius": (0.04, 0.6),
    "depth_bias": (0.0, 0.05),
    "occlusion_strength": (0.5, 4.0),
}

VIEW_MODES = ("escena", "factor de oclusión", "normales", "posiciones")


def generar_kernel(cantidad, generador):
    """Direcciones de muestreo dentro del hemisferio de la normal.

    Las muestras se concentran cerca del origen, porque la oclusión de un punto
    la produce sobre todo lo que tiene al lado. El largo sale de un sorteo y no
    del índice de la muestra: el shader usa las primeras `sample_count` del
    arreglo, así que un largo que creciera con el índice dejaría fuera las
    muestras lejanas cada vez que se baja la cantidad.
    """
    muestras = np.zeros((cantidad, 3), dtype=np.float32)
    for sample_index in range(cantidad):
        direccion = np.array(
            [
                generador.uniform(-1.0, 1.0),
                generador.uniform(-1.0, 1.0),
                generador.uniform(0.0, 1.0),  # z positivo: medio espacio de la normal
            ]
        )
        direccion /= np.linalg.norm(direccion)

        proporcion = generador.uniform(0.0, 1.0)
        direccion *= 0.1 + 0.9 * proporcion * proporcion
        muestras[sample_index] = direccion
    return muestras


def crear_textura_de_ruido(generador):
    """Textura de NOISE_SIZE x NOISE_SIZE con vectores de rotación aleatorios.

    Los vectores están en el plano tangente (z = 0) y giran la base del
    hemisferio de un píxel al siguiente. Se repite sobre toda la pantalla, así
    que su wrap es GL_REPEAT y no el CLAMP del resto de las texturas.
    """
    vectores = np.zeros((NOISE_SIZE * NOISE_SIZE, 3), dtype=np.float32)
    for indice in range(NOISE_SIZE * NOISE_SIZE):
        vectores[indice] = [
            generador.uniform(-1.0, 1.0),
            generador.uniform(-1.0, 1.0),
            0.0,
        ]

    texture_id = GL.GLuint(0)
    GL.glGenTextures(1, texture_id)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    datos = (GL.GLfloat * vectores.size)(*vectores.flatten())
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F,
        NOISE_SIZE, NOISE_SIZE, 0,
        GL.GL_RGB, GL.GL_FLOAT, datos,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return texture_id.value


def crear_quad_pantalla(pipeline):
    """Rectángulo que cubre la pantalla, en NDC, para una pasada completa."""
    posiciones = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    coordenadas = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    quad = pipeline.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    quad.position[:] = posiciones
    quad.uv[:] = coordenadas
    return quad


@click.command("ssao", short_help="Oclusión ambiental en espacio de pantalla")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
@click.option("--seed", type=int, default=7, help="Semilla del kernel y del ruido")
def ssao(width, height, seed):
    """Oclusión ambiental calculada por cuadro desde el buffer de geometría."""

    window = pyglet.window.Window(width, height)
    generador = np.random.default_rng(seed)

    here = Path(os.path.dirname(__file__))

    # -------------------------------------------------------------------------
    # Escena
    # -------------------------------------------------------------------------
    # La caja de Cornell sirve porque es casi toda esquinas interiores, que es
    # donde la ambiental constante se nota plana y la oclusión aparece.

    graph = Scenegraph("root")
    graph.load_and_register_mesh("cornell_box", "assets/CornellBox_original.obj")
    graph.load_and_register_mesh(
        "squirtle", "assets/Squirtle.STL", force_color=np.array([220, 220, 225, 255])
    )
    graph.load_and_register_mesh("sphere", "assets/sphere.off")

    graph.load_and_register_pipeline(
        "gbuffer_shader",
        here / "gbuffer_vertex_program.glsl",
        here / "gbuffer_fragment_program.glsl",
    )
    graph.load_and_register_pipeline(
        "lighting_shader",
        here / "lighting_vertex_program.glsl",
        here / "lighting_fragment_program.glsl",
    )

    # El cargador centra cada malla y la escala a una diagonal de 2, así que la
    # caja de Cornell ocupa +-0.57 y su piso queda en y = -0.571. Los dos
    # objetos se apoyan ahí: la mitad de su alto por sobre el piso.
    piso = -0.571
    escala_pokemon, alto_pokemon = 0.42, 0.64
    escala_esfera, radio_esfera = 0.30, 0.577

    graph.add_mesh_instance("caja", "cornell_box", "lighting_shader", parent="root")
    graph.add_mesh_instance(
        "pokemon",
        "squirtle",
        "lighting_shader",
        transform=(
            tr.translate(0.16, piso + escala_pokemon * alto_pokemon, 0.28)
            @ tr.uniformScale(escala_pokemon)
        ),
        parent="root",
    )
    graph.add_mesh_instance(
        "esfera",
        "sphere",
        "lighting_shader",
        transform=(
            tr.translate(-0.25, piso + escala_esfera * radio_esfera, 0.26)
            @ tr.uniformScale(escala_esfera)
        ),
        parent="root",
    )

    near_plane, far_plane = 0.1, 6.0
    projection_camera = tr.perspective(
        45, float(width) / float(height), near_plane, far_plane
    )
    camera_distance = 2.0
    light_world_position = np.array([0.0, 0.5, 0.2])

    def matriz_de_vista(azimuth_degrees):
        angulo = np.radians(azimuth_degrees)
        eye = np.array(
            [camera_distance * np.sin(angulo), 0.0, camera_distance * np.cos(angulo)]
        )
        return tr.lookAt(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

    graph.register_view_transform(matriz_de_vista(0.0), name="camera_view")
    graph.set_global_attributes(
        projection=projection_camera,
        resolution=np.array([float(width), float(height)]),
        ambient_strength=0.45,
    )

    # -------------------------------------------------------------------------
    # Framebuffers y pasadas de pantalla completa
    # -------------------------------------------------------------------------

    geometry_framebuffer, position_texture, normal_texture = (
        create_geometry_framebuffer(width, height)
    )
    occlusion_framebuffer, occlusion_texture = create_single_channel_framebuffer(
        width, height
    )
    blur_framebuffer, blurred_texture = create_single_channel_framebuffer(width, height)

    ssao_pipeline = load_pipeline(
        here / "screen_vertex_program.glsl", here / "ssao_fragment_program.glsl"
    )
    blur_pipeline = load_pipeline(
        here / "screen_vertex_program.glsl", here / "blur_fragment_program.glsl"
    )
    debug_pipeline = load_pipeline(
        here / "screen_vertex_program.glsl", here / "debug_fragment_program.glsl"
    )

    ssao_quad = crear_quad_pantalla(ssao_pipeline)
    blur_quad = crear_quad_pantalla(blur_pipeline)
    debug_quad = crear_quad_pantalla(debug_pipeline)

    kernel = generar_kernel(MAX_SAMPLES, generador)
    noise_texture_id = crear_textura_de_ruido(generador)

    # El arreglo de un uniform `vec3 kernel[64]` se sube como 64 elementos de
    # tres flotantes, no como 192 flotantes seguidos.
    kernel_buffer = (GL.GLfloat * 3 * MAX_SAMPLES)(*(tuple(fila) for fila in kernel))

    # El ruido es una textura de 4 x 4 que se repite: la escala lleva las
    # coordenadas de pantalla a esa grilla.
    noise_scale = np.array(
        [width / float(NOISE_SIZE), height / float(NOISE_SIZE)], dtype=np.float32
    )

    ssao_pipeline.use()
    ssao_pipeline["position_texture"] = 0
    ssao_pipeline["normal_texture"] = 1
    ssao_pipeline["noise_texture"] = 2
    ssao_pipeline["projection"] = projection_camera.reshape(16, 1, order="F")
    ssao_pipeline["kernel"] = kernel_buffer
    ssao_pipeline["noise_scale"] = noise_scale.reshape(2, 1, order="F")
    ssao_pipeline.stop()

    blur_pipeline.use()
    blur_pipeline["occlusion_texture"] = 0
    blur_pipeline["blur_radius"] = NOISE_SIZE // 2
    blur_pipeline.stop()

    # -------------------------------------------------------------------------
    # Estado interactivo
    # -------------------------------------------------------------------------

    state = dict(SSAO_DEFAULTS)
    view_mode_index = 0
    camera_azimuth = 0.0

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(20, 20, 20))
        .add("radio")
        .add("sesgo")
        .add("intensidad")
        .add("muestras")
        .add("desenfoque")
        .add("vista")
        .footer("1/2 radio   3/4 sesgo   5/6 intensidad   7 muestras   B desenfoque"
                "   O oclusión   SPACE vista   flechas cámara   R reset")
    )

    def apply_state():
        panel["radio"] = f"radio de muestreo:  {state['sample_radius']:.2f}"
        panel["sesgo"] = f"sesgo:              {state['depth_bias']:.3f}"
        panel["intensidad"] = f"intensidad:         {state['occlusion_strength']:.2f}"
        panel["muestras"] = f"muestras:           {state['sample_count']}"
        panel["desenfoque"] = (
            f"desenfoque:         {'on' if state['use_blur'] else 'off'}"
            f"   oclusion: {'on' if state['use_occlusion'] else 'off'}"
        )
        panel["vista"] = (
            f"vista:              {VIEW_MODES[view_mode_index]}"
            f"   camara: {camera_azimuth:+.0f} deg"
        )
        print(
            f"[ssao]  radio={state['sample_radius']:.2f}"
            f"  sesgo={state['depth_bias']:.3f}"
            f"  intensidad={state['occlusion_strength']:.2f}"
            f"  muestras={state['sample_count']}"
            f"  desenfoque={state['use_blur']}"
            f"  oclusion={state['use_occlusion']}"
            f"  vista={VIEW_MODES[view_mode_index]}"
            f"  azimuth={camera_azimuth:.0f}"
        )

    def ajustar_parametro(nombre, signo):
        paso = SSAO_STEPS[nombre]
        minimo, maximo = SSAO_RANGES[nombre]
        state[nombre] = float(np.clip(state[nombre] + signo * paso, minimo, maximo))
        apply_state()

    apply_state()

    # -------------------------------------------------------------------------
    # Eventos
    # -------------------------------------------------------------------------

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal view_mode_index, camera_azimuth
        teclas = {
            pyglet.window.key._1: ("sample_radius", -1),
            pyglet.window.key._2: ("sample_radius", +1),
            pyglet.window.key._3: ("depth_bias", -1),
            pyglet.window.key._4: ("depth_bias", +1),
            pyglet.window.key._5: ("occlusion_strength", -1),
            pyglet.window.key._6: ("occlusion_strength", +1),
        }
        if symbol in teclas:
            nombre, signo = teclas[symbol]
            ajustar_parametro(nombre, signo)
        elif symbol == pyglet.window.key._7:
            # 8, 16, 32 y 64 muestras: el costo de la pasada es proporcional
            cantidades = [8, 16, 32, 64]
            siguiente = (cantidades.index(state["sample_count"]) + 1) % len(cantidades)
            state["sample_count"] = cantidades[siguiente]
            apply_state()
        elif symbol == pyglet.window.key.B:
            state["use_blur"] = not state["use_blur"]
            apply_state()
        elif symbol == pyglet.window.key.O:
            state["use_occlusion"] = not state["use_occlusion"]
            apply_state()
        elif symbol == pyglet.window.key.SPACE:
            view_mode_index = (view_mode_index + 1) % len(VIEW_MODES)
            apply_state()
        elif symbol == pyglet.window.key.LEFT:
            camera_azimuth = float(np.clip(camera_azimuth - 5.0, -40.0, 40.0))
            apply_state()
        elif symbol == pyglet.window.key.RIGHT:
            camera_azimuth = float(np.clip(camera_azimuth + 5.0, -40.0, 40.0))
            apply_state()
        elif symbol == pyglet.window.key.R:
            state.update(SSAO_DEFAULTS)
            view_mode_index = 0
            camera_azimuth = 0.0
            apply_state()

    def dibujar_textura(texture_id, debug_mode):
        """Muestra una textura intermedia en pantalla completa."""
        GL.glDisable(GL.GL_DEPTH_TEST)
        debug_pipeline.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        debug_pipeline["debug_texture"] = 0
        debug_pipeline["debug_mode"] = debug_mode
        debug_quad.draw(GL.GL_TRIANGLES)
        GL.glEnable(GL.GL_DEPTH_TEST)

    @window.event
    def on_draw():
        view_camera = matriz_de_vista(camera_azimuth)
        graph.views["camera_view"] = view_camera
        graph.current_view = "camera_view"

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        # ---------------------------------------------------------------------
        # Pasada 1: posiciones y normales en espacio de vista.
        # ---------------------------------------------------------------------
        geometry_framebuffer.bind()
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        graph.render(only_pipelines={"lighting_shader"}, pipeline_override="gbuffer_shader")
        geometry_framebuffer.unbind()

        # ---------------------------------------------------------------------
        # Pasada 2: el factor de oclusión, un valor por píxel.
        # ---------------------------------------------------------------------
        occlusion_framebuffer.bind()
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)

        ssao_pipeline.use()
        for unidad, textura in enumerate(
            (position_texture.id, normal_texture.id, noise_texture_id)
        ):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unidad)
            GL.glBindTexture(GL.GL_TEXTURE_2D, textura)
        ssao_pipeline["sample_radius"] = float(state["sample_radius"])
        ssao_pipeline["depth_bias"] = float(state["depth_bias"])
        ssao_pipeline["occlusion_strength"] = float(state["occlusion_strength"])
        ssao_pipeline["sample_count"] = int(state["sample_count"])
        ssao_quad.draw(GL.GL_TRIANGLES)
        occlusion_framebuffer.unbind()

        # ---------------------------------------------------------------------
        # Pasada 3: desenfoque que borra el patrón del ruido.
        # ---------------------------------------------------------------------
        blur_framebuffer.bind()
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        blur_pipeline.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, occlusion_texture.id)
        blur_quad.draw(GL.GL_TRIANGLES)
        blur_framebuffer.unbind()
        GL.glEnable(GL.GL_DEPTH_TEST)

        textura_de_oclusion = (
            blurred_texture.id if state["use_blur"] else occlusion_texture.id
        )

        # ---------------------------------------------------------------------
        # Pasada 4: la escena iluminada, con la ambiental atenuada.
        # ---------------------------------------------------------------------
        GL.glViewport(0, 0, window.width, window.height)
        window.clear()

        if view_mode_index == 0:
            for nodo in ("caja", "pokemon", "esfera"):
                graph.add_texture_to_node(nodo, "occlusion_texture", textura_de_oclusion)
            light_view_position = (view_camera @ np.append(light_world_position, 1.0))[:3]
            graph.set_global_attributes(
                light_view_position=light_view_position.astype(np.float32),
                use_occlusion=int(state["use_occlusion"]),
            )
            graph.render(recalculate_transforms=False, only_pipelines={"lighting_shader"})
        elif view_mode_index == 1:
            dibujar_textura(textura_de_oclusion, 0)
        elif view_mode_index == 2:
            dibujar_textura(normal_texture.id, 1)
        else:
            dibujar_textura(position_texture.id, 1)

        with ui_overlay():
            panel.draw()

    pyglet.app.run()
