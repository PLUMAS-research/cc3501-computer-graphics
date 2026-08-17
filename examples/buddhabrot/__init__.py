import ctypes
import os
from pathlib import Path

import click
import numpy as np
import pyglet
from OpenGL import GL

from grafica.utils import load_pipeline


def compute_trajectories(num_samples, max_iter, real_min, real_max, imag_min, imag_max):
    """Calcula trayectorias de Mandelbrot y retorna los puntos que escapan.

    Retorna un array de coordenadas NDC (Normalized Device Coordinates)
    en el rango [-1, 1], listas para enviar a la GPU.
    """
    # Generar puntos c aleatorios en el plano complejo
    seed_real = np.random.uniform(real_min, real_max, num_samples)
    seed_imag = np.random.uniform(imag_min, imag_max, num_samples)

    orbit_real = np.zeros(num_samples, dtype=np.float64)
    orbit_imag = np.zeros(num_samples, dtype=np.float64)

    # Guardar trayectorias (max_iter x num_samples)
    trajectory_real = np.zeros((max_iter, num_samples), dtype=np.float64)
    trajectory_imag = np.zeros((max_iter, num_samples), dtype=np.float64)

    active = np.ones(num_samples, dtype=bool)
    escaped = np.zeros(num_samples, dtype=bool)
    escape_iter = np.full(num_samples, max_iter, dtype=np.int32)

    for i in range(max_iter):
        trajectory_real[i] = orbit_real
        trajectory_imag[i] = orbit_imag

        next_real = orbit_real * orbit_real - orbit_imag * orbit_imag + seed_real
        next_imag = 2.0 * orbit_real * orbit_imag + seed_imag
        orbit_real, orbit_imag = next_real, next_imag

        radius_squared = orbit_real * orbit_real + orbit_imag * orbit_imag
        just_escaped = active & (radius_squared > 4.0)
        if np.any(just_escaped):
            escaped |= just_escaped
            escape_iter[just_escaped] = i + 1
            active[just_escaped] = False

        if not np.any(active):
            break

    # Recolectar todos los puntos de trayectorias que escaparon
    escaped_indices = np.where(escaped)[0]
    if len(escaped_indices) == 0:
        return np.empty((0, 2), dtype=np.float32)

    all_points = []
    for sample_index in escaped_indices:
        num_orbit_steps = escape_iter[sample_index]
        orbit_real_points = trajectory_real[:num_orbit_steps, sample_index]
        orbit_imag_points = trajectory_imag[:num_orbit_steps, sample_index]

        # Filtrar puntos dentro del rango visible
        visible_mask = (
            (orbit_real_points >= real_min) & (orbit_real_points <= real_max) &
            (orbit_imag_points >= imag_min) & (orbit_imag_points <= imag_max)
        )
        orbit_real_points = orbit_real_points[visible_mask]
        orbit_imag_points = orbit_imag_points[visible_mask]

        if len(orbit_real_points) > 0:
            all_points.append(np.column_stack([orbit_real_points, orbit_imag_points]))

    if not all_points:
        return np.empty((0, 2), dtype=np.float32)

    points = np.vstack(all_points)

    # Convertir de coordenadas del plano complejo a NDC [-1, 1].
    # La parte imaginaria va al eje horizontal y la real al vertical, creciendo
    # hacia abajo: así la figura queda de pie, que es como se muestra siempre.
    normalized_x = 2.0 * (points[:, 1] - imag_min) / (imag_max - imag_min) - 1.0
    normalized_y = 1.0 - 2.0 * (points[:, 0] - real_min) / (real_max - real_min)

    return np.column_stack([normalized_x, normalized_y]).astype(np.float32)


@click.command("buddhabrot", short_help="Buddhabrot: fractales por acumulación en GPU")
@click.option("--width", default=512, help="Ancho de la textura")
@click.option("--height", default=512, help="Alto de la textura")
@click.option("--samples", default=2000, help="Muestras aleatorias por frame y canal")
@click.option("--max-iter-r", default=5000, help="Iteraciones máximas (canal rojo)")
@click.option("--max-iter-g", default=500, help="Iteraciones máximas (canal verde)")
@click.option("--max-iter-b", default=50, help="Iteraciones máximas (canal azul)")
def buddhabrot(width, height, samples, max_iter_r, max_iter_g, max_iter_b):
    """Visualización del Buddhabrot (variante Nebulabrot).

    El Buddhabrot es una técnica de visualización del conjunto de Mandelbrot
    inventada por Melinda Green en 1993. En vez de preguntar "¿cuánto demora
    este punto en escapar?", pregunta "¿por dónde pasa mientras escapa?".

    Las trayectorias se calculan en la CPU, pero la acumulación ocurre en la
    GPU usando un framebuffer con blending aditivo sobre una textura float.

    Se usan distintas iteraciones máximas para cada canal RGB (Nebulabrot),
    lo que produce una imagen a color donde las estructuras de baja, media
    y alta frecuencia se separan cromáticamente.
    """

    # Rango del plano complejo. La parte real ocupa el eje vertical, así que
    # este rectángulo se ve girado en la ventana.
    real_min, real_max = -2.05, 0.75
    imag_min, imag_max = -1.4, 1.4

    window = pyglet.window.Window(width, height, caption="Buddhabrot")

    # --- Framebuffer de acumulación con textura float ---
    # Usamos GL_RGBA16F para tener suficiente rango dinámico.
    # Cada punto que pasa por un píxel suma 1.0 a su canal correspondiente.
    accumulation_texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, accumulation_texture)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F,
        width, height, 0,
        GL.GL_RGBA, GL.GL_FLOAT, None
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    framebuffer = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D, accumulation_texture, 0
    )
    # Verificar que el framebuffer esté completo
    framebuffer_status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if framebuffer_status != GL.GL_FRAMEBUFFER_COMPLETE:
        print(f"Error: framebuffer incompleto (status={framebuffer_status})")
    # Limpiar el framebuffer
    GL.glClearColor(0.0, 0.0, 0.0, 0.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # --- Pipeline de puntos (acumulación) ---
    pipeline_points = load_pipeline(
        Path(os.path.dirname(__file__)) / "point_vertex.glsl",
        Path(os.path.dirname(__file__)) / "point_fragment.glsl",
    )

    # VAO y VBO para los puntos (se actualiza cada frame)
    vertex_array_points = GL.glGenVertexArrays(1)
    vertex_buffer_points = GL.glGenBuffers(1)

    GL.glBindVertexArray(vertex_array_points)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertex_buffer_points)
    position_attribute_location = GL.glGetAttribLocation(pipeline_points.id, "position")
    GL.glEnableVertexAttribArray(position_attribute_location)
    GL.glVertexAttribPointer(position_attribute_location, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
    GL.glBindVertexArray(0)

    # --- Pipeline de visualización ---
    pipeline_visualization = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "visualization.glsl",
    )

    # Cuadrilátero que cubre toda la pantalla
    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    uv = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    screen_quad = pipeline_visualization.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    screen_quad.position[:] = vertices
    screen_quad.uv[:] = uv

    # Estado
    paused = False
    current_samples = samples
    frame_count = 0
    total_samples = 0
    exposure = 0.001

    # Canales Nebulabrot: (nombre, max_iter, color RGBA para blending, ratio de muestras)
    channels = [
        ("B", max_iter_b, (0.0, 0.0, 1.0, 0.0), 1.0),
        ("G", max_iter_g, (0.0, 1.0, 0.0, 0.0), 1.0),
        ("R", max_iter_r, (1.0, 0.0, 0.0, 0.0), 0.2),  # menos muestras para el canal lento
    ]

    def render_points(points_ndc, color):
        """Sube puntos al VBO y los dibuja en el framebuffer de acumulación."""
        if len(points_ndc) == 0:
            return

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertex_buffer_points)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            points_ndc.nbytes,
            points_ndc,
            GL.GL_STREAM_DRAW
        )

        pipeline_points.use()
        pipeline_points['channel_color'] = color

        GL.glBindVertexArray(vertex_array_points)
        GL.glDrawArrays(GL.GL_POINTS, 0, len(points_ndc))
        GL.glBindVertexArray(0)

    def tick(delta_time):
        nonlocal frame_count, total_samples

        if paused:
            return

        # --- Paso de acumulación: renderizar puntos al framebuffer ---
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)  # blending aditivo

        for channel_name, channel_max_iter, channel_color, sample_ratio in channels:
            channel_sample_count = max(100, int(current_samples * sample_ratio))
            points = compute_trajectories(
                channel_sample_count, channel_max_iter,
                real_min, real_max, imag_min, imag_max
            )
            render_points(points, channel_color)

        GL.glDisable(GL.GL_BLEND)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        total_samples += current_samples
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {total_samples:,} muestras, "
                  f"exposición={exposure:.4f}")

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal paused, current_samples, exposure, total_samples, frame_count

        if symbol == pyglet.window.key.SPACE:
            paused = not paused
            print("Pausado" if paused else "Reanudado")
        elif symbol == pyglet.window.key.R:
            # Limpiar framebuffer de acumulación
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            total_samples = 0
            frame_count = 0
            print("Reiniciado")
        elif symbol == pyglet.window.key.PLUS or symbol == pyglet.window.key.EQUAL:
            current_samples = min(current_samples * 2, 50000)
            print(f"Muestras por frame: {current_samples}")
        elif symbol == pyglet.window.key.MINUS:
            current_samples = max(current_samples // 2, 100)
            print(f"Muestras por frame: {current_samples}")
        elif symbol == pyglet.window.key.UP:
            exposure *= 2.0
            print(f"Exposición: {exposure:.6f}")
        elif symbol == pyglet.window.key.DOWN:
            exposure /= 2.0
            print(f"Exposición: {exposure:.6f}")

    @window.event
    def on_draw():
        window.clear()
        GL.glViewport(0, 0, width, height)

        # --- Paso de visualización ---
        # Leer la textura de acumulación y aplicar tone mapping
        pipeline_visualization.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, accumulation_texture)

        sampler_uniform_location = GL.glGetUniformLocation(pipeline_visualization.id, "accum_tex")
        if sampler_uniform_location != -1:
            GL.glUniform1i(sampler_uniform_location, 0)

        exposure_uniform_location = GL.glGetUniformLocation(pipeline_visualization.id, "exposure")
        if exposure_uniform_location != -1:
            GL.glUniform1f(exposure_uniform_location, exposure)

        screen_quad.draw(GL.GL_TRIANGLES)

    print("Buddhabrot (Nebulabrot) con acumulación en GPU")
    print("Controles:")
    print("  ESPACIO: pausar/reanudar")
    print("  R: reiniciar")
    print("  +/-: más/menos muestras por frame")
    print("  ARRIBA/ABAJO: ajustar exposición")
    print(f"  Muestras por frame: {current_samples}")
    print(f"  Iteraciones máx: R={max_iter_r}, G={max_iter_g}, B={max_iter_b}")
    print(f"  Exposición inicial: {exposure}")
    print("  (La imagen se forma gradualmente)")

    pyglet.clock.schedule_interval(tick, 1 / 60.0)
    pyglet.app.run()
