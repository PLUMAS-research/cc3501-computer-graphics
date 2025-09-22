import pyglet
from OpenGL import GL
import numpy as np
import os
from pathlib import Path
import click

from grafica.utils import load_pipeline


@click.command("sr_jengibre_numpy", short_help="Atractor Gingerbreadman generado con NumPy")
@click.option("--width", default=512, help="Ancho de la textura")
@click.option("--height", default=512, help="Alto de la textura")
@click.option("--steps", default=100, help="Pasos por frame")
def gingerbread_numpy(width, height, steps):
    
    # Configuración del atractor. Usamos múltiples partículas para ver mejor 
    num_particles = 10
    particles_x = np.random.uniform(-0.5, 0.5, num_particles)
    particles_y = np.random.uniform(-0.5, 0.5, num_particles)
    
    # Buffer de acumulación "unidimensional" (escala de grises)
    accumulator = np.zeros((height, width), dtype=np.float32)
    
    # Parámetros de mapeo del espacio de fase a píxeles
    x_min, x_max = -5.0, 9.0
    y_min, y_max = -5.0, 9.0
    
    # Control de saturación para evitar que puntos viejos dominen
    max_value_per_pixel = 100.0
    
    def gingerbread_step(x, y):
        """Un paso del mapa de Gingerbreadman"""
        x_new = 1 - y + np.abs(x)
        y_new = x
        return x_new, y_new
    
    def world_to_pixel(x, y, width, height):
        """Convierte coordenadas del espacio de fase a píxeles"""
        px = int((x - x_min) / (x_max - x_min) * width)
        py = int((y - y_min) / (y_max - y_min) * height)
        
        # Verificar límites
        if 0 <= px < width and 0 <= py < height:
            return px, py
        return None, None
    
    def update_attractor(steps_count):
        nonlocal particles_x, particles_y
        
        points_added = 0
        
        for step in range(steps_count):
            # Actualizar todas las partículas
            for i in range(num_particles):
                particles_x[i], particles_y[i] = gingerbread_step(particles_x[i], particles_y[i])
                
                px, py = world_to_pixel(particles_x[i], particles_y[i], width, height)
                if px is not None and py is not None:
                    # Saturar el valor para evitar que un píxel domine
                    if accumulator[py, px] < max_value_per_pixel:
                        accumulator[py, px] += 0.5
                        points_added += 1
                    
                    # Si una partícula se sale del rango, reiniciarla
                    if abs(particles_x[i]) > 20 or abs(particles_y[i]) > 20:
                        particles_x[i] = np.random.uniform(-0.5, 0.5)
                        particles_y[i] = np.random.uniform(-0.5, 0.5)
        
        return points_added
    
    def create_texture_data():
        """Convierte el acumulador a datos de textura RGB"""
        # Normalización con saturación controlada
        if accumulator.max() > 0:
            # Usar raíz cuadrada para suavizar la visualización
            normalized = np.sqrt(accumulator / max_value_per_pixel)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = accumulator
        
        # Crear imagen RGB con gradiente de color
        texture_data = np.zeros((height, width, 3), dtype=np.uint8)
        intensity = (normalized * 255).astype(np.uint8)
        
        # Paleta de colores de azul oscuro a amarillo brillante
        texture_data[:, :, 0] = intensity  # rojo
        texture_data[:, :, 1] = (intensity * 0.8).astype(np.uint8)  # verde
        texture_data[:, :, 2] = ((1.0 - normalized) * 100).astype(np.uint8)  # azul inverso
        
        # Voltear verticalmente (OpenGL tiene origen abajo-izquierda)
        texture_data = np.flipud(texture_data)
        
        return texture_data

    # Configuración de ventana y OpenGL
    win = pyglet.window.Window(width, height)
    
    # Crear textura
    texture_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    
    # Inicializar textura con datos vacíos
    initial_data = np.zeros((height, width, 3), dtype=np.uint8)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
        width, height, 0,
        GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
        initial_data.tobytes()
    )
    
    # Geometría del cuadrilátero
    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    uv = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    # Cargar shaders
    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )
    
    gpu_data = pipeline.vertex_list_indexed(4, GL.GL_TRIANGLES, indices)
    gpu_data.position[:] = vertices
    gpu_data.uv[:] = uv

    # Control de velocidad
    paused = False
    current_steps = steps
    frame_count = 0
    
    # Variable para forzar actualización de textura
    texture_needs_update = True

    def tick(dt):
        nonlocal frame_count, texture_needs_update
        
        if not paused:
            # Actualizar atractor
            points_added = update_attractor(current_steps)
            texture_needs_update = True
            
            frame_count += 1
            if frame_count % 30 == 0:  # Cada medio segundo aproximadamente
                non_zero = np.count_nonzero(accumulator)
                print(f"Frame {frame_count}: Puntos agregados: {points_added}, "
                      f"Píxeles activos: {non_zero}, "
                      f"Max valor: {accumulator.max():.2f}")

    @win.event
    def on_key_press(symbol, modifiers):
        nonlocal paused, current_steps, accumulator, particles_x, particles_y, texture_needs_update
        
        if symbol == pyglet.window.key.SPACE:
            paused = not paused
            print("Pausado" if paused else "Reanudado")
        elif symbol == pyglet.window.key.R:
            # Reset
            accumulator.fill(0.0)
            particles_x = np.random.uniform(-0.5, 0.5, num_particles)
            particles_y = np.random.uniform(-0.5, 0.5, num_particles)
            texture_needs_update = True
            print("Reiniciado")
        elif symbol == pyglet.window.key.PLUS or symbol == pyglet.window.key.EQUAL:
            current_steps = min(current_steps * 2, 10000)
            print(f"Pasos por frame: {current_steps}")
        elif symbol == pyglet.window.key.MINUS:
            current_steps = max(current_steps // 2, 1)
            print(f"Pasos por frame: {current_steps}")
        elif symbol == pyglet.window.key.D:
            # Debug detallado
            non_zero = np.count_nonzero(accumulator)
            print(f"\n=== DEBUG ===")
            print(f"Acumulador - Min: {accumulator.min():.2f}, Max: {accumulator.max():.2f}")
            print(f"Píxeles con datos: {non_zero} de {width*height}")
            print(f"Posiciones de partículas:")
            for i in range(min(3, num_particles)):
                print(f"  Partícula {i}: x={particles_x[i]:.4f}, y={particles_y[i]:.4f}")
            
            # Mostrar región con más actividad
            if accumulator.max() > 0:
                max_pos = np.unravel_index(np.argmax(accumulator), accumulator.shape)
                print(f"Píxel más activo: {max_pos} con valor {accumulator[max_pos]:.2f}")
        elif symbol == pyglet.window.key.C:
            # Limpiar acumulador pero mantener partículas
            accumulator.fill(0.0)
            texture_needs_update = True
            print("Acumulador limpiado")

    @win.event
    def on_draw():
        nonlocal texture_needs_update
        
        win.clear()
        
        # Actualizar textura solo si es necesario
        if texture_needs_update:
            texture_data = create_texture_data()
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                width, height, 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                texture_data.tobytes()
            )
            texture_needs_update = False
        
        # Renderizar
        pipeline.use()
        
        # Activar textura
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        
        # Configurar uniform
        sampler_location = GL.glGetUniformLocation(pipeline.id, "sampler_tex")
        if sampler_location != -1:
            GL.glUniform1i(sampler_location, 0)
        
        gpu_data.draw(GL.GL_TRIANGLES)

    print("Controles:")
    print("  ESPACIO: pausar/reanudar")
    print("  R: reiniciar todo")
    print("  C: limpiar acumulador (mantiene partículas)")
    print("  +/-: aumentar/disminuir pasos por frame")
    print("  D: mostrar estadísticas de debug")
    print(f"  Pasos iniciales por frame: {current_steps}")
    print(f"  Partículas: {num_particles}")

    pyglet.clock.schedule_interval(tick, 1 / 60.0)
    pyglet.app.run()