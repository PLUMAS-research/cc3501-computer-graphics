"""Clase Arcball para manipulación 3D de puntos de vista.
Versión refactorizada con zoom logarítmico uniforme.
"""

import numpy as np
import grafica.transformations as tr


class Arcball:
    """Controlador de cámara 3D mediante mouse con zoom logarítmico uniforme."""

    STATE_ROTATE = 0
    STATE_PAN = 1
    STATE_ROLL = 2
    STATE_ZOOM = 3

    def __init__(self, pose, size, scale, target=np.array([0.0, 0.0, 0.0])):
        """Inicializa el arcball.

        Parámetros
        ----------
        pose : (4,4) array
            Matriz de transformación inicial cámara-a-mundo (inversa de view).
        size : (2,) tuple
            Dimensiones de la ventana en píxeles (ancho, alto).
        scale : float
            Escala de la escena para ajustar velocidades de movimiento.
        target : (3,) array
            Centro de rotación en coordenadas del mundo.
            IMPORTANTE: Este punto debe ser el centro del objeto que quieres rotar.
            Si tu objeto no está en el origen, debes establecer target en la
            posición del objeto para que las rotaciones sean alrededor del objeto.
        """
        self._size = np.array(size, dtype=np.float32)
        self._scale = float(scale)
        
        # Estados de transformación
        self._pose = np.array(pose, dtype=np.float64)
        self._target = np.array(target, dtype=np.float64)
        
        # Estado para drag & drop
        self._drag_start = None
        self._drag_pose = None
        self._drag_target = None
        
        # Estado de interacción
        self._state = Arcball.STATE_ROTATE
        
        # Configuración de zoom logarítmico
        # Usamos log(distancia) para movimiento uniforme en toda la escala
        self._zoom_speed = 0.1  # Velocidad de zoom (ajustable)
        
        # Para reset
        self._initial_pose = np.copy(self._pose)
        self._initial_target = np.copy(self._target)
        
        # Límites opcionales (por defecto sin límites prácticos)
        self._min_distance = None
        self._max_distance = None

    @property
    def pose(self):
        """Obtiene la pose actual."""
        return self._pose
    
    @pose.setter
    def pose(self, value):
        """Establece una nueva pose."""
        self._pose = np.array(value, dtype=np.float64)

    @property
    def target(self):
        """Obtiene el objetivo actual."""
        return self._target
    
    @target.setter
    def target(self, value):
        """Establece un nuevo objetivo."""
        self._target = np.array(value, dtype=np.float64)

    def set_state(self, state):
        """Cambia el modo de interacción."""
        self._state = state

    def resize(self, size):
        """Actualiza las dimensiones de la ventana."""
        self._size = np.array(size, dtype=np.float32)

    def reset(self):
        """Restaura el estado inicial."""
        self._pose = np.copy(self._initial_pose)
        self._target = np.copy(self._initial_target)

    def set_initial_state(self, pose=None, target=None):
        """Define un nuevo estado inicial para reset."""
        if pose is not None:
            self._initial_pose = np.copy(pose)
        else:
            self._initial_pose = np.copy(self._pose)
        
        if target is not None:
            self._initial_target = np.copy(target)
        else:
            self._initial_target = np.copy(self._target)

    def set_distance_limits(self, min_distance=None, max_distance=None):
        """Establece límites opcionales de distancia.
        
        Si se dejan en None, no hay límites efectivos.
        """
        self._min_distance = min_distance
        self._max_distance = max_distance

    def get_camera_distance(self):
        """Calcula la distancia actual al objetivo."""
        eye = self._pose[:3, 3]
        return np.linalg.norm(eye - self._target)

    def set_camera_distance(self, distance):
        """Establece una distancia específica al objetivo."""
        eye = self._pose[:3, 3]
        to_eye = eye - self._target
        current_dist = np.linalg.norm(to_eye)
        
        if current_dist < 1e-10:
            # Caso degenerado: usar eje Z de la cámara
            direction = self._pose[:3, 2]
        else:
            direction = to_eye / current_dist
        
        # Aplicar límites si existen
        if self._min_distance is not None:
            distance = max(distance, self._min_distance)
        if self._max_distance is not None:
            distance = min(distance, self._max_distance)
        
        # Nueva posición
        new_eye = self._target + direction * distance
        self._pose[:3, 3] = new_eye

    def down(self, point):
        """Inicia una operación de arrastre."""
        self._drag_start = np.array(point, dtype=np.float32)
        self._drag_pose = np.copy(self._pose)
        self._drag_target = np.copy(self._target)

    def drag(self, point):
        """Actualiza la transformación durante el arrastre."""
        if self._drag_start is None:
            return
        
        point = np.array(point, dtype=np.float32)
        delta = point - self._drag_start
        
        # Factor de escala para el movimiento
        size_factor = 0.3 * min(self._size)
        
        if self._state == Arcball.STATE_ROTATE:
            self._drag_rotate(delta, size_factor)
        elif self._state == Arcball.STATE_PAN:
            self._drag_pan(delta, size_factor)
        elif self._state == Arcball.STATE_ROLL:
            self._drag_roll(point)
        elif self._state == Arcball.STATE_ZOOM:
            self._drag_zoom(delta[1], size_factor)

    def _drag_rotate(self, delta, size_factor):
        """Rotación esférica alrededor del objetivo.
        
        Las rotaciones siempre son alrededor del target, que debe ser
        el centro del objeto que quieres rotar.
        """
        # Ángulos de rotación
        yaw = -delta[0] / size_factor    # Rotación horizontal
        pitch = delta[1] / size_factor   # Rotación vertical
        
        # Para asegurar que la rotación sea alrededor del target,
        # creamos matrices de traslación explícitas
        
        # Matriz para trasladar al origen (mover por -target)
        to_origin = np.eye(4)
        to_origin[:3, 3] = -self._drag_target
        
        # Matriz para trasladar de vuelta (mover por +target)  
        from_origin = np.eye(4)
        from_origin[:3, 3] = self._drag_target
        
        # Rotación horizontal alrededor del eje Y mundial
        # La rotación se hace en el origen, luego se traslada de vuelta
        yaw_matrix = from_origin @ tr.rotationY(yaw) @ to_origin
        
        # Aplicar yaw primero
        temp_pose = yaw_matrix @ self._drag_pose
        
        # Rotación vertical alrededor del eje X local de la cámara
        local_x = temp_pose[:3, 0]
        
        # Matriz de rotación para pitch alrededor del eje X local
        pitch_rot = self._axis_angle_matrix(pitch, local_x)
        pitch_matrix = from_origin @ pitch_rot @ to_origin
        
        # Combinar ambas rotaciones
        self._pose = pitch_matrix @ temp_pose

    def _drag_pan(self, delta, size_factor):
        """Traslación en el plano de la cámara."""
        # Escalar el movimiento
        dx = -delta[0] / (3.0 * size_factor) * self._scale
        dy = -delta[1] / (3.0 * size_factor) * self._scale
        
        # Vectores de la cámara
        right = self._drag_pose[:3, 0]
        up = self._drag_pose[:3, 1]
        
        # Desplazamiento total
        translation = dx * right + dy * up
        
        # Actualizar objetivo y pose
        self._target = self._drag_target + translation
        self._pose[:3, 3] = self._drag_pose[:3, 3] + translation

    def _drag_roll(self, point):
        """Rotación alrededor del eje de vista que pasa por el target.
        
        El roll es rotación alrededor del eje que va desde la cámara al target.
        """
        # Calcular el ángulo de rotación basado en el movimiento del mouse
        center = self._size / 2.0
        v_start = self._drag_start - center
        v_current = point - center
        
        # Normalizar vectores para calcular ángulo
        norm_start = np.linalg.norm(v_start)
        norm_current = np.linalg.norm(v_current)
        
        if norm_start < 1e-6 or norm_current < 1e-6:
            return
        
        v_start = v_start / norm_start
        v_current = v_current / norm_current
        
        # Ángulo de rotación
        roll_angle = np.arctan2(v_current[1], v_current[0]) - np.arctan2(v_start[1], v_start[0])
        
        # Eje de rotación: vector desde target hacia la cámara
        eye = self._drag_pose[:3, 3]
        axis = eye - self._drag_target
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # Caso degenerado: usar eje Z de la cámara
            axis = self._drag_pose[:3, 2]
        else:
            axis = axis / axis_norm
        
        # Matrices de traslación explícitas
        to_origin = np.eye(4)
        to_origin[:3, 3] = -self._drag_target
        
        from_origin = np.eye(4)
        from_origin[:3, 3] = self._drag_target
        
        # Rotar alrededor del eje que pasa por el target
        roll_rot = self._axis_angle_matrix(roll_angle, axis)
        roll_matrix = from_origin @ roll_rot @ to_origin
        
        self._pose = roll_matrix @ self._drag_pose

    def _drag_zoom(self, dy, size_factor):
        """Zoom mediante arrastre vertical."""
        # Usar escala logarítmica para zoom uniforme
        factor = dy / size_factor
        self._apply_zoom(factor)

    def scroll(self, clicks):
        """Zoom con la rueda del mouse usando escala logarítmica.
        
        El zoom es uniforme en escala logarítmica, lo que significa
        que cada clic mueve la misma "cantidad perceptual" sin importar
        la distancia actual.
        """
        if clicks == 0:
            return
        
        # Factor de zoom en escala logarítmica
        # Cada clic cambia el logaritmo de la distancia por una cantidad fija
        factor = -clicks * self._zoom_speed
        self._apply_zoom(factor)

    def _apply_zoom(self, log_factor):
        """Aplica zoom usando factor logarítmico.
        
        Este es el corazón del nuevo sistema de zoom. Usamos:
        nueva_distancia = exp(log(distancia_actual) + factor)
        
        Esto garantiza movimiento perceptualmente uniforme en todas las escalas.
        """
        eye = self._pose[:3, 3]
        to_eye = eye - self._target
        current_distance = np.linalg.norm(to_eye)
        
        # Manejar caso degenerado
        if current_distance < 1e-10:
            direction = self._pose[:3, 2]
            current_distance = 1e-6
        else:
            direction = to_eye / current_distance
        
        # Calcular nueva distancia en escala logarítmica
        # log_distance = log(current) + factor
        # new_distance = exp(log_distance) = exp(log(current) + factor) = current * exp(factor)
        
        # Pero para mejor control cerca de cero, usamos una interpolación suave
        if current_distance < 0.01:
            # Cerca de cero, usar movimiento lineal
            new_distance = current_distance * (1.0 - log_factor * 10)
            new_distance = max(1e-8, new_distance)  # Prevenir negativos
        else:
            # Lejos de cero, usar escala logarítmica completa
            log_distance = np.log(current_distance) + log_factor
            new_distance = np.exp(log_distance)
        
        # Aplicar límites si están definidos
        if self._min_distance is not None:
            new_distance = max(new_distance, self._min_distance)
        if self._max_distance is not None:
            new_distance = min(new_distance, self._max_distance)
        
        # Actualizar posición de la cámara
        new_eye = self._target + direction * new_distance
        self._pose[:3, 3] = new_eye

    def rotate(self, angle, axis=None):
        """Rota la cámara alrededor de un eje que pasa por el objetivo."""
        # Determinar matriz de rotación
        if axis is None or axis == 'y':
            rot_matrix = tr.rotationY(angle)
        elif axis == 'x':
            rot_matrix = tr.rotationX(angle)
        elif axis == 'z':
            rot_matrix = tr.rotationZ(angle)
        else:
            # Eje arbitrario
            rot_matrix = self._axis_angle_matrix(angle, axis)
        
        # Matrices de traslación explícitas para rotar alrededor del target
        to_origin = np.eye(4)
        to_origin[:3, 3] = -self._target
        
        from_origin = np.eye(4)
        from_origin[:3, 3] = self._target
        
        # Aplicar rotación alrededor del objetivo
        transform = from_origin @ rot_matrix @ to_origin
        
        self._pose = transform @ self._pose

    def _axis_angle_matrix(self, angle, axis):
        """Crea matriz de rotación 4x4 desde ángulo y eje (Rodrigues)."""
        axis = np.array(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis
        
        rotation = np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])
        
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        return matrix

    def stabilize_rotation(self):
        """Reortogonaliza la matriz de rotación para evitar drift numérico.
        
        Este método se puede llamar periódicamente si se nota distorsión
        después de muchas operaciones.
        """
        # Extraer componente de rotación
        rotation = self._pose[:3, :3]
        
        # Gram-Schmidt para reortogonalizar
        x = rotation[:, 0]
        y = rotation[:, 1]
        
        # Normalizar X
        x = x / np.linalg.norm(x)
        
        # Hacer Y ortogonal a X y normalizar
        y = y - np.dot(y, x) * x
        y = y / np.linalg.norm(y)
        
        # Z es producto cruz
        z = np.cross(x, y)
        
        # Reconstruir matriz ortogonal
        self._pose[:3, 0] = x
        self._pose[:3, 1] = y
        self._pose[:3, 2] = z