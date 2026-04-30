"""Clase Arcball para manipulación 3D del punto de vista con el mouse.

La clase mantiene la pose de la cámara (matriz 4x4 cámara-a-mundo) y la
modifica según el gesto del usuario: rotación, paneo, roll y zoom. Cada
gesto trabaja relativo a un punto objetivo (target) que define el centro
de las rotaciones.
"""

import numpy as np
import grafica.transformations as tr


class Arcball:
    """Controlador de cámara 3D mediante el mouse."""

    STATE_ROTATE = 0
    STATE_PAN    = 1
    STATE_ROLL   = 2
    STATE_ZOOM   = 3

    def __init__(self, pose, size, scale, target=np.array([0.0, 0.0, 0.0])):
        """Inicializa el arcball.

        Parámetros
        ----------
        pose : (4, 4) array
            Pose inicial cámara-a-mundo. Es la inversa de la matriz de vista,
            por lo que sus columnas son (eje X, eje Y, eje Z, posición) de la
            cámara expresados en coordenadas de mundo.
        size : (2,) tuple
            Dimensiones de la ventana en píxeles (ancho, alto). Se usan para
            normalizar el desplazamiento del mouse.
        scale : float
            Escala de la escena. Controla la velocidad del paneo respecto al
            tamaño del objeto.
        target : (3,) array
            Punto en coordenadas de mundo alrededor del cual se rota. Si el
            objeto que quieres explorar no está en el origen, conviene poner
            target en su centro: así la rotación lo gira en su lugar y no lo
            aleja de la pantalla.
        """
        self._size  = np.array(size, dtype=np.float32)
        self._scale = float(scale)

        # Pose actual y centro de rotación (estado "vivo" del arcball).
        self._pose   = np.array(pose,   dtype=np.float64)
        self._target = np.array(target, dtype=np.float64)

        # Snapshot de pose y target al iniciar un arrastre. Sirve para que el
        # cálculo del gesto se haga contra el estado al apretar el botón, no
        # contra el frame anterior. Así el arrastre es estable y reversible.
        self._drag_start  = None
        self._drag_pose   = None
        self._drag_target = None

        self._state = Arcball.STATE_ROTATE

        # Estado al que vuelve `reset`.
        self._initial_pose   = np.copy(self._pose)
        self._initial_target = np.copy(self._target)

        # Límites opcionales de distancia al target. None equivale a sin límite.
        self._min_distance = None
        self._max_distance = None

    @property
    def pose(self):
        """Pose actual cámara-a-mundo."""
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = np.array(value, dtype=np.float64)

    @property
    def target(self):
        """Punto alrededor del cual se rota."""
        return self._target

    @target.setter
    def target(self, value):
        self._target = np.array(value, dtype=np.float64)

    def set_state(self, state):
        """Cambia el modo de interacción (rotar, paneo, roll, zoom)."""
        self._state = state

    def resize(self, size):
        """Notifica un cambio de tamaño de ventana para mantener la sensibilidad."""
        self._size = np.array(size, dtype=np.float32)

    def reset(self):
        """Restaura pose y target a los valores iniciales."""
        self._pose   = np.copy(self._initial_pose)
        self._target = np.copy(self._initial_target)

    def set_initial_state(self, pose=None, target=None):
        """Define el estado al que vuelve `reset`.

        Si no se pasan argumentos, el estado actual queda como nuevo punto
        de retorno.
        """
        self._initial_pose   = np.copy(pose   if pose   is not None else self._pose)
        self._initial_target = np.copy(target if target is not None else self._target)

    def set_distance_limits(self, min_distance=None, max_distance=None):
        """Restringe la distancia al target en un rango opcional.

        Útil para evitar que la cámara atraviese el objeto al hacer zoom in
        o se pierda en la lejanía al hacer zoom out.
        """
        self._min_distance = min_distance
        self._max_distance = max_distance

    def get_camera_distance(self):
        """Distancia actual entre la cámara y el target."""
        eye = self._pose[:3, 3]
        return np.linalg.norm(eye - self._target)

    def set_camera_distance(self, distance):
        """Mueve la cámara a una distancia dada del target sin cambiar la dirección."""
        eye = self._pose[:3, 3]
        target_to_eye    = eye - self._target
        current_distance = np.linalg.norm(target_to_eye)

        if current_distance < 1e-10:
            # La cámara coincide con el target; sin dirección clara, usamos
            # el eje Z de la cámara como referencia.
            direction = self._pose[:3, 2]
        else:
            direction = target_to_eye / current_distance

        if self._min_distance is not None:
            distance = max(distance, self._min_distance)
        if self._max_distance is not None:
            distance = min(distance, self._max_distance)

        self._pose[:3, 3] = self._target + direction * distance

    def down(self, mouse_position):
        """Marca el punto donde empieza un arrastre y guarda la pose actual."""
        self._drag_start  = np.array(mouse_position, dtype=np.float32)
        self._drag_pose   = np.copy(self._pose)
        self._drag_target = np.copy(self._target)

    def drag(self, mouse_position):
        """Reescribe la pose según el desplazamiento del mouse desde `down`.

        Cada estado responde de forma distinta al mismo desplazamiento. La
        rotación usa el vector completo, el zoom solo el componente vertical,
        el roll usa la posición del mouse en pantalla.
        """
        if self._drag_start is None:
            return

        mouse_position = np.array(mouse_position, dtype=np.float32)
        mouse_delta    = mouse_position - self._drag_start

        # Normalizamos el desplazamiento por el lado más corto de la ventana
        # (escalado por 0.3 para una sensibilidad cómoda). Así el gesto se
        # siente igual sin importar la resolución.
        size_factor = 0.3 * min(self._size)

        if self._state == Arcball.STATE_ROTATE:
            self._drag_rotate(mouse_delta, size_factor)
        elif self._state == Arcball.STATE_PAN:
            self._drag_pan(mouse_delta, size_factor)
        elif self._state == Arcball.STATE_ROLL:
            self._drag_roll(mouse_position)
        elif self._state == Arcball.STATE_ZOOM:
            self._drag_zoom(mouse_delta[1], size_factor)

    def _drag_rotate(self, mouse_delta, size_factor):
        """Rota la cámara alrededor del target combinando yaw y pitch.

        Yaw (rotación horizontal) se hace alrededor del eje Y mundial, así
        el horizonte se mantiene horizontal aunque la cámara mire hacia
        arriba o hacia abajo. Pitch (rotación vertical) se hace alrededor
        del eje X local de la cámara, para que mover el mouse hacia arriba
        siempre suba la mirada.
        """
        yaw   = -mouse_delta[0] / size_factor
        pitch =  mouse_delta[1] / size_factor

        # Rotar alrededor de un punto que no es el origen requiere tres pasos:
        # trasladar el sistema al origen, rotar, y trasladar de vuelta.
        to_origin = np.eye(4)
        to_origin[:3, 3] = -self._drag_target

        from_origin = np.eye(4)
        from_origin[:3, 3] = self._drag_target

        yaw_around_target = from_origin @ tr.rotationY(yaw) @ to_origin
        pose_after_yaw    = yaw_around_target @ self._drag_pose

        # El eje X local sale de la pose ya rotada por yaw, no de la pose
        # original; así el pitch siempre rota "hacia arriba" desde la
        # perspectiva actual de la cámara.
        camera_x_after_yaw  = pose_after_yaw[:3, 0]
        pitch_rotation      = self._axis_angle_matrix(pitch, camera_x_after_yaw)
        pitch_around_target = from_origin @ pitch_rotation @ to_origin

        self._pose = pitch_around_target @ pose_after_yaw

    def _drag_pan(self, mouse_delta, size_factor):
        """Traslada cámara y target en el plano paralelo a la pantalla.

        El paneo no es una rotación: lo que se mueve es el target junto con
        la cámara, así el ángulo y la distancia se mantienen y solo cambia
        el encuadre.
        """
        # Convertimos el desplazamiento en píxeles a desplazamiento en
        # coordenadas de mundo. El factor 1/3 mantiene el paneo más lento
        # que la rotación; `_scale` deja al usuario calibrar según el tamaño
        # del modelo.
        pan_x = -mouse_delta[0] / (3.0 * size_factor) * self._scale
        pan_y = -mouse_delta[1] / (3.0 * size_factor) * self._scale

        # Las dos primeras columnas de la pose son los ejes "derecha" y
        # "arriba" de la cámara expresados en coordenadas de mundo.
        camera_right = self._drag_pose[:3, 0]
        camera_up    = self._drag_pose[:3, 1]
        translation  = pan_x * camera_right + pan_y * camera_up

        self._target      = self._drag_target      + translation
        self._pose[:3, 3] = self._drag_pose[:3, 3] + translation

    def _drag_roll(self, mouse_position):
        """Rota la cámara alrededor del eje que va del target a la cámara.

        El ángulo se calcula entre el vector inicial y el vector actual desde
        el centro de la pantalla al cursor: si el usuario describe un arco
        con el mouse, la cámara gira el mismo ángulo sobre su eje de vista.
        """
        screen_center  = self._size / 2.0
        start_vector   = self._drag_start - screen_center
        current_vector = mouse_position   - screen_center

        start_length   = np.linalg.norm(start_vector)
        current_length = np.linalg.norm(current_vector)

        # Si el mouse está cerca del centro, el ángulo es ambiguo y salimos
        # sin tocar la pose.
        if start_length < 1e-6 or current_length < 1e-6:
            return

        start_vector   = start_vector   / start_length
        current_vector = current_vector / current_length

        # arctan2 da el ángulo absoluto de cada vector respecto al eje X de
        # la pantalla; la diferencia es el arco recorrido por el cursor.
        roll_angle = (
            np.arctan2(current_vector[1], current_vector[0])
            - np.arctan2(start_vector[1], start_vector[0])
        )

        # Eje de rotación: el que une el target con la cámara. Así el roll
        # gira la imagen sin cambiar la distancia ni la dirección de mirada.
        eye              = self._drag_pose[:3, 3]
        roll_axis        = eye - self._drag_target
        roll_axis_length = np.linalg.norm(roll_axis)

        if roll_axis_length < 1e-6:
            roll_axis = self._drag_pose[:3, 2]
        else:
            roll_axis = roll_axis / roll_axis_length

        to_origin = np.eye(4)
        to_origin[:3, 3] = -self._drag_target

        from_origin = np.eye(4)
        from_origin[:3, 3] = self._drag_target

        roll_rotation = self._axis_angle_matrix(roll_angle, roll_axis)
        roll_matrix   = from_origin @ roll_rotation @ to_origin

        self._pose = roll_matrix @ self._drag_pose

    def _drag_zoom(self, vertical_delta, size_factor):
        """Zoom mediante arrastre vertical del mouse.

        Cada arrastre multiplica la distancia al target por un factor que
        depende del desplazamiento. Usamos exp() para que arrastrar arriba
        y abajo sean operaciones inversas: lo que un gesto multiplica, el
        gesto opuesto lo divide.
        """
        factor = np.exp(vertical_delta / size_factor)
        self._scale_distance(factor)

    def scroll(self, clicks):
        """Zoom con la rueda del mouse.

        Cada click multiplica la distancia al target por un factor constante.
        Clicks positivos acercan la cámara; negativos la alejan.
        """
        if clicks == 0:
            return
        # 0.9^clicks: cada paso modifica la distancia un 10%. La base no es
        # mágica, solo controla la velocidad del zoom; bájala para zooms más
        # finos, súbela para zooms más bruscos.
        factor = 0.9 ** clicks
        self._scale_distance(factor)

    def _scale_distance(self, factor):
        """Multiplica la distancia entre cámara y target por un factor.

        La dirección de mirada no cambia, solo la distancia. Esto hace que
        el zoom se sienta como acercarse/alejarse en línea recta hacia el
        target. Si se definieron límites con `set_distance_limits`, se
        respetan acá.
        """
        eye              = self._pose[:3, 3]
        target_to_eye    = eye - self._target
        current_distance = np.linalg.norm(target_to_eye)

        if current_distance < 1e-10:
            # La cámara coincide con el target; usamos el eje Z como dirección.
            direction = self._pose[:3, 2]
        else:
            direction = target_to_eye / current_distance

        new_distance = current_distance * factor

        if self._min_distance is not None:
            new_distance = max(new_distance, self._min_distance)
        if self._max_distance is not None:
            new_distance = min(new_distance, self._max_distance)

        self._pose[:3, 3] = self._target + direction * new_distance

    def rotate(self, angle, axis=None):
        """Rota la cámara un ángulo dado alrededor de un eje que pasa por el target.

        `axis` puede ser 'x', 'y', 'z' (ejes mundiales) o un vector 3D para
        un eje arbitrario. Si es None, se rota alrededor de Y (caso típico
        de "girar el modelo en su pedestal").
        """
        if axis is None or axis == 'y':
            rotation_matrix = tr.rotationY(angle)
        elif axis == 'x':
            rotation_matrix = tr.rotationX(angle)
        elif axis == 'z':
            rotation_matrix = tr.rotationZ(angle)
        else:
            rotation_matrix = self._axis_angle_matrix(angle, axis)

        # Mismo patrón que en `_drag_rotate`: trasladar al origen, rotar,
        # trasladar de vuelta.
        to_origin = np.eye(4)
        to_origin[:3, 3] = -self._target

        from_origin = np.eye(4)
        from_origin[:3, 3] = self._target

        rotation_around_target = from_origin @ rotation_matrix @ to_origin
        self._pose = rotation_around_target @ self._pose

    def _axis_angle_matrix(self, angle, axis):
        """Matriz de rotación 4x4 alrededor de un eje arbitrario.

        Se conoce como fórmula de Rodrigues. Expresa la rotación como suma
        de tres términos:
            R = cos*I + sin*[axis] + (1 - cos)*axis*axis^T,
        donde [axis] es la matriz antisimétrica del producto cruz y
        axis*axis^T es el producto exterior. La forma desplegada que
        escribimos abajo evita construir esas matrices intermedias.
        """
        axis = np.array(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        cos_angle     = np.cos(angle)
        sin_angle     = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        axis_x, axis_y, axis_z = axis

        rotation_3x3 = np.array([
            [one_minus_cos*axis_x*axis_x + cos_angle,
             one_minus_cos*axis_x*axis_y - sin_angle*axis_z,
             one_minus_cos*axis_x*axis_z + sin_angle*axis_y],
            [one_minus_cos*axis_x*axis_y + sin_angle*axis_z,
             one_minus_cos*axis_y*axis_y + cos_angle,
             one_minus_cos*axis_y*axis_z - sin_angle*axis_x],
            [one_minus_cos*axis_x*axis_z - sin_angle*axis_y,
             one_minus_cos*axis_y*axis_z + sin_angle*axis_x,
             one_minus_cos*axis_z*axis_z + cos_angle]
        ])

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation_3x3
        return rotation_matrix

    def stabilize_rotation(self):
        """Reortogonaliza la submatriz de rotación con Gram-Schmidt.

        Con muchas multiplicaciones encadenadas, los errores de punto flotante
        hacen que las columnas de la rotación dejen de ser ortogonales. Eso
        introduce distorsiones (corte, escala) que se acumulan. Este método
        las corrige preservando la pose.
        """
        rotation_3x3 = self._pose[:3, :3]

        x_axis = rotation_3x3[:, 0]
        y_axis = rotation_3x3[:, 1]

        # Mantenemos X como referencia y le sacamos a Y la componente paralela
        # a X (proyección). Lo que queda es ortogonal a X.
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_axis = y_axis / np.linalg.norm(y_axis)

        # El producto cruz garantiza que Z sea ortogonal a X y a Y.
        z_axis = np.cross(x_axis, y_axis)

        self._pose[:3, 0] = x_axis
        self._pose[:3, 1] = y_axis
        self._pose[:3, 2] = z_axis
