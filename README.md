# Computación Gráfica y Modelamiento para ingenieros e ingenieras (CC3501)

Es necesario el administrador de entornos [`uv`](https://docs.astral.sh/uv/getting-started/installation/). 

El archivo `caja_de_juguetes.py` sirve como puerta de entrada a los distintos ejemplos del curso. Se ejecuta así:

`uv run python caja_de_juguetes.py nombre_ejemplo parámetros opciones`

Algunos ejemplos no tienen parámetros ni opciones:

`uv run python caja_de_juguetes hello_world`

Otros requieren parámetros:

`uv run python caja_de_juguetes image_texture assets/dice.jpg`

Y las opciones no son requisito, puesto que cada programa tiene valores por omisión, pero, en caso de haberlas, se especifican así (en este caso, las opciones son `x0 = 10` e `y0 = - 1`):

`uv run python caja_de_juguetes sr_jengibre --x0 10 --y0 -1`

Para ver la lista de ejemplos, puedes ejecutar la caja de juguetes sin incluir un nombre de ejemplo:

`uv run python caja_de_juguetes.py`

Esto debería imprimir en tu pantalla una salida similar a esta:

```
$ uv run python caja_de_juguetes.py
Usage: caja_de_juguetes.py [OPTIONS] COMMAND [ARGS]...


Options:
 --help  Show this message and exit.


Commands:
 arcball_example     Visor interactivo de modelos 3D
 boids_abm           Simulador de vuelo de pajaritos usando Agent-Based
                     Modeling
 cloth_pymunk        Simulación de tela con pymunk
 cloth_verlet        Simulación de tela usando una implementación ingenua de
                     integración de Verlet
 color_wheel         Ejemplo de espacios de color
 compositions        Ejemplo de composición de transformaciones
 dino_runner         Ejemplo de detección de colisiones en 2D
 falling_boxes       Ejemplo de uso de Pymunk
 hello_opengl        ¡Hola, OpenGL!
 hello_world         ¡Hola, mundo!
 image_pixel         Visor de imágenes
 image_texture       Visor de imágenes (versión textura)
 particles           Partículas simples
 projection_example  Ejemplo de proyección
 raytracing_cpu      Prueba de concepto de RT en la CPU
 solar_system        Sistema solar con grafos de escena
 sr_jengibre         Señor Jengibre
 transformed_bunny   Ejemplo de transformaciones con el conejo de Stanford
```
