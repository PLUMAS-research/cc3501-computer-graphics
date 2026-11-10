"""Convierte un modelo con esqueleto a .gltf + .bin, el formato que lee grafica/gltf.py.

Corre dentro de Blender (probado con Blender 5.0.1), no con uv:

    blender --background --factory-startup --python tools/convertir_a_gltf.py -- \\
        MODELO [ANIMACION ...] -o SALIDA.gltf

MODELO puede ser .glb, .gltf, .fbx o .blend. La salida queda en SALIDA.gltf, con
SALIDA.bin y las texturas al lado, y trae todas las animaciones del modelo.

Cada ANIMACION es un .fbx, .glb o .gltf con un esqueleto de huesos con los mismos
nombres, como en los paquetes que traen un .fbx por animacion. De cada uno se toma
la animacion mas larga, que queda con el nombre del archivo; las poses de un
cuadro que agregan algunos exportadores se descartan. La animacion se copia hueso
por hueso en coordenadas de mundo y se hornea en el esqueleto del modelo, porque
el esqueleto del archivo de animacion puede tener otra pose de reposo: en Kenney,
el modelo esta con los brazos en cruz y las animaciones, con los brazos abajo.

Ejemplos:

    # un .glb (Poly Pizza, Sketchfab)
    blender --background --factory-startup --python tools/convertir_a_gltf.py -- \\
        descargas/animated-human.glb -o assets/animated-human/animated-human.gltf

    # un personaje de Kenney: el modelo y tres animaciones, cada uno en su .fbx
    blender --background --factory-startup --python tools/convertir_a_gltf.py -- \\
        Model/characterMedium.fbx Animations/idle.fbx Animations/run.fbx \\
        Animations/jump.fbx -o assets/kenney/characterMedium.gltf

    # el zorzal con otra animacion, en un archivo con los mismos huesos
    blender --background --factory-startup --python tools/convertir_a_gltf.py -- \\
        assets/zorzal-aleteo.gltf descargas/zorzal-salto.fbx -o assets/zorzal/zorzal.gltf

y despues:

    uv run python caja_de_juguetes.py skinning assets/animated-human/animated-human.gltf

Blender guarda las animaciones en cuadros y las exporta muestreando un cuadro
a la vez, a --fps cuadros por segundo (30 por omision; un .blend usa la tasa que
trae). Con 30, los keyframes del zorzal, que estan cada 1/30 de segundo, salen
en los mismos instantes.

Las animaciones que ya traia el modelo quedan en el esqueleto que movian. El
zorzal trae dos esqueletos y el aleteo mueve el segundo, asi que despues de
convertirlo sigue igual que antes: skinning toma, para cada joint de la malla, la
pose global del nodo de igual nombre que mueve la animacion. Las animaciones
nuevas, en cambio, quedan horneadas en el esqueleto de la malla. Las dos maneras
dan la misma pose: el aleteo original y Fly, exportada desde Unreal, coinciden.
Blender ademas aplica la escala de 0.01 que el zorzal trae en la raiz, asi que el
modelo convertido queda 100 veces mas chico; skinning ajusta la camara a ese
tamaño.

Un .blend se abre sin ejecutar los scripts que traiga. Si un .fbx no encuentra sus
texturas, la malla sale blanca: hay que abrirlo en Blender, asignar la imagen al
Base Color del material, guardarlo como .blend y convertir ese archivo.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def importar(ruta):
    """Importa un archivo a la escena y devuelve los objetos que agrego."""
    antes = set(bpy.data.objects)
    sufijo = ruta.suffix.lower()
    if sufijo in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(ruta))
    elif sufijo == ".fbx":
        bpy.ops.wm.fbx_import(filepath=str(ruta))
    else:
        raise SystemExit(f"no se como importar {ruta}")
    return set(bpy.data.objects) - antes


def esqueleto_de_la_malla():
    """La armadura que deforma alguna malla, a traves de su modificador Armature."""
    for objeto in bpy.context.scene.objects:
        if objeto.type != "MESH":
            continue
        for modificador in objeto.modifiers:
            if modificador.type == "ARMATURE" and modificador.object is not None:
                return modificador.object
    raise SystemExit("ninguna malla del modelo tiene esqueleto")


def duracion(accion):
    inicio, fin = accion.frame_range
    return fin - inicio


def hornear(esqueleto, origen, accion, nombre):
    """Copia la animacion de `origen` en `esqueleto` y devuelve la accion horneada.

    Cada hueso del esqueleto copia la transformacion en coordenadas de mundo del
    hueso de igual nombre en `origen` (restriccion Copy Transforms), y nla.bake
    graba la pose resultante en cada cuadro. Asi la pose de reposo de `origen` no
    importa, solo la pose que alcanza en cada instante.
    """
    origen.animation_data.action = accion
    for hueso in esqueleto.pose.bones:
        if hueso.name in origen.pose.bones:
            restriccion = hueso.constraints.new("COPY_TRANSFORMS")
            restriccion.target = origen
            restriccion.subtarget = hueso.name
    inicio, fin = accion.frame_range
    bpy.context.view_layer.objects.active = esqueleto
    esqueleto.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(frame_start=math.floor(inicio), frame_end=math.ceil(fin), only_selected=True,
                     visual_keying=True, clear_constraints=True, use_current_action=False,
                     bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    horneada = esqueleto.animation_data.action
    esqueleto.animation_data.action = None
    horneada.name = nombre
    return horneada


def animacion_de(ruta, esqueleto):
    """La animacion mas larga de otro archivo, horneada en el esqueleto del modelo."""
    antes = set(bpy.data.actions)
    nuevos = importar(ruta)
    origenes = [objeto for objeto in nuevos if objeto.type == "ARMATURE"]
    nuevas = [accion for accion in bpy.data.actions if accion not in antes]
    if not origenes or not nuevas:
        raise SystemExit(f"{ruta} no trae un esqueleto con animaciones")
    accion = max(nuevas, key=duracion)
    # el esqueleto que mueve esa animacion; el zorzal, por ejemplo, trae dos
    origen = duenos().get(accion)
    if origen not in origenes:
        origen = origenes[0]
    huesos = {hueso.name for hueso in esqueleto.data.bones}
    comunes = len(huesos & {hueso.name for hueso in origen.data.bones})
    print(f"[convertir] {ruta.name}: '{accion.name}' ({duracion(accion):.0f} cuadros) "
          f"de {[otra.name for otra in nuevas]}; {comunes} de {len(huesos)} huesos con el mismo nombre")
    horneada = hornear(esqueleto, origen, accion, ruta.stem)
    for objeto in nuevos:
        bpy.data.objects.remove(objeto)
    for otra in nuevas:
        bpy.data.actions.remove(otra)
    return horneada


def duenos():
    """El objeto que usa cada animacion, como accion activa o en una pista NLA."""
    dueno = {}
    for objeto in bpy.context.scene.objects:
        datos = objeto.animation_data
        if datos is None:
            continue
        for pista in datos.nla_tracks:
            for tira in pista.strips:
                if tira.action is not None:
                    dueno.setdefault(tira.action, objeto)
        if datos.action is not None:
            dueno.setdefault(datos.action, objeto)
    return dueno


def asignar(esqueleto, acciones, dueno):
    """Pone cada animacion en una pista NLA propia.

    El exportador de glTF convierte cada pista en una animacion del archivo, y
    junta las pistas de igual nombre de distintos objetos. Las animaciones
    sueltas, que no usa ningun objeto, no se exportan. Cada animacion queda en el
    objeto que la usaba; las sueltas y las de otros archivos, en el esqueleto de
    la malla. Moverla a otro esqueleto la estropea si las poses de reposo de los
    dos no coinciden: el zorzal trae dos esqueletos y el aleteo mueve el segundo.
    """
    for objeto in bpy.context.scene.objects:
        if objeto.animation_data is not None:
            objeto.animation_data.action = None
            for pista in list(objeto.animation_data.nla_tracks):
                objeto.animation_data.nla_tracks.remove(pista)
    for accion in acciones:
        objeto = dueno.get(accion, esqueleto)
        datos = objeto.animation_data or objeto.animation_data_create()
        try:
            pista = datos.nla_tracks.new()
            # el importador de glTF de Blender antepone el esqueleto: "Armature|Run"
            pista.name = accion.name.split("|")[-1]
            pista.strips.new(accion.name, int(accion.frame_range[0]), accion)
        except RuntimeError as error:
            datos.nla_tracks.remove(pista)
            print(f"[convertir] se omite '{accion.name}': {error}")


def main():
    argumentos = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="convertir_a_gltf.py",
                                     description="Convierte un modelo con esqueleto a .gltf + .bin")
    parser.add_argument("modelo", type=Path)
    parser.add_argument("animaciones", type=Path, nargs="*")
    parser.add_argument("-o", "--salida", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argumentos)
    modelo = args.modelo.resolve()
    animaciones = [ruta.resolve() for ruta in args.animaciones]
    salida = args.salida.resolve()

    if modelo.suffix.lower() == ".blend":
        # un .blend ya trae sus animaciones en cuadros, a la tasa del archivo
        bpy.ops.wm.open_mainfile(filepath=str(modelo), load_ui=False, use_scripts=False)
    else:
        # la escena de fabrica trae un cubo, una camara y una luz
        for objeto in list(bpy.data.objects):
            bpy.data.objects.remove(objeto)
        # antes de importar, porque los importadores pasan los tiempos a cuadros
        bpy.context.scene.render.fps = args.fps
        bpy.context.scene.render.fps_base = 1.0
        importar(modelo)

    esqueleto = esqueleto_de_la_malla()
    huesos = {hueso.name for hueso in esqueleto.data.bones}
    acciones = list(bpy.data.actions)
    dueno = duenos()
    print(f"[convertir] {modelo.name}: esqueleto '{esqueleto.name}' con {len(huesos)} huesos, "
          f"animaciones {[accion.name for accion in acciones]}")
    acciones += [animacion_de(ruta, esqueleto) for ruta in animaciones]
    asignar(esqueleto, acciones, dueno)

    salida.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(salida),
        export_format="GLTF_SEPARATE",
        export_yup=True,
        export_skins=True,
        export_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_morph=False,
    )

    documento = json.loads(salida.read_text())
    print(f"[convertir] escrito {salida}")
    print(f"[convertir] skins: {[len(skin['joints']) for skin in documento.get('skins', [])]} huesos")
    for animacion in documento.get("animations", []):
        print(f"[convertir] animacion '{animacion.get('name', '')}': {len(animacion['channels'])} canales")


main()
