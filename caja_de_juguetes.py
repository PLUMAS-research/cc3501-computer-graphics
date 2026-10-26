#!/usr/bin/env python3

import importlib

import click

# título de cada sección de la ayuda -> comandos que muestra, en orden
UNIDADES = {}


class CajaDeJuguetes(click.Group):
    """Grupo de comandos que lista los ejemplos agrupados por unidad del curso."""

    def format_commands(self, ctx, formatter):
        ancho = max(len(nombre) for nombre in self.commands)
        limite = formatter.width - 6 - ancho
        for unidad, comandos in UNIDADES.items():
            filas = [
                (comando.name.ljust(ancho), comando.get_short_help_str(limite))
                for comando in comandos
            ]
            with formatter.section(unidad):
                formatter.write_dl(filas)


@click.group(cls=CajaDeJuguetes)
def grafica_cli():
    pass


def registrar(unidad, *comandos):
    for comando in comandos:
        grafica_cli.add_command(comando)
    UNIDADES.setdefault(unidad, []).extend(comandos)


from examples.chroma_key import chroma_key
from examples.degradado import degradado

registrar("Unidad 2. Imágenes, píxeles y colores", degradado, chroma_key)

from examples.bad_tv import bad_tv
from examples.color import color_wheel
from examples.hello_world import hola_mundo

registrar("Unidad 3. GPU y rendering con OpenGL", hola_mundo, color_wheel, bad_tv)

from examples.buddhabrot import buddhabrot
from examples.image_pixel import image_pixel_viewer
from examples.image_texture import image_viewer
from examples.lorenz import lorenz
from examples.sr_jengibre import sr_jengibre
from examples.sr_jengibre_numpy import gingerbread_numpy

registrar(
    "Unidad 4. Texturas y atractores",
    image_pixel_viewer,
    image_viewer,
    gingerbread_numpy,
    sr_jengibre,
    lorenz,
    buddhabrot,
)

from examples.edo_case_studies import edo_case_studies
from examples.flappy_redpanda.app import flappy_redpanda
from examples.particles.app import particulas

registrar("Unidad 5. EDO y partículas", edo_case_studies, flappy_redpanda, particulas)

# las carpetas tienen un guion :) así que hay que usar otro método
boids_particles = importlib.import_module("examples.boids-particles").boids_particles
boids_abm = importlib.import_module("examples.boids-abm.app").boids_abm
from examples.quadtree import quadtree_demo

registrar(
    "Unidad 6. Simulación basada en agentes y estructuras espaciales",
    boids_particles,
    boids_abm,
    quadtree_demo,
)

from examples.hello_opengl import hola_opengl
from examples.rasterizer import rasterizer

registrar("Unidad 7. Objetos y tuberías", hola_opengl, rasterizer)

from examples.gimbal_lock import gimbal_lock
from examples.spirograph import spirograph
from examples.transformation_composition.app import compositions
from examples.transformations.app import transformed_bunny

registrar(
    "Unidad 8. Transformaciones",
    transformed_bunny,
    compositions,
    spirograph,
    gimbal_lock,
)

from examples.camera_frustum import camera_frustum
from examples.projection.app import projection_example

registrar("Unidad 9. Vistas y proyecciones", projection_example, camera_frustum)

from examples.perspective_correction import perspective_correction
from examples.sistema_tierra import sistema_tierra
from examples.texture_viewer import texture_viewer

registrar(
    "Unidad 10. Mapeo de texturas",
    texture_viewer,
    perspective_correction,
    sistema_tierra,
)

from examples.bosque import bosque
from examples.pecera import pecera

registrar("Unidad 11. Profundidad y transparencia", pecera, bosque)

from examples.carrusel import carrusel
from examples.pokemon_instancing import pokemon_instancing
from examples.scene_graphs.app import solar_system
from examples.webcam_esqueleto import webcam_esqueleto

registrar(
    "Unidad 12. Grafos de escena",
    solar_system,
    pokemon_instancing,
    carrusel,
    webcam_esqueleto,
)

from examples.cel_bunny import cel_bunny
from examples.disco_bunny.app import disco_bunny
from examples.phong_basico import phong_basico
from examples.ssao import ssao

registrar(
    "Unidad 13. Iluminación y materiales",
    phong_basico,
    disco_bunny,
    cel_bunny,
    ssao,
)

from examples.lapped_hatching import lapped_hatching
from examples.lsystem import lsystem
from examples.operaciones_malla import operaciones_malla
from examples.simplificacion_qem import simplificacion_qem

registrar(
    "Unidad 14. Mallas geométricas",
    operaciones_malla,
    simplificacion_qem,
    lsystem,
    lapped_hatching,
)

from examples.pymunk_basico.app import pymunk_basico
from examples.pymunk_boxes.app import falling_boxes
from examples.pymunk_pinball.app import pymunk_pinball

registrar(
    "Unidad 15. Física de cuerpos rígidos",
    pymunk_basico,
    falling_boxes,
    pymunk_pinball,
)

from examples.cloth.app_pymunk import cloth_pymunk
from examples.cloth.app_verlet import cloth_verlet
from examples.cubo_resortes import cubo_resortes
from examples.difusion_calor import difusion_calor
from examples.fem_viga import fem_viga
from examples.masa_resorte import masa_resorte

registrar(
    "Unidad 16. Deformación basada en física",
    masa_resorte,
    cloth_verlet,
    cloth_pymunk,
    cubo_resortes,
    difusion_calor,
    fem_viga,
)

from examples.camera_path import camera_path
from examples.csg_raymarching import csg_raymarching
from examples.cubo_gelatina import cubo_gelatina
from examples.curvas_parametricas import curvas_parametricas
from examples.metaballs import metaballs
from examples.superficie_bezier import superficie_bezier
from examples.terrain import terrain_generation

registrar(
    "Unidad 17. Sólidos, curvas y superficies paramétricas",
    curvas_parametricas,
    camera_path,
    superficie_bezier,
    terrain_generation,
    csg_raymarching,
    metaballs,
    cubo_gelatina,
)

# TODO: hay que actualizar el código de dino_runner
from examples.collision_detection import dino_runner
from examples.katamari import katamari
from examples.ray_triangle import ray_triangle_example

registrar("Unidad 18. Colisiones", dino_runner, katamari, ray_triangle_example)

from examples.raytracing_2d import raytracing_2d
from examples.raytracing_basico import raytracing_basico
from examples.raytracing_bsp import raytracing_bsp
from examples.raytracing_cpu.app import raytracing_cpu

registrar("Unidad 19. Trazado de rayos", raytracing_2d, raytracing_basico, raytracing_bsp, raytracing_cpu)

from examples.pyvista_orbital import orbital
from examples.raycasting_volumen import raycasting_volumen

registrar("Unidad 20. Rendering de volumen", orbital, raycasting_volumen)

from examples.animacion_esqueletica import animacion_esqueletica
from examples.esqueleto_lbs import esqueleto_lbs
from examples.skinning import skinning

registrar(
    "Unidad 21. Animación y esqueletos",
    esqueleto_lbs,
    animacion_esqueletica,
    skinning,
)

from examples.arcball import arcball_example
from examples.cascaded_shadows import cascaded_shadows
from examples.shadows import shadow_mapping
from examples.sugecon import suggestive_contours

registrar(
    "Casos de estudio",
    arcball_example,
    shadow_mapping,
    cascaded_shadows,
    suggestive_contours,
)

if __name__ == "__main__":
    grafica_cli()
