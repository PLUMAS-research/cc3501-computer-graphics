#!/usr/bin/env python3

import importlib

import click


@click.group()
def grafica_cli():
    pass


from examples.image_pixel import image_pixel_viewer

grafica_cli.add_command(image_pixel_viewer)

from examples.color import color_wheel

grafica_cli.add_command(color_wheel)

from examples.hello_world import hola_mundo

grafica_cli.add_command(hola_mundo)

from examples.chroma_key import chroma_key

grafica_cli.add_command(chroma_key)

from examples.image_texture import image_viewer

grafica_cli.add_command(image_viewer)

from examples.sr_jengibre import sr_jengibre

grafica_cli.add_command(sr_jengibre)

from examples.sr_jengibre_numpy import gingerbread_numpy

grafica_cli.add_command(gingerbread_numpy)

from examples.particles.app import particulas

grafica_cli.add_command(particulas)

# la carpeta tiene un guion :) así que hay que usar otro método
boids = importlib.import_module("examples.boids-particles")
grafica_cli.add_command(boids.boids_particles)

from examples.arcball import arcball_example

grafica_cli.add_command(arcball_example)

from examples.cloth.app_pymunk import cloth_pymunk

grafica_cli.add_command(cloth_pymunk)

from examples.cloth.app_verlet import cloth_verlet

grafica_cli.add_command(cloth_verlet)

from examples.masa_resorte import masa_resorte

grafica_cli.add_command(masa_resorte)

from examples.difusion_calor import difusion_calor

grafica_cli.add_command(difusion_calor)

boids = importlib.import_module("examples.boids-abm.app")
grafica_cli.add_command(boids.boids_abm)

# TODO: hay que actualizar el código de este ejemplo
from examples.collision_detection import dino_runner

grafica_cli.add_command(dino_runner)

from examples.katamari import katamari

grafica_cli.add_command(katamari)

from examples.ray_triangle import ray_triangle_example

grafica_cli.add_command(ray_triangle_example)

from examples.hello_opengl import hola_opengl

grafica_cli.add_command(hola_opengl)

from examples.shadows import shadow_mapping

grafica_cli.add_command(shadow_mapping)

from examples.terrain import terrain_generation

grafica_cli.add_command(terrain_generation)

from examples.operaciones_malla import operaciones_malla

grafica_cli.add_command(operaciones_malla)

from examples.simplificacion_qem import simplificacion_qem

grafica_cli.add_command(simplificacion_qem)

from examples.curvas_parametricas import curvas_parametricas

grafica_cli.add_command(curvas_parametricas)

from examples.csg_raymarching import csg_raymarching

grafica_cli.add_command(csg_raymarching)

from examples.superficie_bezier import superficie_bezier

grafica_cli.add_command(superficie_bezier)

from examples.metaballs import metaballs

grafica_cli.add_command(metaballs)

from examples.cubo_gelatina import cubo_gelatina

grafica_cli.add_command(cubo_gelatina)

from examples.cubo_resortes import cubo_resortes

grafica_cli.add_command(cubo_resortes)

from examples.projection.app import projection_example

grafica_cli.add_command(projection_example)

from examples.pymunk_boxes.app import falling_boxes

grafica_cli.add_command(falling_boxes)

from examples.pymunk_basico.app import pymunk_basico

grafica_cli.add_command(pymunk_basico)

from examples.pymunk_pinball.app import pymunk_pinball

grafica_cli.add_command(pymunk_pinball)

from examples.raytracing_basico import raytracing_basico

grafica_cli.add_command(raytracing_basico)

from examples.raytracing_cpu.app import raytracing_cpu

grafica_cli.add_command(raytracing_cpu)

from examples.scene_graphs.app import solar_system

grafica_cli.add_command(solar_system)

from examples.pokemon_instancing import pokemon_instancing

grafica_cli.add_command(pokemon_instancing)

from examples.transformation_composition.app import compositions

grafica_cli.add_command(compositions)

from examples.transformations.app import transformed_bunny

grafica_cli.add_command(transformed_bunny)

from examples.spirograph import spirograph

grafica_cli.add_command(spirograph)

from examples.gimbal_lock import gimbal_lock

grafica_cli.add_command(gimbal_lock)

from examples.phong_basico import phong_basico

grafica_cli.add_command(phong_basico)

from examples.disco_bunny.app import disco_bunny

grafica_cli.add_command(disco_bunny)

from examples.cel_bunny import cel_bunny

grafica_cli.add_command(cel_bunny)

from examples.sistema_tierra import sistema_tierra

grafica_cli.add_command(sistema_tierra)

from examples.camera_path import camera_path

grafica_cli.add_command(camera_path)

from examples.camera_frustum import camera_frustum

grafica_cli.add_command(camera_frustum)

from examples.pyvista_orbital import orbital

grafica_cli.add_command(orbital)

from examples.sugecon import suggestive_contours

grafica_cli.add_command(suggestive_contours)

from examples.degradado import degradado

grafica_cli.add_command(degradado)

from examples.bad_tv import bad_tv

grafica_cli.add_command(bad_tv)

from examples.buddhabrot import buddhabrot

grafica_cli.add_command(buddhabrot)

from examples.flappy_redpanda.app import flappy_redpanda

grafica_cli.add_command(flappy_redpanda)

from examples.edo_case_studies import edo_case_studies

grafica_cli.add_command(edo_case_studies)

from examples.animacion_esqueletica import animacion_esqueletica

grafica_cli.add_command(animacion_esqueletica)

from examples.skinning import skinning

grafica_cli.add_command(skinning)

from examples.quadtree import quadtree_demo

grafica_cli.add_command(quadtree_demo)

from examples.rasterizer import rasterizer

grafica_cli.add_command(rasterizer)

from examples.texture_viewer import texture_viewer

grafica_cli.add_command(texture_viewer)

from examples.lapped_hatching import lapped_hatching

grafica_cli.add_command(lapped_hatching)

from examples.perspective_correction import perspective_correction

grafica_cli.add_command(perspective_correction)

from examples.pecera import pecera

grafica_cli.add_command(pecera)

from examples.bosque import bosque

grafica_cli.add_command(bosque)

if __name__ == "__main__":
    grafica_cli()
