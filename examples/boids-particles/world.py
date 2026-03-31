# mesa es la biblioteca de simulacion basada en agentes que utilizaremos
import mesa
import numpy as np

from .boid import Boid
from scipy import spatial
from grafica.quadtree import QuadTree, Rectangle


# la clase World contiene el mundo simulado.
class World(mesa.Model):

    def __init__(
        self,
        population=100,
        width=100,
        height=100,
        speed=1,
        vision=10,
        distance=2,
        cohere_factor=0.025,
        separation_factor=0.25,
        match_factor=0.04,
        spatial_method="quadtree",
    ):
        super().__init__(seed=666)
        self.population = population
        self.vision = vision
        self.speed = speed
        self.distance = distance
        self.space = mesa.space.ContinuousSpace(width, height, True)
        self.factors = dict(
            cohere_factor=cohere_factor,
            separation_factor=separation_factor,
            match_factor=match_factor,
        )
        self.spatial_method = spatial_method
        self.qt = None
        self.make_agents()
        self.running = True

    def make_agents(self):
        self.id_to_agent = {}
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            velocity = np.random.random(2) * 2 - 1
            boid = Boid(
                self,
                pos,
                self.speed,
                velocity,
                self.vision,
                self.distance,
                **self.factors,
            )
            self.space.place_agent(boid, pos)
            self.id_to_agent[i] = boid

    def step(self):
        if self.spatial_method == "quadtree":
            boundary = Rectangle(
                self.space.x_max / 2,
                self.space.y_max / 2,
                self.space.x_max / 2,
                self.space.y_max / 2,
            )
            self.qt = QuadTree(boundary, capacity=4, max_depth=8)
            for idx, boid in self.id_to_agent.items():
                self.qt.insert(boid.pos[0], boid.pos[1], idx)
        else:
            self.tree = spatial.KDTree(
                [boid.pos for boid in self.id_to_agent.values()]
            )
            self.qt = None

        self.agents.shuffle_do("step")

    def iter_agents(self):
        yield from self.space._agent_to_index.keys()

    def query_area(self, pos, radius):
        if self.spatial_method == "quadtree":
            indices = self.qt.query_circle_points(pos[0], pos[1], radius)
            return [self.id_to_agent[idx] for idx in indices]
        else:
            result_ids = self.tree.query_ball_point(pos, radius)
            return [self.id_to_agent[idx] for idx in result_ids]

    def compute_polarization(self):
        """Polarizacion: grado de alineamiento global [0, 1]."""
        velocities = np.array([b.velocity for b in self.id_to_agent.values()])
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        unit_vels = velocities / norms
        return float(np.linalg.norm(unit_vels.mean(axis=0)))

    def compute_dispersion(self):
        """Dispersion: desviacion estandar de posiciones."""
        positions = np.array([b.pos for b in self.id_to_agent.values()])
        centroid = positions.mean(axis=0)
        dists = np.linalg.norm(positions - centroid, axis=1)
        return float(dists.std())
