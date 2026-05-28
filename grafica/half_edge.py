"""Estructura half-edge (semi-arista) para mallas de triangulos.

Es la estructura de datos central de la unidad de mallas. Separa la geometria
(posiciones de los vertices) de la topologia (como se conectan) y permite
recorrer vecindades y modificar la malla en tiempo O(1) por operacion local,
no en O(n) como una lista de triangulos.

Cada arista se parte en dos semi-aristas orientadas. Una semi-arista `h`
guarda cuatro referencias:

- `he_to[h]`:   el vertice **destino** de la semi-arista.
- `he_next[h]`: la siguiente semi-arista dentro de la misma cara (sentido CCW).
- `he_twin[h]`: la semi-arista opuesta (misma arista, sentido contrario), o
                -1 si la arista esta en el borde de la malla.
- `he_face[h]`: la cara a la izquierda de la semi-arista.

El vertice **origen** no se almacena: en un triangulo es el destino de la
semi-arista anterior, asi que `origin(h) = he_to[next(next(h))]`. Mantener un
campo menos evita que se desincronice al editar.

Sobre esta estructura se implementan las tres operaciones elementales de la
unidad: `flip` (solo reasigna referencias, no crea ni borra nada), `split`
(inserta un vertice en una arista) y `collapse` (funde una arista en un punto).
La simplificacion QEM se construye encima de `collapse`.

La estructura asume mallas de triangulos. Soporta bordes (aristas con una sola
cara) mediante `he_twin == -1`. El `flip` y el `collapse` se restringen a
configuraciones interiores, donde las vecindades estan completas.
"""

import numpy as np

BORDE = -1  # marca de semi-arista sin gemela: la arista esta en el borde.


class HalfEdgeMesh:
    """Malla de triangulos representada con semi-aristas.

    Los elementos borrados por `collapse` no se reindexan en el acto: se marcan
    con las banderas `v_alive`, `f_alive`, `he_alive` y se compactan al exportar
    con `to_arrays()`. Asi cada operacion toca solo su vecindad local.
    """

    def __init__(self):
        self.positions = []   # lista de np.array de 3 floats, indexada por vertice.
        self.he_to = []       # vertice destino de cada semi-arista.
        self.he_next = []     # siguiente semi-arista en la cara.
        self.he_twin = []     # semi-arista opuesta, o BORDE.
        self.he_face = []     # cara de la semi-arista.
        self.v_he = []        # una semi-arista que sale de cada vertice.
        self.f_he = []        # una semi-arista de cada cara.
        self.v_alive = []
        self.f_alive = []
        self.he_alive = []

    # ------------------------------------------------------------------
    # Construccion
    # ------------------------------------------------------------------
    @classmethod
    def from_faces(cls, positions, faces):
        """Construye la malla desde la representacion face-vertex (OBJ/OFF).

        `positions` es un arreglo (V, D) con D = 2 o 3. `faces` es un arreglo
        (F, 3) de indices. El emparejamiento de gemelas usa un diccionario que
        mapea cada arista dirigida (origen, destino) a su semi-arista; la
        gemela es la semi-arista de la arista opuesta (destino, origen).
        """
        mesh = cls()
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[1] == 2:
            # llevamos todo a 3D con z = 0 para una sola ruta de codigo.
            z = np.zeros((positions.shape[0], 1))
            positions = np.hstack([positions, z])
        mesh.positions = [positions[i].copy() for i in range(len(positions))]
        mesh.v_he = [BORDE] * len(positions)
        mesh.v_alive = [True] * len(positions)

        edge_to_halfedge = {}  # (origen, destino) -> semi-arista.
        for face_index, (a, b, c) in enumerate(np.asarray(faces, dtype=np.int64)):
            a, b, c = int(a), int(b), int(c)
            h0 = len(mesh.he_to)
            h1, h2 = h0 + 1, h0 + 2
            mesh.he_to += [b, c, a]
            mesh.he_next += [h1, h2, h0]
            mesh.he_twin += [BORDE, BORDE, BORDE]
            mesh.he_face += [face_index, face_index, face_index]
            mesh.he_alive += [True, True, True]
            mesh.f_he.append(h0)
            mesh.f_alive.append(True)
            for halfedge, (origin, dest) in zip((h0, h1, h2), ((a, b), (b, c), (c, a))):
                edge_to_halfedge[(origin, dest)] = halfedge
                mesh.v_he[origin] = halfedge

        for (origin, dest), halfedge in edge_to_halfedge.items():
            twin = edge_to_halfedge.get((dest, origin), BORDE)
            mesh.he_twin[halfedge] = twin

        return mesh

    # ------------------------------------------------------------------
    # Navegacion (todo asume caras triangulares)
    # ------------------------------------------------------------------
    def head(self, h):
        """Vertice destino de la semi-arista."""
        return self.he_to[h]

    def tail(self, h):
        """Vertice origen: destino de la anterior en el triangulo."""
        return self.he_to[self.he_next[self.he_next[h]]]

    def is_boundary_edge(self, h):
        return self.he_twin[h] == BORDE

    def outgoing_ring(self, v):
        """Semi-aristas que salen de v, girando alrededor del vertice.

        Devuelve `(halfedges, cerrado)`. `cerrado` es False si el giro choca
        con un borde antes de cerrar el anillo (vertice de borde). El giro usa
        `next(twin(h))`: la gemela de una saliente llega a v, y su siguiente
        vuelve a salir de v.
        """
        start = self.v_he[v]
        halfedges = []
        h = start
        while True:
            halfedges.append(h)
            twin = self.he_twin[h]
            if twin == BORDE:
                return halfedges, False
            h = self.he_next[twin]
            if h == start:
                return halfedges, True

    def is_interior_vertex(self, v):
        return self.outgoing_ring(v)[1]

    def neighbors(self, v):
        """Conjunto de vertices vecinos de v (su vecindad de anillo)."""
        halfedges, _ = self.outgoing_ring(v)
        return {self.he_to[h] for h in halfedges}

    # ------------------------------------------------------------------
    # Conteos y validacion
    # ------------------------------------------------------------------
    def n_vertices(self):
        return sum(self.v_alive)

    def n_faces(self):
        return sum(self.f_alive)

    def n_edges(self):
        interior = sum(
            1 for h in range(len(self.he_to))
            if self.he_alive[h] and self.he_twin[h] != BORDE
        )
        borde = sum(
            1 for h in range(len(self.he_to))
            if self.he_alive[h] and self.he_twin[h] == BORDE
        )
        return interior // 2 + borde

    def euler_characteristic(self):
        return self.n_vertices() - self.n_edges() + self.n_faces()

    def is_manifold(self):
        """Chequeo basico: gemelas simetricas y anillos de vertice cerrados o de borde."""
        for h in range(len(self.he_to)):
            if not self.he_alive[h]:
                continue
            twin = self.he_twin[h]
            if twin != BORDE and self.he_twin[twin] != h:
                return False
        return True

    # ------------------------------------------------------------------
    # Exportar a face-vertex para graficar
    # ------------------------------------------------------------------
    def to_arrays(self):
        """Compacta los elementos vivos a (positions (V,3), faces (F,3))."""
        remap = {}
        positions = []
        for v in range(len(self.positions)):
            if self.v_alive[v]:
                remap[v] = len(positions)
                positions.append(self.positions[v])
        faces = []
        for f in range(len(self.f_he)):
            if not self.f_alive[f]:
                continue
            h = self.f_he[f]
            a = remap[self.tail(h)]
            b = remap[self.he_to[h]]
            c = remap[self.he_to[self.he_next[h]]]
            faces.append((a, b, c))
        return np.array(positions, dtype=np.float64), np.array(faces, dtype=np.int64)

    # ------------------------------------------------------------------
    # Operacion elemental: flip
    # ------------------------------------------------------------------
    def flip(self, h):
        """Da vuelta la diagonal del cuadrilatero formado por las dos caras de h.

        Solo reasigna referencias: no crea ni borra vertices, aristas ni caras.
        Requiere una arista interior. Devuelve True si la dio vuelta.
        """
        t = self.he_twin[h]
        if t == BORDE:
            return False

        h1 = self.he_next[h]
        h2 = self.he_next[h1]
        t1 = self.he_next[t]
        t2 = self.he_next[t1]

        a = self.tail(h)
        b = self.he_to[h]
        c = self.he_to[h1]   # apice de la cara de h.
        d = self.he_to[t1]   # apice de la cara de t.
        if c == d:
            return False  # degeneraria.

        fa = self.he_face[h]
        fb = self.he_face[t]

        # h pasa a ser c->d; t pasa a ser d->c. Las caras se reagrupan.
        self.he_to[h] = d
        self.he_to[t] = c

        # cara fa = (c, d, b): h (c->d), t2 (d->b), h1 (b->c).
        self.he_next[h] = t2
        self.he_next[t2] = h1
        self.he_next[h1] = h
        # cara fb = (d, c, a): t (d->c), h2 (c->a), t1 (a->d).
        self.he_next[t] = h2
        self.he_next[h2] = t1
        self.he_next[t1] = t

        self.he_face[t2] = fa
        self.he_face[h2] = fb

        self.f_he[fa] = h
        self.f_he[fb] = t

        # los vertices a y b dejan de tener a esta arista como saliente.
        self.v_he[a] = t1
        self.v_he[b] = h1
        self.v_he[c] = h2
        self.v_he[d] = t2
        return True

    # ------------------------------------------------------------------
    # Operacion elemental: split
    # ------------------------------------------------------------------
    def _new_halfedge(self, to, twin, face):
        h = len(self.he_to)
        self.he_to.append(to)
        self.he_next.append(h)   # se corrige despues.
        self.he_twin.append(twin)
        self.he_face.append(face)
        self.he_alive.append(True)
        return h

    def split(self, h):
        """Inserta un vertice en el punto medio de la arista de h.

        Cada cara adyacente se parte en dos. Devuelve el indice del nuevo
        vertice. Funciona tanto en aristas interiores como de borde.
        """
        a = self.tail(h)
        b = self.he_to[h]
        m = len(self.positions)
        self.positions.append(0.5 * (self.positions[a] + self.positions[b]))
        self.v_he.append(BORDE)
        self.v_alive.append(True)

        h1 = self.he_next[h]       # b->c
        h2 = self.he_next[h1]      # c->a
        c = self.he_to[h1]
        f0 = self.he_face[h]
        t = self.he_twin[h]

        if t == BORDE:
            # solo se parte la cara de h: (a,b,c) -> (a,m,c) + (m,b,c).
            f_new = len(self.f_he)
            self.f_he.append(BORDE)
            self.f_alive.append(True)

            A = self._new_halfedge(m, BORDE, f0)     # a->m (borde)
            Mc = self._new_halfedge(c, BORDE, f0)    # m->c
            B = self._new_halfedge(b, BORDE, f_new)  # m->b (borde)
            Cm = self._new_halfedge(m, Mc, f_new)    # c->m
            self.he_twin[Mc] = Cm

            self.he_next[A] = Mc
            self.he_next[Mc] = h2
            self.he_next[h2] = A
            self.he_face[h2] = f0
            self.f_he[f0] = A

            self.he_next[B] = h1
            self.he_next[h1] = Cm
            self.he_next[Cm] = B
            self.he_face[h1] = f_new
            self.f_he[f_new] = B

            self.v_he[m] = Mc
            self.v_he[a] = A
            self.v_he[b] = h1
            self.v_he[c] = h2

            self.he_alive[h] = False
            return m

        # arista interior: ambas caras se parten en dos (cuatro triangulos).
        t1 = self.he_next[t]       # a->d
        t2 = self.he_next[t1]      # d->b
        d = self.he_to[t1]
        f1 = self.he_face[t]
        f_t1 = len(self.f_he)      # cara nueva del lado de h.
        f_t2 = f_t1 + 1            # cara nueva del lado de t.
        self.f_he += [BORDE, BORDE]
        self.f_alive += [True, True]

        # Cuatro triangulos: (a,m,c),(m,b,c),(b,m,d),(m,a,d).
        A = self._new_halfedge(m, BORDE, f0)     # a->m
        Mc = self._new_halfedge(c, BORDE, f0)    # m->c
        B = self._new_halfedge(b, BORDE, f1)     # m->b
        Cm = self._new_halfedge(m, Mc, f1)       # c->m
        Bm = self._new_halfedge(m, B, f_t1)      # b->m
        Md = self._new_halfedge(d, BORDE, f_t1)  # m->d
        Am = self._new_halfedge(a, A, f_t2)      # m->a
        Dm = self._new_halfedge(m, Md, f_t2)     # d->m
        self.he_twin[Mc] = Cm
        self.he_twin[B] = Bm
        self.he_twin[A] = Am
        self.he_twin[Md] = Dm

        # (a,m,c): A -> Mc -> h2
        self.he_next[A] = Mc
        self.he_next[Mc] = h2
        self.he_next[h2] = A
        self.he_face[h2] = f0
        self.f_he[f0] = A
        # (m,b,c): B -> h1 -> Cm
        self.he_next[B] = h1
        self.he_next[h1] = Cm
        self.he_next[Cm] = B
        self.he_face[h1] = f1
        self.f_he[f1] = B
        # (b,m,d): Bm -> Md -> t2
        self.he_next[Bm] = Md
        self.he_next[Md] = t2
        self.he_next[t2] = Bm
        self.he_face[t2] = f_t1
        self.f_he[f_t1] = Bm
        # (m,a,d): Am -> t1 -> Dm
        self.he_next[Am] = t1
        self.he_next[t1] = Dm
        self.he_next[Dm] = Am
        self.he_face[t1] = f_t2
        self.f_he[f_t2] = Am

        self.v_he[m] = Mc
        self.v_he[a] = A
        self.v_he[b] = h1
        self.v_he[c] = h2
        self.v_he[d] = t2

        self.he_alive[h] = False
        self.he_alive[t] = False
        return m

    # ------------------------------------------------------------------
    # Operacion elemental: collapse
    # ------------------------------------------------------------------
    def can_collapse(self, h):
        """True si colapsar la arista de h preserva una malla manifold.

        Condicion del enlace (link condition): los unicos vecinos comunes de
        los dos extremos deben ser los dos apices de las caras que comparten la
        arista. Si hay mas, el colapso crea una arista no-manifold.
        """
        t = self.he_twin[h]
        if t == BORDE:
            return False
        a = self.tail(h)
        b = self.he_to[h]
        if not (self.is_interior_vertex(a) and self.is_interior_vertex(b)):
            return False
        c = self.he_to[self.he_next[h]]
        d = self.he_to[self.he_next[t]]
        if c == d:
            return False
        common = self.neighbors(a) & self.neighbors(b)
        return common == {c, d}

    def collapse(self, h, new_position=None):
        """Funde la arista de h en un punto, eliminando el vertice origen.

        El vertice origen `a` se funde en el destino `b`, que se mueve a
        `new_position` (por defecto el punto medio). Desaparecen las dos caras
        que compartian la arista. Devuelve True si colapso.
        """
        if not self.can_collapse(h):
            return False

        t = self.he_twin[h]
        a = self.tail(h)
        b = self.he_to[h]

        h1 = self.he_next[h]   # b->c
        h2 = self.he_next[h1]  # c->a
        t1 = self.he_next[t]   # a->d
        t2 = self.he_next[t1]  # d->b
        c = self.he_to[h1]     # apice de la cara de h.
        d = self.he_to[t1]     # apice de la cara de t.
        f0 = self.he_face[h]
        f1 = self.he_face[t]

        # gemelas externas que sobreviven al borrar las dos caras.
        o1 = self.he_twin[h1]  # c->b (lado externo de la cara f0)
        o2 = self.he_twin[h2]  # a->c -> b->c
        p1 = self.he_twin[t1]  # d->a -> d->b
        p2 = self.he_twin[t2]  # b->d (lado externo de la cara f1)
        borradas = {h, h1, h2, t, t1, t2}
        if {o1, o2, p1, p2} & borradas or BORDE in (o1, o2, p1, p2):
            return False  # malla demasiado pequena o borde: no colapsar.

        # toda semi-arista que llegaba a `a` pasa a llegar a `b`.
        salientes, _ = self.outgoing_ring(a)
        for saliente in salientes:
            self.he_to[self.he_twin[saliente]] = b

        # re-emparejar las gemelas a traves de las caras borradas.
        self.he_twin[o1] = o2
        self.he_twin[o2] = o1
        self.he_twin[p1] = p2
        self.he_twin[p2] = p1

        # los vertices que sobreviven apuntan a semi-aristas vivas y salientes.
        # o2 quedo saliente de b; o1 sale de c; p1 sale de d.
        self.v_he[b] = o2
        self.v_he[c] = o1
        self.v_he[d] = p1

        if new_position is None:
            new_position = 0.5 * (self.positions[a] + self.positions[b])
        self.positions[b] = np.asarray(new_position, dtype=np.float64)

        self.v_alive[a] = False
        self.f_alive[f0] = False
        self.f_alive[f1] = False
        for halfedge in borradas:
            self.he_alive[halfedge] = False
        return True
