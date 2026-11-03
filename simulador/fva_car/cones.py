"""Centro do corredor usando geometria da cena, sem depender de pintura.

Este modo usa as posicoes reais dos cones do CoppeliaSim (ground truth).
Nao e um algoritmo de percepcao transferivel para o veiculo real.
"""
import math

import numpy as np

from .lateral import StanleyController, curve_speed


class ConeCenterPath:
    """Caminho entre pares de cores opostas, ligados por triangulos adjacentes.

    A adjacencia preserva a sequencia da pista em curvas e perto de outros
    ramos. Cada triangulo de duas cores liga exatamente dois pares de bordas.
    """
    def __init__(self, red, green, position, forward):
        from matplotlib.tri import Triangulation

        points = np.vstack((red, green))
        colors = np.arange(len(points)) < len(red)
        triangles = Triangulation(*points.T).triangles
        centers, widths, adjacency = {}, {}, {}
        for triangle in triangles:
            edges = []
            for i, j in zip(triangle, np.roll(triangle, -1)):
                if colors[i] == colors[j]:
                    continue
                edge = tuple(sorted((int(i), int(j))))
                width = float(np.linalg.norm(points[i]-points[j]))
                if 2.5 <= width <= 5.5:
                    centers[edge] = (points[i]+points[j])/2
                    widths[edge] = width
                    edges.append(edge)
            if len(edges) == 2:
                a, b = edges
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
        if not adjacency:
            raise ValueError('Nao foi possivel construir um corredor entre os cones.')
        start = min(adjacency, key=lambda e: np.linalg.norm(centers[e]-np.asarray(position)[:2]))
        previous, current, sequence = None, start, []
        while current not in sequence:
            sequence.append(current)
            options = adjacency[current] - {previous}
            if not options:
                break
            if previous is None:
                following = max(options, key=lambda e: np.dot(
                    centers[e]-centers[current], np.asarray(forward)[:2]))
            else:
                following = next(iter(options))
            previous, current = current, following
        self.closed = current == start
        if len(sequence) < 4:
            raise ValueError('Corredor muito curto ou desconectado.')
        nodes = np.asarray([centers[e] for e in sequence])
        node_widths = np.asarray([widths[e] for e in sequence])
        # Suaviza a alternancia de diagonais sem mudar a sequencia dos pares.
        for _ in range(2):
            smoothed = .25*np.roll(nodes, 1, axis=0)+.5*nodes+.25*np.roll(nodes, -1, axis=0)
            if not self.closed:
                smoothed[[0, -1]] = nodes[[0, -1]]
            nodes = smoothed
        # Interpolacao Catmull-Rom: tangentes continuas entre os pares de cones.
        dense, dense_width = [], []
        for i in range(len(nodes) if self.closed else len(nodes)-1):
            def node(j):
                return nodes[j % len(nodes)] if self.closed else nodes[np.clip(j, 0, len(nodes)-1)]
            p0, p1, p2, p3 = [node(j) for j in (i-1, i, i+1, i+2)]
            count = max(2, int(math.ceil(np.linalg.norm(p2-p1)/.08)))
            t = np.arange(count)/count
            segment = .5*(2*p1 + t[:, None]*(-p0+p2)
                          + t[:, None]**2*(2*p0-5*p1+4*p2-p3)
                          + t[:, None]**3*(-p0+3*p1-3*p2+p3))
            dense.extend(segment)
            dense_width.extend((1-t)*node_widths[i]+t*node_widths[(i+1) % len(nodes)])
        if not self.closed:
            dense.append(nodes[-1])
            dense_width.append(node_widths[-1])
        self.points = np.asarray(dense)
        self.widths = np.asarray(dense_width)
        self.index = 0

    def errors(self, front, forward):
        # Busca local na sequencia, impedindo saltos para uma pista vizinha.
        n = len(self.points)
        indices = np.arange(self.index-30, self.index+150)
        indices = indices % n if self.closed else indices[(indices >= 0) & (indices < n)]
        following = (indices+1) % n
        if not self.closed:
            following = np.minimum(following, n-1)
        a, b = self.points[indices], self.points[following]
        delta = b-a
        length2 = np.sum(delta*delta, axis=1)
        t = np.clip(np.sum((np.asarray(front)[:2]-a)*delta, axis=1)/np.maximum(length2, 1e-12), 0., 1.)
        feet = a+t[:, None]*delta
        chosen = int(np.argmin(np.sum((feet-np.asarray(front)[:2])**2, axis=1)))
        self.index = int(indices[chosen])
        tangent = delta[chosen]/max(math.sqrt(length2[chosen]), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        lateral = float(np.dot(feet[chosen]-np.asarray(front)[:2], normal))
        heading = math.atan2(tangent[1], tangent[0])-math.atan2(forward[1], forward[0])
        heading = math.atan2(math.sin(heading), math.cos(heading))
        # Curvatura por tres pontos separados por cerca de meio metro.
        before, after = self.index-6, self.index+6
        if self.closed:
            before, after = before % n, after % n
        else:
            before, after = max(0, before), min(n-1, after)
        p, q, r = self.points[[before, self.index, after]]
        u, v = q-p, r-q
        denominator = np.linalg.norm(u)*np.linalg.norm(v)*np.linalg.norm(r-p)
        curvature = 2*(u[0]*v[1]-u[1]*v[0])/max(denominator, 1e-12)
        valid = abs(lateral) < self.widths[self.index]/2-.15 and abs(heading) < math.pi/2
        if not self.closed and self.index >= n-3:
            valid = False
        return (lateral, heading, float(self.widths[self.index]), float(curvature)) if valid else None


class ConeCenterFollower:
    def __init__(self, car):
        sim = car.sim
        red, green = [], []
        for handle in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
            alias = sim.getObjectAlias(handle).lower()
            if 'cone' not in alias:
                continue
            if 'vermelho' in alias:
                destination = red
            elif 'verde' in alias:
                destination = green
            else:
                color = sim.getShapeColor(handle, None, sim.colorcomponent_ambient_diffuse)
                if len(color) == 2 and isinstance(color[1], (list, tuple)):
                    color = color[1]  # API retorna (resultado, RGB)
                destination = red if color[0] > color[1] else green
            destination.append(sim.getObjectPosition(handle, -1)[:2])
        self.red = np.unique(np.asarray(red).reshape(-1, 2), axis=0)
        self.green = np.unique(np.asarray(green).reshape(-1, 2), axis=0)
        if min(len(self.red), len(self.green)) < 3:
            raise ValueError('A cena precisa de cones vermelhos e verdes identificados pelos nomes ou cores.')
        self.front_local = (np.asarray(sim.getObjectPosition(car.steerL, car.robot))
                            + np.asarray(sim.getObjectPosition(car.steerR, car.robot)))/2
        self.sign = car.parameters.get('steering_sign', -1.)
        if self.sign not in (-1., 1.):
            raise ValueError('steering_sign deve ser -1 ou +1.')
        matrix = np.asarray(sim.getObjectMatrix(car.robot, -1)).reshape(3, 4)
        self.path = ConeCenterPath(self.red, self.green, matrix[:, 3], matrix[:, 2])
        self.controller = StanleyController(gain=1.5, softening=.5,
                                            max_rate=math.radians(40.))

    def __call__(self, car):
        matrix = np.asarray(car.sim.getObjectMatrix(car.robot, -1)).reshape(3, 4)
        front = matrix[:, :3] @ self.front_local + matrix[:, 3]
        errors = self.path.errors(front, matrix[:, 2])
        if errors is None:
            lateral = heading = width = curvature = float('nan')
            self.controller.steer = 0.
            steer = 0.
            car.set_vel(0.)
        else:
            lateral, heading, width, curvature = errors
            anticipation = math.atan(.302*curvature)
            steer = self.controller.update(lateral, heading+anticipation, car.v, car.dt)
            car.set_vel(curve_speed(car.parameters.get('velocity_reference', 1.),
                                    curvature, lateral, heading,
                                    lateral_accel=car.parameters.get("curve_lateral_accel", .6)))
        car.set_steer(self.sign*steer)
        car.log(line_valid=int(errors is not None), cross_track_m=lateral,
                heading_error_rad=heading, corridor_width_m=width, curvature_m_inv=curvature,
                cone_left_distance_m=width/2+lateral,
                cone_right_distance_m=width/2-lateral,
                steer_geometric_rad=steer, steer_cmd_rad=float(car.st),
                steer_left_rad=car.sim.getJointPosition(car.steerL),
                steer_right_rad=car.sim.getJointPosition(car.steerR),
                vref=car.vref, u=car.u, motor_pwm_percent=car.motor_pwm_percent)
