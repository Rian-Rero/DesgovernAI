"""Referencias geometricas (reta, circulo e oito) para o controle lateral.

Cada trajetoria e uma polilinha densa no plano do mundo, gerada a partir da
pose inicial do carro. O erro e calculado no centro do eixo dianteiro, com as
mesmas convencoes do Stanley em lateral.py:
  e_y   > 0 quando a referencia esta a ESQUERDA do eixo dianteiro [m]
  e_psi = direcao da tangente - direcao do carro, em (-pi, pi] [rad]
A busca do ponto mais proximo usa uma janela que so avanca, para nao saltar
de ramo no cruzamento do oito.
"""
import math

import numpy as np


def _unit(angle):
    return np.array([math.cos(angle), math.sin(angle)])


class PolylinePath:
    def __init__(self, points, closed=False, step=.02):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError('A trajetoria requer pelo menos dois pontos 2D.')
        self.points, self.closed = points, closed
        delta = np.diff(points, axis=0)
        self.heading = np.arctan2(delta[:, 1], delta[:, 0])
        self.heading = np.append(self.heading, self.heading[-1])
        segment = np.hypot(delta[:, 0], delta[:, 1])
        self.s = np.concatenate(([0.], np.cumsum(segment)))
        self.length = float(self.s[-1])
        self.index = 0
        # Janela de busca: ~1.5 m adiante, 0.3 m para tras.
        self.ahead = max(2, int(1.5 / step))
        self.behind = max(1, int(.3 / step))
        self.laps = 0

    def errors(self, front, heading):
        """Retorna (e_y, e_psi, s) para o eixo dianteiro em `front`."""
        n = len(self.points)
        lo, hi = self.index - self.behind, self.index + self.ahead
        if self.closed:
            candidates = np.arange(lo, hi) % n
        else:
            candidates = np.arange(max(0, lo), min(n, hi))
        diff = self.points[candidates] - np.asarray(front)[:2]
        best = int(candidates[np.argmin(np.einsum('ij,ij->i', diff, diff))])
        if self.closed and best < self.index - n // 2:
            self.laps += 1
        elif self.closed and best > self.index + n // 2:
            self.laps -= 1
        self.index = best
        tangent = self.heading[best]
        # Distancia assinada na normal esquerda da referencia.
        normal = _unit(tangent + math.pi / 2)
        e_y = float(np.dot(self.points[best] - np.asarray(front)[:2], normal))
        e_psi = math.atan2(math.sin(tangent - heading), math.cos(tangent - heading))
        return e_y, e_psi, float(self.s[best] + self.laps * self.length)

    @property
    def finished(self):
        return (not self.closed) and self.index >= len(self.points) - 2


def straight(start, heading, length=28., offset=.5, step=.02):
    """Reta paralela a direcao inicial, deslocada `offset` m para a esquerda."""
    start = np.asarray(start, dtype=float)[:2] + offset * _unit(heading + math.pi / 2)
    s = np.arange(0., length + step, step)
    return PolylinePath(start + s[:, None] * _unit(heading), step=step)


def circle(start, heading, radius=3., step=.02):
    """Circulo anti-horario tangente a pose inicial (centro a esquerda)."""
    center = np.asarray(start, dtype=float)[:2] + radius * _unit(heading + math.pi / 2)
    phi0 = heading - math.pi / 2
    phi = phi0 + np.arange(0., 2 * math.pi, step / radius)
    points = center + radius * np.column_stack((np.cos(phi), np.sin(phi)))
    return PolylinePath(points, closed=True, step=step)


def figure_eight(start, heading, radius=3., step=.02):
    """Oito: circulo a esquerda (anti-horario) seguido de circulo a direita.

    Os dois circulos sao tangentes no ponto inicial, onde a curvatura troca
    de sinal (degrau de curvatura 2/R, o caso mais exigente para o lateral).
    """
    start = np.asarray(start, dtype=float)[:2]
    left_center = start + radius * _unit(heading + math.pi / 2)
    right_center = start - radius * _unit(heading + math.pi / 2)
    d = step / radius
    phi_l = (heading - math.pi / 2) + np.arange(0., 2 * math.pi, d)
    phi_r = (heading + math.pi / 2) - np.arange(0., 2 * math.pi, d)
    points = np.vstack((
        left_center + radius * np.column_stack((np.cos(phi_l), np.sin(phi_l))),
        right_center + radius * np.column_stack((np.cos(phi_r), np.sin(phi_r))),
    ))
    return PolylinePath(points, closed=True, step=step)


def curvature_at(path, index):
    """Curvatura nominal (1/m) por diferenca de direcao entre vizinhos."""
    n = len(path.points)
    i0, i1 = (index - 1) % n, (index + 1) % n
    dpsi = math.atan2(math.sin(path.heading[i1] - path.heading[i0]),
                      math.cos(path.heading[i1] - path.heading[i0]))
    ds = np.hypot(*(path.points[i1] - path.points[i0]))
    return dpsi / ds if ds > 0 else 0.
