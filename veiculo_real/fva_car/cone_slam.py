"""SLAM de cones (EKF-SLAM): pose do carro e posicao de cada cone estimadas juntas.

A camera so ve cones de ~0,8 a 3 m a frente, num cone de visao estreito. Os cones
ao lado e atras do carro (os que mais importam para nao bater numa curva) ja
sairam da imagem. Aqui cada cone visto vira um marco num mapa local:

    estado   [x, y, th, l1x, l1y, l2x, l2y, ...]   (referencial do mapa, m e rad)
    previsao odometria do eixo dianteiro (avanco, lateral, guinada); o ruido
             cresce com o andado (a guinada pela direcao, sem IMU, e a pior)
    medida   cada cone (x, y) no referencial do carro, com a covariancia da
             homografia; associado ao marco da mesma cor mais proximo pela
             distancia de Mahalanobis (portao de 99%); sem par, vira marco novo

Revendo um cone, o filtro corrige a POSE do carro (a deriva da odometria) e a
posicao do cone. Marcos vistos uma vez so ficam se reaparecerem em pouco tempo
(reflexo, falso positivo); marcos que ficaram para tras (alem de drop_behind_m)
saem do mapa. landmarks() devolve o mapa no referencial ATUAL do carro (eixo
dianteiro, x frente, y esquerda), pronto para o tracado (svm_path.py).
"""
from dataclasses import dataclass
import math

import numpy as np

COLORS = ('green', 'red')


@dataclass(frozen=True)
class SlamSettings:
    gate_chi2: float = 9.21          # 2 graus de liberdade, 99%
    distance_noise: float = .05      # desvio do avanco por metro andado
    lateral_noise: float = .02       # deriva lateral por metro
    heading_noise: float = .06       # rad por metro (direcao sem IMU)
    heading_rate_noise: float = .1   # fracao da guinada da odometria
    min_observation_sigma: float = .03  # m, piso do erro de cada cone
    confirm_hits: int = 2            # vezes vistas para o marco valer no tracado
    tentative_s: float = 1.          # marco visto 1 vez e nao revisto neste tempo sai
    drop_behind_m: float = 1.2       # marco mais que isto atras do eixo sai
    drop_range_m: float = 6.         # ou mais longe que isto
    max_landmarks: int = 40


class ConeSLAM:
    def __init__(self, settings=None):
        self.settings = settings or SlamSettings()
        self.reset()

    def reset(self):
        self.mu = np.zeros(3)
        self.P = np.diag([1e-6, 1e-6, 1e-6])
        self.colors = []
        self.hits = []
        self.last_seen = []
        self.time = None

    @property
    def count(self):
        return len(self.colors)

    # ------------------------------------------------------------------ previsao
    def predict(self, motion):
        """motion: (avanco, lateral, guinada) do EIXO DIANTEIRO desde o quadro anterior."""
        if motion is None or not np.all(np.isfinite(motion)):
            return
        dx, dy, dth = (float(v) for v in motion)
        x, y, th = self.mu[:3]
        c, s = math.cos(th), math.sin(th)
        self.mu[0] = x + c*dx - s*dy
        self.mu[1] = y + s*dx + c*dy
        self.mu[2] = math.atan2(math.sin(th + dth), math.cos(th + dth))
        F = np.eye(3)
        F[0, 2] = -s*dx - c*dy
        F[1, 2] = c*dx - s*dy
        cfg = self.settings
        distance = math.hypot(dx, dy)
        q_local = np.diag([(cfg.distance_noise*distance)**2, (cfg.lateral_noise*distance)**2])
        rotation = np.array([[c, -s], [s, c]])
        Q = np.zeros((3, 3))
        Q[:2, :2] = rotation @ q_local @ rotation.T
        Q[2, 2] = (cfg.heading_noise*distance)**2 + (cfg.heading_rate_noise*dth)**2 + 1e-8
        n = len(self.mu)
        J = np.eye(n)
        J[:3, :3] = F
        self.P = J @ self.P @ J.T
        self.P[:3, :3] += Q

    # ------------------------------------------------------------------ medida
    def _expected(self, index):
        """Marco `index` no referencial do carro e o jacobiano (2 x n)."""
        x, y, th = self.mu[:3]
        k = 3 + 2*index
        lx, ly = self.mu[k:k+2]
        c, s = math.cos(th), math.sin(th)
        ddx, ddy = lx - x, ly - y
        z = np.array([c*ddx + s*ddy, -s*ddx + c*ddy])
        H = np.zeros((2, len(self.mu)))
        H[:, 0:2] = [[-c, -s], [s, -c]]
        H[:, 2] = [-s*ddx + c*ddy, -c*ddx - s*ddy]
        H[:, k:k+2] = [[c, s], [-s, c]]
        return z, H

    def _relative(self):
        """Todos os marcos no referencial do carro (m x 2), de uma vez."""
        x, y, th = self.mu[:3]
        c, s = math.cos(th), math.sin(th)
        d = self.mu[3:].reshape(-1, 2) - (x, y)
        return np.column_stack((c*d[:, 0] + s*d[:, 1], -s*d[:, 0] + c*d[:, 1]))  # np.c_ e lento

    def _gate(self, candidates, point, R):
        """Distancia de Mahalanobis da medida a cada marco candidato, vetorizada.

        Antes era um _expected() + solve por marco e por medida: ~1,4 ms/quadro
        no notebook (dezenas de ms no Pi) so na associacao."""
        x, y, th = self.mu[:3]
        c, s = math.cos(th), math.sin(th)
        idx = np.asarray(candidates)
        cols = 3 + 2*idx
        rows = np.column_stack((cols, cols + 1))                       # (m, 2)
        d = self.mu[rows] - (x, y)                                     # (m, 2)
        z = np.column_stack((c*d[:, 0] + s*d[:, 1], -s*d[:, 0] + c*d[:, 1]))
        m = len(idx)
        Hp = np.zeros((m, 2, 3))
        Hp[:, 0, 0], Hp[:, 0, 1], Hp[:, 1, 0], Hp[:, 1, 1] = -c, -s, s, -c
        Hp[:, 0, 2] = -s*d[:, 0] + c*d[:, 1]
        Hp[:, 1, 2] = -c*d[:, 0] - s*d[:, 1]
        Hl = np.array([[c, s], [-s, c]])
        P = self.P
        Ppp = P[:3, :3]
        Ppl = P[:3][:, rows].transpose(1, 0, 2)                        # (m, 3, 2)
        Pll = P[rows[:, :, None], rows[:, None, :]]                    # (m, 2, 2)
        cross = Hp @ Ppl @ Hl.T                                        # (m, 2, 2)
        S = Hp @ Ppp @ Hp.transpose(0, 2, 1) + cross + cross.transpose(0, 2, 1) + Hl @ Pll @ Hl.T + R
        innovation = point - z
        d2 = np.einsum('mi,mi->m', innovation, np.linalg.solve(S, innovation[:, :, None])[:, :, 0])
        return d2

    def _add(self, point, covariance, color, now):
        x, y, th = self.mu[:3]
        c, s = math.cos(th), math.sin(th)
        rotation = np.array([[c, -s], [s, c]])
        world = np.array([x, y]) + rotation @ point
        n = len(self.mu)
        # Jacobianos da inicializacao em relacao a pose e a medida.
        Gp = np.array([[1., 0., -s*point[0] - c*point[1]], [0., 1., c*point[0] - s*point[1]]])
        Gz = rotation
        P = np.zeros((n + 2, n + 2))
        P[:n, :n] = self.P
        cross = Gp @ self.P[:3, :]
        P[n:, :n] = cross
        P[:n, n:] = cross.T
        P[n:, n:] = Gp @ self.P[:3, :3] @ Gp.T + Gz @ covariance @ Gz.T
        self.mu = np.r_[self.mu, world]
        self.P = P
        self.colors.append(color)
        self.hits.append(1)
        self.last_seen.append(now)

    def update(self, observations, now):
        """observations: lista de (x, y, cor, covariancia 2x2) no referencial do carro.

        Devolve quantas foram associadas a marcos existentes."""
        cfg = self.settings
        self.time = now
        matched, used = 0, set()
        floor = np.eye(2)*cfg.min_observation_sigma**2
        for x, y, color, covariance in observations:
            point = np.array([x, y], float)
            R = np.asarray(covariance, float) + floor
            best = None
            candidates = [k for k in range(self.count) if self.colors[k] == color and k not in used]
            if candidates:
                d2 = self._gate(candidates, point, R)
                nearest = int(np.argmin(d2))
                if d2[nearest] < cfg.gate_chi2:
                    best = candidates[nearest]
            if best is None:
                if self.count < cfg.max_landmarks:
                    self._add(point, R, color, now)
                    used.add(self.count - 1)
                continue
            z, H = self._expected(best)
            S = H @ self.P @ H.T + R
            K = np.linalg.solve(S, H @ self.P).T
            self.mu = self.mu + K @ (point - z)
            self.mu[2] = math.atan2(math.sin(self.mu[2]), math.cos(self.mu[2]))
            I_KH = np.eye(len(self.mu)) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self.hits[best] += 1
            self.last_seen[best] = now
            used.add(best)
            matched += 1
        self._prune(now)
        return matched

    def _prune(self, now):
        cfg = self.settings
        keep = []
        relative = self._relative()
        for index in range(self.count):
            z = relative[index]
            stale = self.hits[index] < cfg.confirm_hits and now - self.last_seen[index] > cfg.tentative_s
            gone = z[0] < -cfg.drop_behind_m or math.hypot(*z) > cfg.drop_range_m
            if not (stale or gone):
                keep.append(index)
        if len(keep) == self.count:
            return
        rows = [0, 1, 2] + [3 + 2*k + j for k in keep for j in (0, 1)]
        self.mu = self.mu[rows]
        self.P = self.P[np.ix_(rows, rows)]
        self.colors = [self.colors[k] for k in keep]
        self.hits = [self.hits[k] for k in keep]
        self.last_seen = [self.last_seen[k] for k in keep]

    # ------------------------------------------------------------------ saida
    def landmarks(self, min_hits=None):
        """{cor: Nx2} no referencial ATUAL do carro (so marcos confirmados)."""
        min_hits = self.settings.confirm_hits if min_hits is None else min_hits
        out = {color: [] for color in COLORS}
        relative = self._relative()
        for index in range(self.count):
            if self.hits[index] >= min_hits:
                out[self.colors[index]].append(relative[index])
        return {color: np.asarray(points, float).reshape(-1, 2) for color, points in out.items()}

    def step(self, timestamp, cones, motion, covariance_fn=None):
        """Previsao + medida. cones: {cor: Nx2} no referencial do carro;
        covariance_fn(pontos) -> (N, 2, 2) (homografia) ou None (piso fixo)."""
        if self.time is not None and timestamp <= self.time:
            self.reset()
        self.predict(motion)
        observations = []
        for color in COLORS:
            points = np.asarray(cones.get(color, ()), float).reshape(-1, 2)
            if not len(points):
                continue
            covariances = covariance_fn(points) if covariance_fn is not None else np.zeros((len(points), 2, 2))
            observations.extend((p[0], p[1], color, C) for p, C in zip(points, covariances))
        return self.update(observations, float(timestamp))
