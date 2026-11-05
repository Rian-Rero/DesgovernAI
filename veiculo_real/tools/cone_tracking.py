"""Mapa local de cones com associacao por cor, Kalman e expiracao por cone."""

from dataclasses import dataclass

import cv2
import numpy as np


def transform_points(points, matrix):
    points = np.asarray(points, float).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    valid = np.all(np.isfinite(homogeneous), axis=1) & (abs(homogeneous[:, 2]) > 1e-8)
    output = np.full_like(points, np.nan)
    output[valid] = homogeneous[valid, :2]/homogeneous[valid, 2, None]
    return output


def ego_transform(motion):
    """Delta do carro (frente m, esquerda m, yaw rad) -> mapa no novo referencial."""
    delta = np.asarray(motion, float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError('motion requer (frente_m, esquerda_m, yaw_rad) finitos.')
    c, s = np.cos(delta[2]), np.sin(delta[2])
    rotation = np.array([[c, s], [-s, c]])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = -rotation @ delta[:2]
    return matrix


@dataclass
class Track:
    color: str
    point: np.ndarray
    variance: float
    last_seen: float
    hits: int = 1
    confirmed: bool = True  # Tentativo ate confirm_hits observacoes seguidas.


class ConeTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = []
        self.timestamp = None

    def update(self, observations, now, ttl, *, matrix=None, metric=False, smoothing=35,
               confirm_hits=1, behind_x=None):
        """confirm_hits: observacoes em quadros seguidos para um cone novo entrar
        nas bordas (um falso positivo de um quadro nao entra). Sem mapa anterior
        (inicio ou perda total), aceita de imediato. behind_x: com odometria,
        descarta cones que ficaram atras do eixo (x < behind_x, em metros)."""
        if self.timestamp is not None and now <= self.timestamp:
            self.reset()
        dt = 0. if self.timestamp is None else now-self.timestamp
        self.timestamp = now
        self.tracks = [t for t in self.tracks if now-t.last_seen <= ttl]
        bootstrap = not any(t.confirmed for t in self.tracks)
        noise = .035 if metric else 2.
        process_noise = .12 if metric else 12.
        gate = .35 if metric else 28.
        if matrix is not None and self.tracks:
            points = transform_points([t.point for t in self.tracks], matrix)
            for track, point in zip(self.tracks, points):
                track.point = point
        self.tracks = [t for t in self.tracks if np.all(np.isfinite(t.point))]
        if behind_x is not None:
            self.tracks = [t for t in self.tracks if t.point[0] >= behind_x]
        for track in self.tracks:
            track.variance += (process_noise*dt)**2
        for color in ('green', 'red'):
            points = np.asarray(observations[color], float).reshape(-1, 2)
            tracks = [t for t in self.tracks if t.color == color]
            pairs = []
            for i, point in enumerate(points):
                for j, track in enumerate(tracks):
                    distance = float(np.linalg.norm(point-track.point))
                    if distance <= gate:
                        pairs.append((distance, i, j))
            used_points, used_tracks = set(), set()
            for _, i, j in sorted(pairs):
                if i in used_points or j in used_tracks:
                    continue
                used_points.add(i)
                used_tracks.add(j)
                track = tracks[j]
                gain = track.variance/(track.variance+noise**2)
                gain = float(np.clip(gain, 1-smoothing/100, 1.))
                track.point += gain*(points[i]-track.point)
                track.variance = max(noise**2*.1, (1-gain)*track.variance)
                track.last_seen = now
                track.hits += 1
                track.confirmed = track.confirmed or track.hits >= confirm_hits
            for i, point in enumerate(points):
                if i not in used_points:
                    self.tracks.append(Track(color, point.copy(), noise**2, now,
                                             confirmed=bootstrap or confirm_hits <= 1))
        # Tentativo nao revisto neste quadro sai: precisa de quadros seguidos.
        self.tracks = [t for t in self.tracks if t.confirmed or t.last_seen >= now]
        # Limite de trabalho/ruido: cada cor conserva os 80 cones mais recentes.
        self.tracks = [t for color in ('green', 'red')
                       for t in sorted((t for t in self.tracks if t.color == color),
                                       key=lambda t: t.last_seen, reverse=True)[:80]]

    def points(self, color, confirmed_only=True):
        return np.asarray([t.point for t in self.tracks if t.color == color
                           and (t.confirmed or not confirmed_only)], float).reshape(-1, 2)

    def age(self, now, freshest=False):
        """Idade do mapa: do cone mais antigo ou, com odometria, do mais recente."""
        ages = [now-t.last_seen for t in self.tracks if t.confirmed]
        if not ages:
            return 0.
        return min(ages) if freshest else max(ages)


class FloorMotion:
    """Movimento entre frames via fluxo optico no piso e homografia RANSAC.

    Retorna None quando nao ha suporte suficiente. Nao inventa odometria em
    uma imagem sem textura; a memoria continuara sujeita ao TTL e confianca.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.gray = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous, self.gray = self.gray, gray
        if previous is None or previous.shape != gray.shape:
            return None
        mask = np.zeros_like(gray)
        mask[int(.55*len(gray)):] = 255
        points = cv2.goodFeaturesToTrack(previous, 120, .015, 8, mask=mask)
        if points is None or len(points) < 12:
            return None
        current, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if current is None:
            return None
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, previous, current, None)
        if back is None:
            return None
        good = status.ravel().astype(bool) & backward.ravel().astype(bool)
        good &= np.linalg.norm((points-back).reshape(-1, 2), axis=1) < 1.5
        a, b = points[good].reshape(-1, 2), current[good].reshape(-1, 2)
        if len(a) < 12:
            return None
        matrix, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.)
        if matrix is None or inliers is None or np.mean(inliers) < .65:
            return None
        h, w = gray.shape
        probes = np.array([[0, h-1], [w-1, h-1], [w/2, .6*h]])
        moved = transform_points(probes, matrix)
        if not np.all(np.isfinite(moved)) or np.max(np.linalg.norm(moved-probes, axis=1)) > .3*w:
            return None
        return matrix
