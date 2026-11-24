"""Pose e foco da camera a partir da homografia do chao (calibracao da GUI).

A calibracao guarda so a homografia pixel <-> chao. Ela e K [r1 r2 t]: com o
centro optico no centro da imagem e pixels quadrados, as colunas r1, r2 sao
ortonormais, e isso da o foco f (duas equacoes lineares em 1/f^2, resolvidas
por minimos quadrados; so a de "mesma norma" ja basta sem giro, que e o caso do
carro). Dai saem a rotacao R e a posicao da camera, e um ponto FORA do chao
(placa ArUco ao lado da pista) pode ser medido:

    pelo tamanho:  distancia Z = f * lado_real / lado_em_pixels, na direcao do pixel
    pela altura:   o raio do pixel cortando o plano z = altura do centro da placa

Na calibracao de 19/11: f = 521 px (63 graus de campo horizontal).
Referencial do chao: o da calibracao (eixo dianteiro, x frente, y esquerda, z cima).
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    focal_px: float
    center_px: tuple
    rotation: np.ndarray    # mundo -> camera
    translation: np.ndarray
    position: np.ndarray    # centro da camera no referencial do chao (m)

    @property
    def height_m(self):
        return float(self.position[2])

    def ray(self, pixel):
        """Direcao (mundo) do raio do pixel."""
        u, v = pixel
        direction = np.array([(u - self.center_px[0])/self.focal_px, (v - self.center_px[1])/self.focal_px, 1.])
        return self.rotation.T @ direction

    def point_from_size(self, pixel, side_px, side_m):
        """Ponto 3D (mundo) de um quadrado de lado side_m visto com side_px pixels."""
        if not side_px > 0 or not side_m > 0:
            return None
        depth = self.focal_px*side_m/side_px
        u, v = pixel
        camera = np.array([(u - self.center_px[0])*depth/self.focal_px,
                           (v - self.center_px[1])*depth/self.focal_px, depth])
        return self.rotation.T @ (camera - self.translation)

    def point_at_height(self, pixel, height_m):
        """Ponto 3D (mundo) onde o raio do pixel corta o plano z = height_m."""
        direction = self.ray(pixel)
        if abs(direction[2]) < 1e-6:
            return None
        scale = (height_m - self.position[2])/direction[2]
        return self.position + scale*direction if scale > 0 else None


def camera_pose(ground_from_pixel, image_size):
    """CameraPose pela homografia pixel -> chao, ou None se ela nao for de camera real."""
    G = np.linalg.inv(np.asarray(ground_from_pixel, float))   # chao -> pixel
    cx, cy = (image_size[0] - 1)/2., (image_size[1] - 1)/2.
    shifted = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.]]) @ G
    h1, h2 = shifted[:, 0], shifted[:, 1]
    # (a) h1 . h2 = 0 e (b) |h1| = |h2| em K^-1: cada uma da w = 1/f^2. Sem giro
    # nem rolagem, (a) e degenerada (0/0) e o ruido do ajuste a deixa negativa
    # (calibracao de 19/11): vale a media das que dao um foco possivel.
    a = (h1[0]*h2[0] + h1[1]*h2[1], h1[0]**2 + h1[1]**2 - h2[0]**2 - h2[1]**2)
    b = (h1[2]*h2[2], h1[2]**2 - h2[2]**2)
    focals = []
    for ai, bi in zip(a, b):
        if abs(ai) > 1e-12:
            w = -bi/ai
            if w > 0 and 50. < 1./math.sqrt(w) < 5000.:
                focals.append(1./math.sqrt(w))
    if not focals:
        return None
    focal = float(np.exp(np.mean(np.log(focals))))
    K_inv = np.array([[1/focal, 0, 0], [0, 1/focal, 0], [0, 0, 1.]])
    A = K_inv @ shifted
    for sign in (1., -1.):
        lam = sign*2./(np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
        r1, r2, t = lam*A[:, 0], lam*A[:, 1], lam*A[:, 2]
        U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
        R = U @ Vt
        position = -R.T @ t
        if position[2] > 0:   # camera acima do chao
            return CameraPose(focal, (cx, cy), R, t, position)
    return None


def marker_ground(pose, ground_from_pixel, pixel, side_px, side_m):
    """(x, y, altura) em m de um marcador de lado side_m: direcao pela homografia,
    distancia pelo tamanho aparente. None sem dados para isso.

    A DIRECAO vem da propria homografia da calibracao (exata para ela): o raio do
    pixel corta o chao em P, e o marcador esta no segmento camera -> P. A
    DISTANCIA (ao longo do raio) vem do tamanho: f*lado/lado_px, corrigida pela
    direcao do pixel. Assim o erro de centro optico da decomposicao (~40 px na
    calibracao de 19/11) nao vira erro lateral da placa."""
    if pose is None or not side_px > 0 or not side_m > 0:
        return None
    u, v = pixel
    normalized = np.array([(u - pose.center_px[0])/pose.focal_px, (v - pose.center_px[1])/pose.focal_px, 1.])
    distance = pose.focal_px*side_m/side_px*float(np.linalg.norm(normalized))
    q = np.asarray(ground_from_pixel, float) @ np.array([u, v, 1.])
    camera = pose.position
    if q[2] != 0:
        far = q[:2]/q[2]
        reach = math.hypot(*(far - camera[:2]))
        along = far[0] - camera[0]
        if along > 0 and reach > 0:
            fraction = distance/math.hypot(reach, camera[2])
            if 0 < fraction < 1:
                point = camera[:2] + fraction*(far - camera[:2])
                return float(point[0]), float(point[1]), float(camera[2]*(1 - fraction))
    point = pose.point_from_size(pixel, side_px, side_m)   # acima do horizonte: so a pose
    return None if point is None else (float(point[0]), float(point[1]), float(point[2]))
