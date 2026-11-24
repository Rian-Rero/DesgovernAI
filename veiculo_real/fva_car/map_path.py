"""Tracado pelo mapa: SLAM dos cones (cone_slam.py) + centro pelo SVM (svm_path.py).

Roda junto com o estimador do corredor (corridor.py), que continua decidindo
estado, largura e velocidade. A cada quadro:

  1. o SLAM recebe os cones do quadro (com a covariancia da homografia) e a
     odometria do eixo dianteiro, e mantem o mapa no referencial do carro, com
     os cones que ja sairam da imagem (ao lado e atras);
  2. se o estimador PERDEU o corredor (sem cones aceitos alem da memoria), o SVM
     acha o centro de margem maxima entre a borda esquerda e a direita do mapa
     e um polinomio ponderado pela proximidade da (e, rumo, curvatura) no eixo:
     o carro segue devagar (estado MAPA) em vez de parar.

Por que so de reserva (pista virtual, 5 pistas x 4 sementes, 20/11): com cones
a 1 m, a corda entre dois cones de fora corta o arco nas curvas fechadas e o
centro de margem maxima fica 10-15 cm deslocado; o estimador (modelo de
clotoide, largura conhecida) erra 5-6 cm. Substituindo o estimador
(replace=True) o erro rms foi de 6,7 para 9-13 cm e a folga minima de 29 para
5-12 cm; como reserva, igual ou melhor (5 fps com iscas: 8 -> 7 toques).
"""
from dataclasses import replace
import math

import numpy as np

from .cone_slam import ConeSLAM, SlamSettings
from .corridor import CorridorEstimate
from .svm_path import centerline, path_errors


class MapPath:
    def __init__(self, wheelbase=.36, default_width=1.3, window=(-1., 3.5), slam_settings=None, replace=False):
        """replace: o tracado do mapa substitui tambem o do estimador valido
        (experimento; no simulador piorou). Padrao: so com o estimador perdido."""
        self.slam = ConeSLAM(slam_settings or SlamSettings())
        self.replace = bool(replace)
        self.wheelbase = float(wheelbase)
        self.default_width = float(default_width)
        self.window = window
        self.last = {}

    def reset(self):
        self.slam.reset()

    def _front_motion(self, motion, pose_frame):
        if motion is None or not np.all(np.isfinite(motion)):
            return None
        dx, dy, dth = (float(v) for v in motion)
        if pose_frame:
            dx += self.wheelbase*(math.cos(dth) - 1)
            dy += self.wheelbase*math.sin(dth)
        return dx, dy, dth

    def step(self, timestamp, green, red, green_left, motion, pose_frame, estimate, covariance_fn=None):
        """green/red: cones do quadro (Nx2, referencial do carro). Devolve o
        CorridorEstimate com os erros do tracado do mapa (ou o original)."""
        self.slam.step(timestamp, {'green': green, 'red': red}, self._front_motion(motion, pose_frame),
                       covariance_fn)
        cones = self.slam.landmarks()
        low, high = self.window
        inside = {c: p[(p[:, 0] > low) & (p[:, 0] < high)] for c, p in cones.items()}
        left, right = (inside['green'], inside['red']) if green_left else (inside['red'], inside['green'])
        width = estimate.width_m if math.isfinite(estimate.width_m) else (
            estimate.learned_width_m if math.isfinite(estimate.learned_width_m) else self.default_width)
        self.last = {'marcos': self.slam.count, 'esquerda': len(left), 'direita': len(right), 'usado': False}
        if len(left) + len(right) < 3 or (estimate.valid and not self.replace):
            return estimate  # reserva: com o estimador valido, so o mapa e mantido (sem SVM)
        points, margin = centerline(left, right, width)
        errors = path_errors(points) if points is not None else None
        if errors is None or abs(errors[0]) > .8*max(width, .5) or abs(errors[1]) > 1.2:
            return estimate
        e, psi, kappa, dkappa = errors
        both_near = (np.sum(left[:, 0] < 2.) >= 1 and np.sum(right[:, 0] < 2.) >= 1)
        if not estimate.valid and not both_near:
            return estimate
        self.last.update(usado=True, margem_m=round(margin, 3), e=round(e, 4), rumo=round(psi, 4))
        vector = (e, psi, kappa, dkappa, float(width))
        if estimate.valid:
            return replace(estimate, errors=(e, psi, kappa), state_vector=vector,
                           reason=estimate.reason + ' | tracado do mapa (SVM)')
        return CorridorEstimate(True, 'MAPA', (e, psi, kappa), float(width), estimate.sigma,
                                (len(left), len(right)), estimate.rejected, estimate.blind_m,
                                'Mapa (SLAM) + tracado SVM: estimador sem cones suficientes', vector, slow=True,
                                learned_width_m=estimate.learned_width_m)
