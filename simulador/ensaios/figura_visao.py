# -*- coding: utf-8 -*-
"""Figura da cadeia de visao num unico quadro (sem memoria), em uma curva.

Posiciona o carro (cena parada; nada e salvo), assenta a suspensao por 1 s,
captura a imagem e mostra: (a) imagem com centros detectados, (b) mascara HSV,
(c) pontos projetados no chao, ajuste local e erros e_y / e_psi.
Uso: venv/bin/python simulador/ensaios/figura_visao.py --saida pasta [--port 23000]
"""
import argparse
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import rgb_to_hsv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car import Car  # noqa: E402
from fva_car.lateral import fit_line_errors, project_ground, yellow_centers  # noqa: E402
from controle_lateral import place_car  # noqa: E402
from main import parameters  # noqa: E402

POSE = (-1.32, -3.79, 150.2)  # curva a direita, |kappa| ~ 0.58 1/m (log do Stanley)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saida', type=Path, required=True)
    p.add_argument('--port', type=int, default=23000)
    p.add_argument('--pose', type=float, nargs=3, default=POSE)
    a = p.parse_args()
    car = Car(dict(parameters, save=False, beep=False, logfile=str(a.saida / '_log'), port=a.port,
                   explicit_vision=True))
    original = place_car(car.sim, car.robot, a.pose)
    try:
        car.start_mission()
        for _ in range(20):
            car.step()
        image = car.get_image()
        sim = car.sim
        lo, hi = parameters['yellow_hsv_lower'], parameters['yellow_hsv_upper']
        pixels = yellow_centers(image, lo, hi)
        angle = sim.getObjectFloatParam(car.cam, sim.visionfloatparam_perspective_angle)
        points = project_ground(pixels, image.shape, sim.getObjectMatrix(car.cam, -1),
                                angle, parameters['line_ground_z'])
        matrix = np.asarray(sim.getObjectMatrix(car.robot, -1)).reshape(3, 4)
        left = np.asarray(sim.getObjectPosition(car.steerL, car.robot))
        right = np.asarray(sim.getObjectPosition(car.steerR, car.robot))
        front = matrix[:, :3] @ ((left + right) / 2) + matrix[:, 3]
        forward = matrix[:, 2]
        e_y, e_psi, kappa = fit_line_errors(points, front, forward, with_curvature=True)
    finally:
        car.close()
        sim.setObjectMatrix(car.robot, original, -1)

    hsv = rgb_to_hsv(image / 255.)
    hsv[..., 0] *= 360
    mask = np.all((hsv >= lo) & (hsv <= hi), axis=-1)
    f = forward[:2] / np.linalg.norm(forward[:2])
    lft = np.array([-f[1], f[0]])
    rel = points[:, :2] - front[:2]
    x, y = rel @ f, rel @ lft

    a.saida.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 3.3))
    g = fig.add_gridspec(1, 3, width_ratios=(1.5, 1.5, 1))
    ax = fig.add_subplot(g[0])
    ax.imshow(image)
    ax.plot(pixels[:, 0], pixels[:, 1], 'c.', ms=2)
    ax.set(title='(a) Câmera e centros detectados', xticks=[], yticks=[])
    ax = fig.add_subplot(g[1])
    ax.imshow(mask, cmap='gray')
    ax.set(title='(b) Máscara HSV', xticks=[], yticks=[])
    ax = fig.add_subplot(g[2])
    ax.plot(-y, x, 'c.', ms=3, label='Pontos projetados')
    xs = np.linspace(-.2, max(1.5, x.max()), 100)
    if abs(kappa) > 1e-6:
        # arco tangente no ponto mais proximo, apenas ilustrativo
        R = 1 / kappa
        s = np.linspace(-.3, 1.6, 100)
        px = -e_y * math.sin(e_psi) + (np.sin(s / R) * R) * math.cos(e_psi) - (R - R * np.cos(s / R)) * math.sin(e_psi)
        py = e_y * math.cos(e_psi) + (np.sin(s / R) * R) * math.sin(e_psi) + (R - R * np.cos(s / R)) * math.cos(e_psi)
        ax.plot(-py, px, 'k-', lw=1, label='Ajuste local')
    ax.plot([0], [0], 'rs', ms=6, label='Eixo dianteiro')
    ax.set(title='(c) Chão, referencial do carro', xlabel='direita (m)', ylabel='frente (m)',
           xlim=(-1.2, 1.2), ylim=(-.3, 2.4))
    ax.set_aspect('equal')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc='upper left')
    ax.text(.02, .02, f'$e_y$ = {100*e_y:.1f} cm\n$e_\\psi$ = {math.degrees(e_psi):.1f}°\n'
            f'$\\kappa$ = {kappa:.2f} m$^{{-1}}$', transform=ax.transAxes, fontsize=8,
            va='bottom', bbox=dict(fc='w', ec='0.7'))
    fig.tight_layout()
    fig.savefig(a.saida / 'visao_pipeline.png', dpi=200, bbox_inches='tight')
    print(f'e_y={e_y:.3f} m, e_psi={math.degrees(e_psi):.1f} deg, kappa={kappa:.3f}')


if __name__ == '__main__':
    main()
