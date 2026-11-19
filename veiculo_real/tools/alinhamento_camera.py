#!/usr/bin/env python3
"""Alinhamento da calibracao com o carro, medido pelos proprios ensaios.

A calibracao liga pixels ao chao no referencial da TRENA usada para medir os
pontos. Se o carro nao estava alinhado com a trena (ou a origem nao estava no
centro do eixo dianteiro), tudo o que o carro ve fica girado/deslocado em
relacao a ele, e nao ha como perceber na hora de calibrar.

Os ensaios mostram isso sem recalibrar: um cone parado, com o carro andando,
se desloca para tras exatamente ao longo do eixo do carro. Comparando a posicao
de cada cone em quadros seguidos com o movimento da odometria:

    residuo lateral = d*sen(giro) + (x_cone + L)*d*curva_nao_vista

    d                 avanco do carro entre os quadros (odometria, eixo dianteiro)
    giro              rotacao da percepcao em relacao ao carro (+ = esquerda)
    curva_nao_vista   guinada real que a odometria nao viu, rad por metro (vies
                      da direcao: a roda vira diferente do comando)

O giro nao depende da distancia do cone; a guinada sim (braco x_cone + L): os
dois se separam por minimos quadrados. Nos ensaios de 17/11: giro -11,3 graus
(cones ate 1,4 m: -13,5; de 1,4 a 2,2 m: -13,1), curva nao vista -0,17 rad/m.

    python3 tools/alinhamento_camera.py logs/2026*/ --calibration calibracao/camera_lateral.json
    python3 tools/calibrar_camera_lateral.py --revisar calibracao/camera_lateral.json --girar 11.3

Precisa do cones_debug.jsonl (opcao "Debug de cones" ligada) e de odometria.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fva_car.lateral import GroundCalibration  # noqa: E402

WHEELBASE = .36   # m: da pose (eixo traseiro) ao eixo dianteiro, origem dos cones
MATCH_M = .2


def cone_pairs(calibration, records, scale=2., x_range=(.6, 2.2), wheelbase=WHEELBASE):
    """(avanco d, x do cone, residuo lateral) de cada cone visto em dois quadros seguidos."""
    rows, previous = [], None
    for record in records:
        vision = record.get('vision') or {}
        detection = vision.get('detection') or {}
        tracking = vision.get('tracking') or {}
        motion = tracking.get('motion_m') if tracking.get('motion') == 'odometria' else None
        current = {}
        for color in ('green', 'red'):
            pixels = np.asarray((detection.get(color) or {}).get('px') or [], float).reshape(-1, 2)
            points = (np.asarray(calibration.project_pixels((pixels + .5)*scale - .5), float).reshape(-1, 2)
                      if len(pixels) else np.empty((0, 2)))
            keep = np.all(np.isfinite(points), axis=1) & (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1])
            current[color] = points[keep]
        if previous is not None and motion is not None and motion[0] > .03:
            dx, dy, dth = (float(v) for v in motion)
            dx += wheelbase*(math.cos(dth) - 1)
            dy += wheelbase*math.sin(dth)
            c, s = math.cos(dth), math.sin(dth)
            for color in ('green', 'red'):
                before = previous[color]
                for x, y in current[color]:
                    if not len(before):
                        continue
                    expected = np.array([c*x - s*y + dx, s*x + c*y + dy])
                    distance = np.linalg.norm(before - expected, axis=1)
                    k = int(np.argmin(distance))
                    if distance[k] < MATCH_M:
                        rows.append((dx, x, before[k][1] - expected[1]))
        previous = current
    return np.asarray(rows, float).reshape(-1, 3)


def fit(rows, wheelbase=WHEELBASE):
    """(giro em graus, curva nao vista em rad/m)."""
    d, x, residual = rows.T
    design = np.c_[d, (x + wheelbase)*d]
    (sine, turn), *_ = np.linalg.lstsq(design, residual, rcond=None)
    return math.degrees(math.asin(float(np.clip(sine, -1, 1)))), float(turn)


def load_records(folder):
    path = Path(folder)/'cones_debug.jsonl'
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ensaios', nargs='+', type=Path)
    parser.add_argument('--calibration', default=str(ROOT/'calibracao'/'camera_lateral.json'),
                        help='a calibracao USADA nos ensaios (os pixels sao reprojetados por ela)')
    args = parser.parse_args(argv)
    calibration = GroundCalibration.from_file(args.calibration)
    everything, per_run = [], []
    for folder in args.ensaios:
        if not (folder/'cones_debug.jsonl').exists():
            print(f'{folder.name}: sem cones_debug.jsonl (ligue "Debug de cones")')
            continue
        rows = cone_pairs(calibration, load_records(folder))
        if len(rows) < 40:
            print(f'{folder.name}: so {len(rows)} pares de cones, pouco para medir')
            continue
        angle, turn = fit(rows)
        per_run.append(angle)
        everything.append(rows)
        print(f'{folder.name}: {len(rows):4d} pares | giro {angle:+5.1f} graus | curva nao vista {turn:+.3f} rad/m')
    if not everything:
        return 1
    rows = np.vstack(everything)
    angle, turn = fit(rows)
    steer = math.degrees(math.atan(WHEELBASE*turn))
    print(f'\nTODOS ({len(rows)} pares): giro {angle:+.1f} graus (mediana por ensaio {np.median(per_run):+.1f}),'
          f' curva nao vista {turn:+.3f} rad/m (roda {abs(steer):.1f} graus mais para a'
          f' {"esquerda" if steer > 0 else "direita"} que o comando)')
    for name, part in (('cones ate 1,4 m', rows[rows[:, 1] < 1.4]), ('cones de 1,4 a 2,2 m', rows[rows[:, 1] >= 1.4])):
        if len(part) >= 40:
            print(f'  {name}: giro {fit(part)[0]:+.1f} graus')
    if abs(angle) > 2:
        print(f'\nA percepcao esta girada {abs(angle):.1f} graus para a {"esquerda" if angle > 0 else "direita"}'
              f' em relacao ao carro. Para corrigir a calibracao usada:\n'
              f'  python3 tools/calibrar_camera_lateral.py --revisar {args.calibration} --girar {-angle:.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
