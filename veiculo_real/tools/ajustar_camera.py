#!/usr/bin/env python3
"""Descobre os valores de camera_settings (main.py) para a webcam do carro.

Rode NO RASPBERRY, com o carro parado na pista e os cones a 1-3 m:

    python3 veiculo_real/tools/ajustar_camera.py
    python3 veiculo_real/tools/ajustar_camera.py --exposures 40 80 120 160 200 --fourcc MJPG

Para cada exposicao manual (e para a automatica, como referencia) mede:
  fps real entregue pela camera, brilho medio do piso, % de pixels estourados
  (brancos) e a saturacao mediana dos pixels coloridos (cones);
e salva um JPG por ajuste em --out (padrao logs/ajuste_camera/<data_hora>, que o
Coletar Dados da GUI baixa) para comparar a olho. No fim imprime a
linha pronta para colar em main.py:

    "camera_settings": {"fourcc": "MJPG", "exposure": 120, "auto_wb": False},

Criterio da sugestao: fps >= 90% do pedido, menos de 2% estourado e, entre os
que passam, o de cones mais saturados. Exposicao alta demais derruba o fps e
borra em movimento; baixa demais escurece e apaga a cor dos cones na sombra.
A unidade de "exposure" depende da webcam (UVC comum: 100 us; 333 = 33 ms,
limite para 30 fps). Os controles disponiveis aparecem no inicio (v4l2-ctl).

Iluminacao: o detector leva o piso a cor de referencia (cone_settings
floor_b/g/r, auto_light). As colunas "piso BGR" e "corr. luz" mostram o piso
medido e a correcao que o detector aplicaria (1.00 = nenhuma; ganho no limite
3.5 = escuro demais, aumente a exposicao). So troque floor_* se reajustar os
limiares de cor (S/V/croma) sob esta luz: ai use o "piso BGR" daqui.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.camera import Camera, FRAME_RATE  # noqa: E402
from tools.cone_path import ConeSettings, floor_color, normalize_light  # noqa: E402


def measure(cam, frames, warmup):
    for _ in range(warmup):
        cam.get_image()
    started, images = time.monotonic(), []
    for _ in range(frames):
        image = cam.get_image()
        if image is not None:
            images.append(image)
    elapsed = time.monotonic() - started
    if not images:
        return None
    last = images[-1]
    hsv = cv2.cvtColor(last, cv2.COLOR_BGR2HSV)
    floor = hsv[last.shape[0]*2//3:, :, 2]
    colored = hsv[..., 1][(hsv[..., 1] > 60) & (hsv[..., 2] > 40)]
    _, light = normalize_light(last, ConeSettings())
    return {
        'floor_bgr': [int(round(v)) for v in floor_color(last)],
        'light': 1. if light is None else float(np.exp(np.mean(np.log(light)))),
        'fps': len(images)/elapsed if elapsed > 0 else 0.,
        'floor_v': float(np.mean(floor)),
        'blown_pct': 100.*float(np.mean(hsv[..., 2] >= 250)),
        'color_s': float(np.median(colored)) if colored.size > 200 else 0.,
        'image': last,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', type=int, default=None, help='indice da camera (padrao: primeira encontrada)')
    parser.add_argument('--exposures', type=float, nargs='+', default=[30, 60, 100, 150, 200, 300])
    parser.add_argument('--fourcc', default='MJPG', help='codec pedido (MJPG, YUYV); "" mantem o padrao')
    parser.add_argument('--wb', type=float, default=None,
                        help='temperatura de cor fixa (K), ex. 4500; sem ela so desliga o WB automatico')
    parser.add_argument('--frames', type=int, default=45)
    parser.add_argument('--out', type=Path, default=Path('logs')/'ajuste_camera'/time.strftime('%Y-%m-%d_%Hh%Mm%S'),
                        help='pasta das fotos (padrao em logs/: vem no Coletar Dados da GUI)')
    args = parser.parse_args(argv)

    if shutil.which('v4l2-ctl'):
        device = f'/dev/video{args.index or 0}'
        print(f'Controles de {device} (v4l2-ctl --list-ctrls):', flush=True)
        subprocess.run(['v4l2-ctl', '-d', device, '--list-ctrls'], check=False)
    else:
        print('v4l2-ctl ausente (sudo apt install v4l-utils): os limites dos controles nao serao listados.')

    args.out.mkdir(parents=True, exist_ok=True)
    cam = Camera(cam_index=args.index, fourcc=args.fourcc or None)
    rows = []
    try:
        auto = measure(cam, args.frames, warmup=30)  # exposicao automatica estabiliza
        if auto:
            cv2.imwrite(str(args.out / 'auto.jpg'), auto['image'])
            rows.append(('auto', auto))
        for exposure in args.exposures:
            cam._configure_image(exposure, False, args.wb)
            result = measure(cam, args.frames, warmup=10)
            if result:
                cv2.imwrite(str(args.out / f'exp_{exposure:g}.jpg'), result['image'])
                rows.append((exposure, result))
    finally:
        cam.close()

    print(f'\n{"exposicao":>10} {"fps":>6} {"piso V":>7} {"estouro%":>9} {"S cones":>8} {"piso BGR":>15} {"corr. luz":>9}')
    for exposure, r in rows:
        print(f'{str(exposure):>10} {r["fps"]:6.1f} {r["floor_v"]:7.0f} {r["blown_pct"]:9.2f} {r["color_s"]:8.0f}'
              f' {str(r["floor_bgr"]):>15} {r["light"]:9.2f}')
    manual = [(e, r) for e, r in rows if e != 'auto'
              and r['fps'] >= .9*FRAME_RATE and r['blown_pct'] < 2. and r['floor_v'] > 60]
    print(f'\nImagens em {args.out.resolve()}/ (compare os cones na sombra).')
    if not manual:
        print('Nenhum ajuste manual manteve fps e brilho: teste exposicoes menores/maiores ou mais luz.')
        return 1
    exposure, best = max(manual, key=lambda item: item[1]['color_s'])
    exposure = int(exposure) if float(exposure).is_integer() else exposure
    settings = {'fourcc': args.fourcc, 'exposure': exposure, 'auto_wb': False}
    if args.wb is not None:
        settings['wb_temperature'] = args.wb
    if not args.fourcc:
        settings.pop('fourcc')
    print('Sugestao para main.py (confira a imagem antes):')
    print(f'    "camera_settings": {settings!r},'.replace("'", '"'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
