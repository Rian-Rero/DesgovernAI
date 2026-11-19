#!/usr/bin/env python3
"""Vista de cima (bird's-eye view) de um ensaio: o que o Stanley seguiu.

O carro ja segue a trajetoria no plano do CHAO: cada cone e projetado no piso
pela homografia da calibracao (pixel -> metros), o estimador do corredor ajusta
bordas e centro em metros, e o Stanley recebe o deslocamento e o rumo desse
centro junto ao eixo dianteiro. Esta ferramenta desenha isso por quadro:

    fundo     a imagem da camera "achatada" no chao (warpPerspective pela
              homografia): cada pixel no lugar em metros, como vista de cima
    pontos    cones aceitos (verde/vermelho), em metros
    linhas    bordas e centro estimados (clotoide do estimador)
    seta      do eixo dianteiro ao ponto do centro mais proximo: o erro lateral
              e do Stanley; o rumo e a diferenca de direcao ali

Usa os quadros gravados (frames/, "Gravar quadros" na GUI) e o cones_debug.jsonl.

    python3 tools/vista_de_cima.py logs/20261117_205131            # pasta vista_de_cima/
    python3 tools/vista_de_cima.py logs/20261117_205131 --video     # + vista_de_cima.mp4
"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fva_car.corridor import ARC, sample_centerline  # noqa: E402
from fva_car.lateral import GroundCalibration  # noqa: E402

X_RANGE = (-.45, 3.2)   # m a frente do eixo dianteiro (o carro fica embaixo)
Y_RANGE = (-1.3, 1.3)   # m, esquerda +
SCALE = 170             # px por metro
CAR_HALF_WIDTH, CAR_LENGTH = .13, .45
GREEN, RED, YELLOW, WHITE = (60, 200, 60), (60, 60, 230), (0, 220, 255), (255, 255, 255)


def ground_to_view():
    """Matriz 3x3 chao (x frente, y esquerda) -> pixel da vista de cima."""
    return np.array([[0., -SCALE, Y_RANGE[1]*SCALE], [-SCALE, 0., X_RANGE[1]*SCALE], [0., 0., 1.]])


def view_size():
    return int(round((Y_RANGE[1] - Y_RANGE[0])*SCALE)), int(round((X_RANGE[1] - X_RANGE[0])*SCALE))


def to_view(points):
    points = np.asarray(points, float).reshape(-1, 2)
    q = np.c_[points, np.ones(len(points))] @ ground_to_view().T
    return q[:, :2]/q[:, 2:3]


def warp(frame, calibration):
    """Imagem da camera no chao (vista de cima). calibration: GroundCalibration."""
    size = calibration.image_size
    if frame.shape[1::-1] != tuple(size):
        frame = cv2.resize(frame, tuple(size))
    matrix = ground_to_view() @ np.asarray(calibration.homography, float)
    return cv2.warpPerspective(frame, matrix, view_size(), flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40))


def _poly(image, points, color, thickness=2):
    pixels = np.round(to_view(points)).astype(np.int32)
    if len(pixels) >= 2:
        cv2.polylines(image, [pixels], False, color, thickness, cv2.LINE_AA)


def draw(view, record):
    """Cones, corredor estimado e o erro do Stanley sobre a vista de cima."""
    vision = record.get('vision') or {}
    detection = vision.get('detection') or {}
    corridor = vision.get('corridor') or {}
    width, height = view.shape[1], view.shape[0]
    # Grade de 0,5 m.
    for x in np.arange(0., X_RANGE[1] + 1e-9, .5):
        (a, b) = to_view([[x, Y_RANGE[0]], [x, Y_RANGE[1]]])
        cv2.line(view, tuple(np.int32(a)), tuple(np.int32(b)), (90, 90, 90), 1)
        cv2.putText(view, f'{x:.1f}', (2, int(a[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, .35, (170, 170, 170), 1)
    # Carro (eixo dianteiro em x = 0).
    body = to_view([[0., CAR_HALF_WIDTH], [0., -CAR_HALF_WIDTH], [-CAR_LENGTH, -CAR_HALF_WIDTH],
                    [-CAR_LENGTH, CAR_HALF_WIDTH]])
    cv2.polylines(view, [np.int32(body)], True, WHITE, 2)
    errors, dkappa, road_width = corridor.get('errors'), corridor.get('dkappa'), corridor.get('width_m')
    if errors and dkappa is not None and road_width:
        state = np.array([*errors[:3], dkappa, road_width], float)
        arc = ARC[(ARC >= -.4) & (ARC <= 3.4)]
        _poly(view, sample_centerline(state, arc), YELLOW, 2)
        _poly(view, sample_centerline(state, arc, road_width/2), (200, 200, 200), 1)
        _poly(view, sample_centerline(state, arc, -road_width/2), (200, 200, 200), 1)
        closest = sample_centerline(state, [0.])[0]
        a, b = to_view([[0., 0.], closest])
        cv2.arrowedLine(view, tuple(np.int32(a)), tuple(np.int32(b)), (255, 0, 255), 2, tipLength=.3)
    green_left = (corridor.get('sides') or {}).get('green_left')
    for color, bgr in (('green', GREEN), ('red', RED)):
        for x, y in (detection.get(color) or {}).get('m') or []:
            u, v = to_view([[x, y]])[0]
            cv2.circle(view, (int(u), int(v)), 6, bgr, -1, cv2.LINE_AA)
            cv2.circle(view, (int(u), int(v)), 6, WHITE, 1, cv2.LINE_AA)
    control = record.get('control') or {}
    steer = control.get('steer_cmd_rad', control.get('steer_rad', 0.)) or 0.
    lines = [f"quadro {record.get('frame')}  {corridor.get('state', '-')}",
             (f"e {100*errors[0]:+.0f} cm  rumo {math.degrees(errors[1]):+.0f} graus" if errors else 'sem corredor'),
             (f"largura {road_width:.2f} m" if road_width else ''),
             f"v {control.get('v', 0):.2f}  vref {control.get('vref', 0):.2f} m/s",
             f"esterco {math.degrees(steer):+.0f} graus",
             f"verde a {'esquerda' if green_left else 'direita' if green_left is not None else '?'}"]
    cv2.rectangle(view, (0, 0), (215, 8 + 17*len(lines)), (0, 0, 0), -1)
    for k, text in enumerate(lines):
        cv2.putText(view, text, (6, 18 + 17*k), cv2.FONT_HERSHEY_SIMPLEX, .45, WHITE, 1, cv2.LINE_AA)
    return view


def render(frame, calibration, record):
    """Painel: camera (esquerda) e vista de cima com o corredor (direita)."""
    view = draw(warp(frame, calibration), record)
    camera = cv2.resize(frame, (int(frame.shape[1]*view.shape[0]/frame.shape[0]), view.shape[0]))
    return np.hstack([camera, view])


def records_by_frame(folder):
    path = Path(folder)/'cones_debug.jsonl'
    with path.open(encoding='utf-8') as handle:
        return {r['frame']: r for r in (json.loads(line) for line in handle if line.strip())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ensaio', type=Path)
    parser.add_argument('--calibration', default=str(ROOT/'calibracao'/'camera_lateral.json'),
                        help='a calibracao USADA no ensaio')
    parser.add_argument('--video', action='store_true', help='tambem grava vista_de_cima.mp4')
    parser.add_argument('--fps', type=float, default=3.)
    args = parser.parse_args(argv)
    calibration = GroundCalibration.from_file(args.calibration)
    records = records_by_frame(args.ensaio)
    frames = sorted((args.ensaio/'frames').glob('f*.jpg'))
    if not frames:
        parser.error(f'{args.ensaio}: sem frames/ (grave com "Gravar quadros" na GUI)')
    out = args.ensaio/'vista_de_cima'
    out.mkdir(exist_ok=True)
    writer = None
    written = 0
    for path in frames:
        record = records.get(int(path.stem[1:]))
        frame = cv2.imread(str(path))
        if record is None or frame is None:
            continue
        panel = render(frame, calibration, record)
        cv2.imwrite(str(out/path.name), panel)
        if args.video:
            if writer is None:
                writer = cv2.VideoWriter(str(args.ensaio/'vista_de_cima.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.fps, panel.shape[1::-1])
            writer.write(panel)
        written += 1
    if writer is not None:
        writer.release()
    print(f'{written} quadros em {out}' + (f' e {args.ensaio/"vista_de_cima.mp4"}' if args.video else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
