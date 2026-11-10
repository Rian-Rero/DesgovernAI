#!/usr/bin/env python3
"""Compara larguras de processamento em uma foto, sem abrir camera ou motores.

--cones mede o detector de cones (modo padrao do carro). Rode no Raspberry:
se o tempo aqui for bem menor que o detection_s do log, a diferenca e disputa
de CPU/GIL com as outras threads.

Sem --image usa a vision_camera.jpg mais recente em logs/ (gravada pelas
missoes); sem nenhuma, tira uma foto com a camera e guarda em logs/benchmark/.

    python3 tools/benchmark_visao.py --cones
    python3 tools/benchmark_visao.py --cones --image logs/<ensaio>/vision_camera.jpg
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import RealConeDetector
from fva_car.lateral import GroundCalibration, RealLineDetector
from main import corridor_settings, parameters


def default_image(logs=Path('logs')):
    """vision_camera.jpg mais recente das missoes; sem nenhuma, uma foto da camera."""
    shots = sorted(logs.glob('*/vision_camera.jpg'), key=lambda path: path.stat().st_mtime)
    if shots:
        return shots[-1]
    from fva_car.camera import Camera

    camera = Camera()
    try:
        frame = None
        for _ in range(30):  # exposicao automatica estabiliza
            frame = camera.get_image()
    finally:
        camera.close()
    if frame is None:
        raise SystemExit('Sem vision_camera.jpg em logs/ e a camera nao entregou imagem.')
    path = logs / 'benchmark' / time.strftime('%Y-%m-%d_%Hh%Mm%S.jpg')
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path,
                        help='foto 640x480 (padrao: vision_camera.jpg mais recente em logs/, ou uma foto da camera)')
    parser.add_argument('--calibration', type=Path, default=Path(parameters['camera_calibration_file']))
    parser.add_argument('--widths', type=int, nargs='+', default=[0, 480, 320],
                        help='0 usa a resolucao calibrada; padrao: 0 480 320')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--cones', action='store_true', help='detector de cones em vez da linha amarela')
    parser.add_argument('--classifier', help='modelo .onnx do classificador de cones (com --cones)')
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error('--iterations deve ser positivo')
    cv2.setNumThreads(2)
    if args.image is None:
        args.image = default_image()
    print(f'Imagem: {args.image}', flush=True)
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f'Nao foi possivel ler {args.image}')
    try:
        calibration = GroundCalibration.from_file(args.calibration)
        if frame.shape[1::-1] != calibration.image_size:
            raise ValueError('A imagem deve ter a resolucao original da calibracao.')
        if args.cones:
            detectors = [RealConeDetector(calibration, settings=parameters['cone_settings'],
                                          processing_width=width,
                                          classifier=args.classifier or parameters.get('cone_classifier'),
                                          corridor=corridor_settings(parameters))
                         for width in args.widths]
        else:
            detectors = [RealLineDetector(calibration, path_settings=parameters['path_settings'],
                                         anchor=parameters['path_anchor'], render_preview=False,
                                         processing_width=width) for width in args.widths]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Tempo de processamento da foto; nao inclui captura, motores ou demais threads.', flush=True)
    for detector in detectors:
        for _ in range(2):
            detector.observe_frame(frame, time.monotonic())
        durations, detection, geometry = [], [], []
        valid = 0
        for _ in range(args.iterations):
            started = time.monotonic()
            observation = detector.observe_frame(frame, started)
            durations.append(time.monotonic() - started)
            detection.append(observation.detection_seconds)
            geometry.append(observation.geometry_seconds)
            valid += int(observation.valid)
        report = {
            'processing_size': detector.processing_size,
            'median_ms': round(float(np.median(durations)) * 1000, 2),
            'p95_ms': round(float(np.percentile(durations, 95)) * 1000, 2),
            'processing_hz': round(len(durations) / sum(durations), 2),
            'detection_median_ms': round(float(np.median(detection)) * 1000, 2),
            'geometry_median_ms': (round(float(np.median(geometry)) * 1000, 2)
                                   if np.all(np.isfinite(geometry)) else None),
            'valid_frames': valid, 'total_frames': args.iterations,
            'last_reason': observation.reason,
            'last_errors': observation.errors,
        }
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
