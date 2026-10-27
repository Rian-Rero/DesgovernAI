#!/usr/bin/env python3
"""Compara larguras de processamento em uma foto, sem abrir camera ou motores."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import GroundCalibration, RealLineDetector
from main import parameters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path, help='vision_camera.jpg coletada do ensaio')
    parser.add_argument('--calibration', type=Path, default=Path(parameters['camera_calibration_file']))
    parser.add_argument('--widths', type=int, nargs='+', default=[0, 480, 320],
                        help='0 usa a resolucao calibrada; padrao: 0 480 320')
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error('--iterations deve ser positivo')
    cv2.setNumThreads(2)
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f'Nao foi possivel ler {args.image}')
    try:
        calibration = GroundCalibration.from_file(args.calibration)
        if frame.shape[1::-1] != calibration.image_size:
            raise ValueError('A imagem deve ter a resolucao original da calibracao.')
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
