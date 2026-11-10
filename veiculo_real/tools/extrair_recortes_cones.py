#!/usr/bin/env python3
"""Gera o conjunto de treino do classificador de cones a partir dos rotulos.

Uso, no notebook, depois de rotular com rotular_cones.py:
    python3 extrair_recortes_cones.py <ensaio> [<ensaio> ...] --out cones_dataset.npz
    python3 extrair_recortes_cones.py logs/a logs/b --out ds.npz --preview ds_amostra.jpg

Cada <ensaio> precisa de frames/fNNNNNN.jpg e labels.json; so quadros
"reviewed" entram. Para cada quadro:
  positivos  cada base rotulada vira um recorte da sua cor (cone_verde/cone_vermelho)
  negativos  cada candidato da cor (com os ajustes largos de cone_classifier.LOOSE_COLOR)
             a mais de --match-px de qualquer rotulo vira nao_cone
O recorte e o mesmo do carro (cone_classifier.extract_crop): quadrado alinhado
pela base, com o lado pela altura esperada do cone naquela linha (homografia).

Saida .npz: X (N, 32, 32, 3) uint8 BGR, y (N,) indice em CLASSES, group (N,)
quadro de origem (o treino separa validacao por quadro, sem vazamento) e meta
(JSON: classes, recorte, ajustes de cor dos candidatos, contagens).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__:
    from .cone_classifier import CLASSES, DEFAULT_CROP, LOOSE_COLOR, extract_crop
    from .cone_path import ConePathDetector, ConeSettings
    from .rotular_cones import DEFAULT_CALIBRATION, list_frames, load_calibration, load_labels
else:
    from cone_classifier import CLASSES, DEFAULT_CROP, LOOSE_COLOR, extract_crop
    from cone_path import ConePathDetector, ConeSettings
    from rotular_cones import DEFAULT_CALIBRATION, list_frames, load_calibration, load_labels


def _near(point, labels, radius):
    return bool(len(labels)) and float(np.min(np.linalg.norm(labels - point, axis=1))) <= radius


def extract_folder(folder, calibration, *, candidate_settings=None, crop=None, match_px=12.,
                   max_negatives=40, group_offset=0, seed=0):
    """(X, y, group, contagens) de uma pasta de ensaio rotulada."""
    rng = np.random.default_rng(seed)
    settings = ConeSettings(tracking=False, **(candidate_settings or LOOSE_COLOR))
    detector = ConePathDetector(calibration)
    labels = load_labels(folder)
    X, y, groups = [], [], []
    counts = Counter()
    cone_height = settings.cone_height_cm/100
    frames = [path for path in list_frames(folder) if labels.get(path.name, {}).get('reviewed')]
    for number, path in enumerate(frames):
        frame = cv2.imread(str(path))
        if frame is None:
            counts['ilegivel'] += 1
            continue
        if calibration is not None and frame.shape[1::-1] != tuple(calibration.image_size):
            raise ValueError(f'{path}: resolucao {frame.shape[1::-1]} difere da calibracao '
                             f'{tuple(calibration.image_size)}')
        found = detector.candidates(frame, settings)
        frame = found['frame']  # luz normalizada, como no carro
        scale = found['scale']
        if scale is None:
            raise ValueError('A extracao requer calibracao (escala do cone por linha).')
        entry = labels[path.name]
        truth = {color: np.asarray(entry[color], float).reshape(-1, 2) for color in ('green', 'red')}
        every_label = np.vstack([truth['green'], truth['red']])
        group = group_offset + number

        def add(point, label):
            px = float(scale[int(np.clip(round(point[1]), 0, len(scale)-1))])
            if not np.isfinite(px) or px <= 0:
                counts['sem_escala'] += 1
                return
            X.append(extract_crop(frame, point, px, cone_height, crop))
            y.append(CLASSES.index(label))
            groups.append(group)
            counts[label] += 1

        for color, label in (('green', 'cone_verde'), ('red', 'cone_vermelho')):
            for point in truth[color]:
                add(point, label)
        negatives = [point for color in ('green', 'red') for point in found[color]
                     if not _near(point, every_label, match_px)]
        if len(negatives) > max_negatives:
            negatives = [negatives[i] for i in rng.choice(len(negatives), max_negatives, replace=False)]
        for point in negatives:
            add(point, 'nao_cone')
        counts['quadros'] += 1
    size = int({**DEFAULT_CROP, **(crop or {})}['size'])
    return (np.asarray(X, np.uint8).reshape(-1, size, size, 3), np.asarray(y, np.int64),
            np.asarray(groups, np.int64), counts)


def preview(X, y, path, per_class=24):
    """Grade com amostras de cada classe (uma linha por classe), ampliada 3x."""
    rows = []
    for index, name in enumerate(CLASSES):
        samples = X[y == index][:per_class]
        tiles = list(samples) + [np.zeros_like(X[0])]*(per_class-len(samples))
        row = cv2.resize(np.hstack(tiles), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.putText(row, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(str(path), np.vstack(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pastas', nargs='+', type=Path, help='ensaios com frames/ e labels.json')
    parser.add_argument('--out', type=Path, required=True, help='arquivo .npz de saida')
    parser.add_argument('--calibration', default=str(DEFAULT_CALIBRATION))
    parser.add_argument('--match-px', type=float, default=12.,
                        help='candidato a ate esta distancia de um rotulo nao vira negativo')
    parser.add_argument('--max-negatives', type=int, default=40, help='negativos por quadro (sorteados)')
    parser.add_argument('--preview', type=Path, help='JPG com amostras de cada classe')
    args = parser.parse_args(argv)
    calibration = load_calibration(args.calibration)
    if calibration is None:
        parser.error('a extracao requer a calibracao da camera')
    parts, counts, offset = [], Counter(), 0
    for folder in args.pastas:
        X, y, group, folder_counts = extract_folder(folder, calibration, match_px=args.match_px,
                                                    max_negatives=args.max_negatives, group_offset=offset)
        print(f'{folder}: {dict(folder_counts)}', flush=True)
        if len(group):
            offset = int(group.max()) + 1
        parts.append((X, y, group))
        counts.update(folder_counts)
    if not parts or not sum(len(p[1]) for p in parts):
        print('Nenhum recorte: rotule e revise quadros com rotular_cones.py antes.', file=sys.stderr)
        return 1
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    group = np.concatenate([p[2] for p in parts])
    meta = {'classes': list(CLASSES), 'crop': DEFAULT_CROP, 'candidate_settings': LOOSE_COLOR,
            'match_px': args.match_px, 'sources': [str(p) for p in args.pastas], 'counts': dict(counts)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, group=group, meta=json.dumps(meta))
    print(f'{len(y)} recortes em {args.out}: ' +
          ', '.join(f'{name}={int(np.sum(y == i))}' for i, name in enumerate(CLASSES)))
    if args.preview:
        preview(X, y, args.preview)
        print(f'Amostras em {args.preview}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
