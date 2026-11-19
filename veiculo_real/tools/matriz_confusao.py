#!/usr/bin/env python3
"""Matriz de confusao do detector de cones contra os rotulos humanos (sim/nao).

A verdade vem da rotulagem (rotular_sim_nao.py): cada candidato da cor larga,
em cada quadro, marcado "e cone" ou "nao e cone" por uma pessoa. Aqui o
detector roda de novo nesses quadros, como no carro (imagem reduzida a
--largura px, regras de cor, forma e tamanho, e opcionalmente a rede), e cada
item rotulado e "aceito" se o detector entregou um cone da mesma cor a ate
--raio px da base dele.

                       detector aceitou   detector descartou
    e cone (humano)    verdadeiro +       falso -  (cone perdido)
    nao e cone         falso +            verdadeiro -

Precisao = VP/(VP+FP): dos cones que o carro usa, quantos sao cones.
Revocacao = VP/(VP+FN): dos cones que existem, quantos o carro ve.

Mede so o que foi rotulado: um cone que nem a cor larga pegou nao entra.
A rede deve ser avaliada em ensaios fora do treino (treinar_classificador_cones.py
--validacao-por-ensaio); aqui, com --modelo, ela confirma os candidatos.

    python3 tools/matriz_confusao.py ~/fva_experimentos/rotulagem \\
        --variante "antes" cone_height_cm=20.0 cone_base_cm=15.0 SIZE_FOOT_MAX=2.2 SIZE_HEIGHT_MAX=2.5 \\
        --variante "agora" \\
        --png calibracao/matriz_detector.png
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_classifier import load_classifier  # noqa: E402
from cone_path import ConePathDetector, ConeSettings  # noqa: E402
from rotular_cones import load_calibration  # noqa: E402
import rotular_sim_nao as labeling  # noqa: E402

COLORS = ('green', 'red')


def ground_truth(out, with_calibration=False):
    """{(pasta, quadro): [(cor, base x, base y, e_cone)]} so dos itens rotulados.

    with_calibration: tambem {(pasta, quadro): caminho da calibracao do item ou None}."""
    items, labels = labeling.load(out)
    frames, calibrations = defaultdict(list), {}
    for item in items:
        if item['id'] in labels:
            key = (item['pasta'], item['quadro'])
            frames[key].append((item['cor'], float(item['base'][0]), float(item['base'][1]),
                                bool(labels[item['id']])))
            calibrations[key] = item.get('calibracao')
    return (frames, calibrations) if with_calibration else frames


def frame_path(folder, name):
    path = Path(folder)/'frames'/name
    return path if path.exists() else Path(folder)/name


def detect(detector, frame, settings, width, classifier=None):
    """Bases aceitas por cor, em px da imagem cheia (como no carro: imagem reduzida)."""
    height = round(frame.shape[0]*width/frame.shape[1])
    small = frame if frame.shape[1] == width else cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    scale = frame.shape[1]/small.shape[1]
    found = detector.candidates(small, settings)
    points = {color: np.asarray(found[color], float).reshape(-1, 2) for color in COLORS}
    if classifier is not None and found['scale'] is not None:
        verdict = classifier.classify(found['frame'], points, found['scale'])
        points = {color: points[color][verdict[color][0]] for color in COLORS}
    return {color: (p + .5)*scale - .5 for color, p in points.items()}


def confusion(frames, calibration, settings, width=320, classifier=None, radius=12., calibrations=None):
    """Contagens {'vp','fn','fp','vn','extras'} no total e por cor.

    calibrations: {(pasta, quadro): caminho} da calibracao de cada quadro (a camera
    muda de posicao entre dias); sem ela, `calibration` vale para todos."""
    detectors = {}

    def detector_for(key):
        path = (calibrations or {}).get(key)
        if path not in detectors:
            detectors[path] = ConePathDetector(load_calibration(path) if path else calibration, classifier=None)
        return detectors[path]

    counts = {key: defaultdict(int) for key in ('total', *COLORS)}
    for (folder, name), truth in frames.items():
        detector = detector_for((folder, name))
        frame = cv2.imread(str(frame_path(folder, name)))
        if frame is None:
            continue
        found = detect(detector, frame, settings, width, classifier)
        used = {color: np.zeros(len(found[color]), bool) for color in COLORS}
        for color, x, y, is_cone in truth:
            points = found[color]
            near = np.linalg.norm(points - (x, y), axis=1) <= radius if len(points) else np.zeros(0, bool)
            accepted = bool(np.any(near))
            used[color] |= near
            key = ('vp' if accepted else 'fn') if is_cone else ('fp' if accepted else 'vn')
            for group in ('total', color):
                counts[group][key] += 1
        for color in COLORS:
            extras = int(np.count_nonzero(~used[color]))
            counts['total']['extras'] += extras
            counts[color]['extras'] += extras
    return {group: dict(values) for group, values in counts.items()}


def metrics(c):
    vp, fn, fp, vn = (c.get(k, 0) for k in ('vp', 'fn', 'fp', 'vn'))
    precision = vp/(vp + fp) if vp + fp else float('nan')
    recall = vp/(vp + fn) if vp + fn else float('nan')
    specificity = vn/(vn + fp) if vn + fp else float('nan')
    return {'precisao': precision, 'revocacao': recall, 'especificidade': specificity,
            'f1': 2*precision*recall/(precision + recall) if precision + recall else float('nan')}


def text_report(name, counts):
    c = counts['total']
    m = metrics(c)
    lines = [f'== {name}',
             f'                     aceito   descartado',
             f'  e cone (humano)   {c.get("vp", 0):7d}   {c.get("fn", 0):9d}',
             f'  nao e cone        {c.get("fp", 0):7d}   {c.get("vn", 0):9d}',
             f'  precisao {m["precisao"]:.3f} | revocacao {m["revocacao"]:.3f} | especificidade '
             f'{m["especificidade"]:.3f} | F1 {m["f1"]:.3f} | cones entregues fora dos rotulados: {c.get("extras", 0)}']
    for color in COLORS:
        mc = metrics(counts[color])
        lines.append(f'  {color:5s}: precisao {mc["precisao"]:.3f} revocacao {mc["revocacao"]:.3f}'
                     f' (VP {counts[color].get("vp", 0)}, FN {counts[color].get("fn", 0)},'
                     f' FP {counts[color].get("fp", 0)}, VN {counts[color].get("vn", 0)})')
    return '\n'.join(lines)


def plot(results, path, title='Detector de cones x rotulo humano'):
    """Uma matriz por variante (contagem e % da linha)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(4.2*len(results), 4.), squeeze=False)
    for ax, (name, counts) in zip(axes[0], results.items()):
        c = counts['total']
        matrix = np.array([[c.get('vp', 0), c.get('fn', 0)], [c.get('fp', 0), c.get('vn', 0)]], float)
        share = matrix/np.maximum(matrix.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{int(matrix[i, j])}\n{100*share[i, j]:.0f}%', ha='center', va='center',
                        color='white' if share[i, j] > .55 else 'black', fontsize=11)
        ax.set_xticks([0, 1], ['aceito', 'descartado'])
        ax.set_yticks([0, 1], ['é cone', 'não é cone'])
        m = metrics(c)
        ax.set_title(f'{name}\nprecisão {m["precisao"]:.2f} · revocação {m["revocacao"]:.2f}', fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def parse_changes(values):
    changes = {}
    for item in values:
        key, _, value = item.partition('=')
        changes[key] = float(value) if '.' in value else int(value) if value.lstrip('-').isdigit() else value
    return changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rotulagem', type=Path, help='pasta da rotulagem (rotular_sim_nao.py)')
    parser.add_argument('--calibration', help='calibracao da camera dos quadros rotulados sem a sua'
                                               ' (itens antigos; padrao: a da fila)')
    parser.add_argument('--largura', type=int, default=320, help='largura do processamento (vision_width)')
    parser.add_argument('--modelo', help='rede .onnx confirmando os candidatos (todas as variantes)')
    parser.add_argument('--variante', nargs='+', action='append', metavar=('NOME', 'AJUSTE=VALOR'),
                        help='nome e ajustes de ConeSettings; sem nenhuma: o detector atual')
    parser.add_argument('--png', type=Path, help='figura com as matrizes')
    parser.add_argument('--json', type=Path, help='contagens em JSON')
    args = parser.parse_args(argv)
    meta = json.loads((args.rotulagem/labeling.QUEUE).read_text(encoding='utf-8')).get('meta', {})
    calibration = load_calibration(args.calibration or meta.get('calibracao'))
    if calibration is None:
        parser.error('calibracao da camera nao encontrada (--calibration)')
    frames, calibrations = ground_truth(args.rotulagem, with_calibration=True)
    classifier = load_classifier(args.modelo) if args.modelo else None
    candidate = classifier.candidate_settings if classifier is not None else {}
    variants = args.variante or [['detector atual']]
    results = {}
    import cone_path
    for name, *changes in variants:
        changes = parse_changes(changes)
        # Chaves em MAIUSCULAS sao constantes do detector (tolerancias de tamanho).
        constants = {key: value for key, value in changes.items() if key.isupper()}
        saved = {key: getattr(cone_path, key) for key in constants}
        for key, value in constants.items():
            setattr(cone_path, key, value)
        try:
            settings = ConeSettings(tracking=False, **{**candidate, **{k: v for k, v in changes.items()
                                                                       if not k.isupper()}})
            results[name] = confusion(frames, calibration, settings, args.largura, classifier,
                                      calibrations=calibrations)
        finally:
            for key, value in saved.items():
                setattr(cone_path, key, value)
        print(text_report(name, results[name]), flush=True)
    print(f'{sum(len(v) for v in frames.values())} itens rotulados em {len(frames)} quadros')
    if args.png:
        print('figura:', plot(results, args.png))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
