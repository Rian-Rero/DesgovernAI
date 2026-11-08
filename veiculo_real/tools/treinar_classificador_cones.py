#!/usr/bin/env python3
"""Treina o classificador de cones (CNN pequena) e exporta ONNX para o carro.

Roda NO NOTEBOOK (precisa de PyTorch; o carro so precisa do OpenCV):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx
    python3 treinar_classificador_cones.py cones_dataset.npz --out ../modelos/cones.onnx
    python3 treinar_classificador_cones.py a.npz b.npz --out ../modelos/cones.onnx --epochs 60

Entrada: .npz de extrair_recortes_cones.py (pode juntar varios ensaios).
Validacao separada POR QUADRO (recortes do mesmo quadro nunca ficam dos dois
lados). Aumento de dados: brilho, contraste, ganho por canal (balanco de
branco), ruido, desfoque, espelho e deslocamento de 2 px.

Saida: <out>.onnx e <out>.json (classes, recorte, ajustes de cor dos
candidatos, confianca minima e metricas). No carro:
    "cone_classifier": "veiculo_real/modelos/cones.onnx"   (main.py)
ou  python3 veiculo_real/main.py --classifier veiculo_real/modelos/cones.onnx

No fim o modelo exportado e recarregado com cv2.dnn (o mesmo caminho do carro)
e comparado com o PyTorch: a exportacao so e aceita se as saidas baterem.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_classifier import CLASSES, ConeClassifier  # noqa: E402


def load_datasets(paths):
    X, y, group, meta, offset = [], [], [], None, 0
    for path in paths:
        data = np.load(path, allow_pickle=False)
        current = json.loads(str(data['meta']))
        if meta is None:
            meta = current
        elif (current['crop'] != meta['crop'] or current['classes'] != meta['classes']
              or current['candidate_settings'] != meta['candidate_settings']):
            raise ValueError(f'{path}: recorte/classes/cores diferentes dos outros conjuntos')
        X.append(data['X'])
        y.append(data['y'])
        group.append(data['group'] + offset)
        offset = int(group[-1].max()) + 1 if len(group[-1]) else offset
    if tuple(meta['classes']) != CLASSES:
        raise ValueError(f'Classes do conjunto {meta["classes"]} diferem de {CLASSES}')
    return np.concatenate(X), np.concatenate(y), np.concatenate(group), meta


def split_by_group(group, fraction, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(group)
    validation = set(rng.choice(unique, max(1, int(round(fraction*len(unique)))), replace=False).tolist())
    mask = np.array([g in validation for g in group])
    return ~mask, mask


def build_model(torch, classes):
    nn = torch.nn

    def block(a, b):
        return [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]

    # ~24 mil parametros: ~0,1-0,3 ms por recorte 32x32 no Raspberry Pi 4 (cv2.dnn).
    return nn.Sequential(*block(3, 16), nn.MaxPool2d(2), *block(16, 32), nn.MaxPool2d(2),
                         *block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2),
                         nn.Linear(64, classes))


def augment(torch, batch):
    """batch: N x 3 x H x W em 0-1 (BGR)."""
    n = batch.shape[0]
    device = batch.device
    gain = 1 + .35*(torch.rand(n, 1, 1, 1, device=device)-.5)          # brilho
    channel = 1 + .16*(torch.rand(n, 3, 1, 1, device=device)-.5)       # balanco de branco
    mean = batch.mean(dim=(1, 2, 3), keepdim=True)
    contrast = 1 + .4*(torch.rand(n, 1, 1, 1, device=device)-.5)
    batch = ((batch-mean)*contrast + mean)*gain*channel
    blur = torch.rand(n, device=device) < .25
    if blur.any():
        smooth = torch.nn.functional.avg_pool2d(batch[blur], 3, stride=1, padding=1, count_include_pad=False)
        batch[blur] = smooth
    flip = torch.rand(n, device=device) < .5
    batch[flip] = batch[flip].flip(-1)
    padded = torch.nn.functional.pad(batch, (2, 2, 2, 2), mode='replicate')
    dx, dy = np.random.randint(0, 5, 2)
    batch = padded[:, :, dy:dy+batch.shape[2], dx:dx+batch.shape[3]]
    batch = batch + .02*torch.randn_like(batch)
    return batch.clamp(0, 1)


def confusion(truth, predicted, classes):
    matrix = np.zeros((classes, classes), int)
    for t, p in zip(truth, predicted):
        matrix[t, p] += 1
    return matrix


def report(matrix):
    lines = ['real \\ previsto  ' + ' '.join(f'{name[:13]:>13}' for name in CLASSES)]
    for i, name in enumerate(CLASSES):
        lines.append(f'{name:>16} ' + ' '.join(f'{v:13d}' for v in matrix[i]))
    recall = np.diag(matrix)/np.maximum(matrix.sum(1), 1)
    precision = np.diag(matrix)/np.maximum(matrix.sum(0), 1)
    for i, name in enumerate(CLASSES):
        lines.append(f'{name:>16}  precisao {precision[i]:.3f}  revocacao {recall[i]:.3f}')
    return '\n'.join(lines), precision, recall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datasets', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='caminho do .onnx (o .json vai ao lado)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--val', type=float, default=.2, help='fracao de quadros para validacao')
    parser.add_argument('--min-confidence', type=float, default=.6,
                        help='probabilidade minima da classe de cone no carro')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        import torch
    except ImportError:
        parser.error('instale o PyTorch no notebook: pip install torch --index-url '
                     'https://download.pytorch.org/whl/cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    X, y, group, meta = load_datasets(args.datasets)
    train, val = split_by_group(group, args.val, args.seed)
    counts = np.bincount(y, minlength=len(CLASSES))
    print(f'{len(y)} recortes ({", ".join(f"{n}={c}" for n, c in zip(CLASSES, counts))}); '
          f'treino {train.sum()}, validacao {val.sum()} ({len(np.unique(group[val]))} quadros)', flush=True)
    if min(counts) == 0:
        parser.error('todas as classes precisam de exemplos; rotule mais quadros')

    def tensor(images):  # N x H x W x 3 uint8 BGR -> N x 3 x H x W float 0-1 (igual a to_blob)
        return torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.)

    Xt, yt = tensor(X[train]), torch.from_numpy(y[train])
    Xv, yv = tensor(X[val]), y[val]
    model = build_model(torch, len(CLASSES))
    print(f'parametros: {sum(p.numel() for p in model.parameters())}', flush=True)
    weights = torch.tensor(len(y[train])/np.maximum(np.bincount(y[train], minlength=len(CLASSES)), 1),
                           dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights/weights.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs*max(1, int(np.ceil(len(yt)/args.batch))))
    best, best_state = -1., None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(yt))
        started, total = time.time(), 0.
        for start in range(0, len(order), args.batch):
            index = order[start:start+args.batch]
            optimizer.zero_grad()
            loss = loss_fn(model(augment(torch, Xt[index].clone())), yt[index])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()*len(index)
        model.eval()
        with torch.no_grad():
            predicted = model(Xv).argmax(1).numpy()
        matrix = confusion(yv, predicted, len(CLASSES))
        balanced = float(np.mean(np.diag(matrix)/np.maximum(matrix.sum(1), 1)))
        if balanced > best:
            best, best_state = balanced, {k: v.clone() for k, v in model.state_dict().items()}
        print(f'epoca {epoch+1:3d}  perda {total/len(yt):.4f}  acuracia balanceada {balanced:.3f}  '
              f'({time.time()-started:.1f} s)', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xv)
    probabilities = torch.softmax(logits, 1).numpy()
    text, precision, recall = report(confusion(yv, probabilities.argmax(1), len(CLASSES)))
    print('\nMelhor epoca na validacao:\n' + text)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    size = int(meta['crop']['size'])
    dummy = torch.zeros(1, 3, size, size)
    export = dict(input_names=['input'], output_names=['logits'], opset_version=11,
                  dynamic_axes={'input': {0: 'n'}, 'logits': {0: 'n'}})
    # opset 11 e o exportador classico: abrem tambem no OpenCV 4.5+ do Raspberry.
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    try:
        torch.onnx.export(model, dummy, str(args.out), dynamo=False, **export)
    except TypeError:  # PyTorch antigo, sem o argumento dynamo
        torch.onnx.export(model, dummy, str(args.out), **export)
    info = {
        'classes': list(CLASSES), 'crop': meta['crop'], 'candidate_settings': meta['candidate_settings'],
        'min_confidence': args.min_confidence,
        'trained': {'datasets': [str(p) for p in args.datasets], 'samples': int(len(y)),
                    'validation_frames': int(len(np.unique(group[val]))),
                    'balanced_accuracy': round(best, 4),
                    'precision': dict(zip(CLASSES, np.round(precision, 4).tolist())),
                    'recall': dict(zip(CLASSES, np.round(recall, 4).tolist())),
                    'date': time.strftime('%Y-%m-%d %H:%M')}}
    args.out.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')

    # Mesmo caminho do carro: cv2.dnn + to_blob, comparado ao PyTorch.
    runtime = ConeClassifier(args.out)
    check = X[val][:256]
    reference = probabilities[:len(check)]
    deployed = runtime.predict(list(check))
    agreement = float(np.mean(deployed.argmax(1) == reference.argmax(1)))
    difference = float(np.max(np.abs(deployed - reference))) if len(check) else 0.
    print(f'\ncv2.dnn x PyTorch: concordancia {agreement:.3f}, maior diferenca de probabilidade {difference:.2e}')
    if agreement < .999 or difference > 1e-3:
        print('ERRO: o modelo exportado nao reproduz o PyTorch no OpenCV.', file=sys.stderr)
        return 1
    started = time.perf_counter()
    for _ in range(20):
        runtime.predict(list(check[:20]))
    per_crop = (time.perf_counter()-started)/(20*min(20, len(check)))*1000
    print(f'Modelo: {args.out} (+ .json); {per_crop:.3f} ms por recorte neste computador')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
