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


def load_runs(paths):
    """Ensaio de cada recorte (para validar em ensaios inteiros fora do treino).

    Conjuntos sem a chave 'run' (antigos) caem no quadro de cada recorte."""
    runs, offset = [], 0
    for path in paths:
        data = np.load(path, allow_pickle=False)
        values = data['run'] if 'run' in data else data['group']
        runs.append(values + offset)
        offset = int(runs[-1].max()) + 1 if len(runs[-1]) else offset
    return np.concatenate(runs)


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


def tensor(torch, images):
    """N x H x W x 3 uint8 BGR -> N x 3 x H x W float 0-1 (igual a to_blob)."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.)


def fit(torch, X, y, train, val, epochs=40, batch=128, lr=3e-3, verbose=True):
    """Treina com a validacao escolhendo a melhor epoca; (modelo, acuracia
    balanceada, probabilidades na validacao)."""
    Xt, yt = tensor(torch, X[train]), torch.from_numpy(y[train])
    Xv, yv = tensor(torch, X[val]), y[val]
    model = build_model(torch, len(CLASSES))
    weights = torch.tensor(len(y[train])/np.maximum(np.bincount(y[train], minlength=len(CLASSES)), 1),
                           dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights/weights.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs*max(1, int(np.ceil(len(yt)/batch))))
    best, best_state = -1., None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(yt))
        started, total = time.time(), 0.
        for start in range(0, len(order), batch):
            index = order[start:start+batch]
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
        if verbose:
            print(f'epoca {epoch+1:3d}  perda {total/len(yt):.4f}  acuracia balanceada {balanced:.3f}  '
                  f'({time.time()-started:.1f} s)', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(Xv), 1).numpy()
    return model, best, probabilities


def cross_validate(torch, X, y, runs, folds, epochs, batch, lr, seed=0):
    """Validacao por ensaio: cada ensaio fica fora do treino em uma das `folds`
    rodadas. Devolve (matriz 3x3 somada, previsao de cada recorte).

    A melhor epoca e escolhida num quadro de validacao separado do treino, nunca
    no ensaio de teste (senao a medida fica otimista)."""
    unique = np.unique(runs)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    assignment = {run: k % folds for k, run in enumerate(order)}
    fold_of = np.array([assignment[r] for r in runs])
    predicted = np.full(len(y), -1)
    for fold in range(folds):
        test = fold_of == fold
        rest = np.flatnonzero(~test)
        inner = rng.random(len(rest)) < .15
        train, val = np.zeros(len(y), bool), np.zeros(len(y), bool)
        train[rest[~inner]], val[rest[inner]] = True, True
        model, _, _ = fit(torch, X, y, train, val, epochs, batch, lr, verbose=False)
        with torch.no_grad():
            predicted[test] = model(tensor(torch, X[test])).argmax(1).numpy()
        held = sorted({int(r) for r in runs[test]})
        hits = float(np.mean(predicted[test] == y[test])) if test.any() else float('nan')
        print(f'rodada {fold+1}/{folds}: ensaios fora do treino {held}, {int(test.sum())} recortes,'
              f' acerto {hits:.3f}', flush=True)
    return confusion(y, predicted, len(CLASSES)), predicted


def plot_matrix(matrix, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    share = matrix/np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    names = ['não cone', 'cone verde', 'cone vermelho']
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{int(matrix[i, j])}\n{100*share[i, j]:.0f}%', ha='center', va='center',
                    color='white' if share[i, j] > .55 else 'black')
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('previsto pela rede')
    ax.set_ylabel('rótulo humano')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


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
    parser.add_argument('--validacao-por-ensaio', type=int, metavar='K', default=0,
                        help='antes do treino final: validacao cruzada em K rodadas, cada ensaio fora do treino'
                             ' em uma delas (matriz de confusao honesta)')
    parser.add_argument('--matriz', type=Path, help='PNG da matriz de confusao (validacao por ensaio)')
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
    cross = None
    if args.validacao_por_ensaio:
        runs = load_runs(args.datasets)
        folds = min(args.validacao_por_ensaio, len(np.unique(runs)))
        print(f'\nValidacao por ensaio: {len(np.unique(runs))} ensaios em {folds} rodadas', flush=True)
        cross, _ = cross_validate(torch, X, y, runs, folds, args.epochs, args.batch, args.lr, args.seed)
        text, _, _ = report(cross)
        print('\nMatriz de confusao, ensaios fora do treino (todas as rodadas somadas):\n' + text + '\n', flush=True)
        if args.matriz:
            balanced = float(np.mean(np.diag(cross)/np.maximum(cross.sum(1), 1)))
            print('figura:', plot_matrix(cross, args.matriz, f'Rede de cones, ensaios fora do treino ({folds} rodadas)'
                                         f'\nacurácia balanceada {balanced:.3f}'), flush=True)

    model, best, probabilities = fit(torch, X, y, train, val, args.epochs, args.batch, args.lr)
    print(f'parametros: {sum(p.numel() for p in model.parameters())}', flush=True)
    yv = y[val]
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
        'cone_height_m': float(meta.get('cone_height_m', .2)),
        'min_confidence': args.min_confidence,
        'trained': {'datasets': [str(p) for p in args.datasets], 'samples': int(len(y)),
                    'validation_frames': int(len(np.unique(group[val]))),
                    'balanced_accuracy': round(best, 4),
                    'cross_validation_by_run': None if cross is None else cross.tolist(),
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
