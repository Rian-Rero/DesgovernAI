#!/usr/bin/env python3
"""Mede o detector de cones contra os rotulos de rotular_cones.py.

Uso, no notebook:
    python3 avaliar_cones.py <pasta_do_ensaio>
    python3 avaliar_cones.py <pasta_do_ensaio> --no-tracking --match-px 15
    python3 avaliar_cones.py <pasta_do_ensaio> --save-worst 10
    python3 avaliar_cones.py --check-calibration

Roda ConePathDetector sobre <pasta>/frames/f*.jpg em ordem (0,2 s entre
quadros) e compara as bases com <pasta>/labels.json (so quadros "reviewed").

Metricas:
    precisao/revocacao por cor   bases casadas a <= --match-px (guloso, menor distancia primeiro)
    FP/quadro                    bases detectadas sem rotulo, por quadro revisado
    revocacao por faixa de y     linhas da imagem: quanto mais alto, mais distante o cone
    quadros validos              fracao de quadros com corredor (result.valid)
    tremor                       desvio-padrao da variacao de lateral_error_m entre
                                 quadros validos consecutivos
    ligacoes erradas             quadros validos cuja borda passa a <= --match-px de
                                 um falso positivo (o corredor usou um cone falso)

Grava <pasta>/avaliacao.json; --save-worst N grava os N piores quadros em
<pasta>/avaliacao_piores/ (detector desenhado + rotulos em circulos brancos).
"""

from collections import Counter
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__:
    from .cone_path import ConePathDetector, ConeSettings, render_cones
    from .rotular_cones import (DEFAULT_CALIBRATION, list_frames, load_calibration, load_labels,
                                make_parser)
else:
    from cone_path import ConePathDetector, ConeSettings, render_cones
    from rotular_cones import (DEFAULT_CALIBRATION, list_frames, load_calibration, load_labels,
                               make_parser)


FRAME_PERIOD_S = .2
RESULT_FILE = "avaliacao.json"
WORST_DIR = "avaliacao_piores"
MIN_CALIBRATED_RANGE_M = 1.5


def _r(value, digits=4):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _ratio(num, den):
    return num / den if den else None


def match_points(detected, labeled, max_px):
    """Casamento guloso por distancia: pares (i_det, j_lab) com d <= max_px.

    Retorna (pares, detectados sem par, rotulos sem par), indices.
    """
    detected = np.asarray(detected, float).reshape(-1, 2)
    labeled = np.asarray(labeled, float).reshape(-1, 2)
    if not len(detected) or not len(labeled):
        return [], list(range(len(detected))), list(range(len(labeled)))
    distance = np.linalg.norm(detected[:, None, :] - labeled[None, :, :], axis=2)
    candidates = sorted((distance[i, j], i, j) for i, j in zip(*np.nonzero(distance <= max_px)))
    used_det, used_lab, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_det or j in used_lab:
            continue
        used_det.add(i)
        used_lab.add(j)
        pairs.append((int(i), int(j)))
    return (pairs, [i for i in range(len(detected)) if i not in used_det],
            [j for j in range(len(labeled)) if j not in used_lab])


def polyline_distance(points, polyline):
    """Menor distancia de cada ponto a uma polilinha (px); inf se vazia."""
    points = np.asarray(points, float).reshape(-1, 2)
    polyline = np.asarray(polyline, float).reshape(-1, 2)
    if not len(points):
        return np.empty(0)
    if not len(polyline):
        return np.full(len(points), np.inf)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    length2 = np.maximum(np.sum(ab**2, axis=1), 1e-12)
    t = np.clip(np.einsum("pk,sk->ps", points, ab) / length2 - np.sum(a*ab, axis=1)/length2, 0, 1)
    closest = a[None] + t[..., None]*ab[None]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def _row_bands(height, count):
    edges = np.linspace(0, height, count + 1).round().astype(int)
    return list(zip(edges[:-1], edges[1:]))


def evaluate(folder, *, calibration=None, settings=None, match_px=12., bands=6, save_worst=0):
    """Avalia a pasta e devolve o dicionario gravado em avaliacao.json."""
    folder = Path(folder)
    frames = list_frames(folder)
    if not frames:
        raise ValueError(f"Nenhum quadro em {folder}/frames/f*.jpg")
    labels = load_labels(folder)
    reviewed = {name for name, entry in labels.items() if entry.get("reviewed")}
    settings = settings or ConeSettings()
    detector = ConePathDetector(calibration)

    totals = {color: {"tp": 0, "fp": 0, "fn": 0} for color in ("green", "red")}
    band_counts = None
    per_frame = {}
    valid_count = 0
    invalid_status = Counter()  # por que o corredor falhou
    lateral = []         # (indice do quadro, lateral_error_m) dos quadros validos
    lateral_px = []
    worst = []           # (erros, ligacao_errada, nome, imagem, resultado, rotulo)
    height = None

    for index, path in enumerate(frames):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Aviso: nao foi possivel ler {path}", file=sys.stderr)
            continue
        result = detector.process(frame, settings, timestamp=index*FRAME_PERIOD_S)
        if height is None:
            height = frame.shape[0]
            band_counts = [[lo, hi, 0, 0] for lo, hi in _row_bands(height, bands)]
        if not result.valid:
            invalid_status[result.status] += 1
        if result.valid:
            valid_count += 1
            if result.lateral_error_m is not None and math.isfinite(result.lateral_error_m):
                lateral.append((index, float(result.lateral_error_m)))
            if result.lateral_error_px is not None and math.isfinite(result.lateral_error_px):
                lateral_px.append((index, float(result.lateral_error_px)))
        if path.name not in reviewed:
            continue
        entry = labels[path.name]
        stats = {"valid": bool(result.valid), "status": result.status}
        false_positives = []
        for color, detected in (("green", result.green_points), ("red", result.red_points)):
            truth = np.asarray(entry[color], float).reshape(-1, 2)
            detected = np.asarray(detected, float).reshape(-1, 2)
            pairs, fp, fn = match_points(detected, truth, match_px)
            totals[color]["tp"] += len(pairs)
            totals[color]["fp"] += len(fp)
            totals[color]["fn"] += len(fn)
            stats[color] = {"tp": len(pairs), "fp": len(fp), "fn": len(fn)}
            false_positives.extend(detected[fp].tolist())
            matched = {j for _, j in pairs}
            for j, (_, y) in enumerate(truth):
                for band in band_counts:
                    if band[0] <= y < band[1] or (band[1] == height and y >= height):
                        band[2] += 1
                        band[3] += j in matched
                        break
        wrong = False
        if result.valid and false_positives:
            boundary = min(polyline_distance(false_positives, result.left_boundary).min(),
                           polyline_distance(false_positives, result.right_boundary).min())
            wrong = bool(boundary <= match_px)
        stats["wrong_link"] = wrong
        stats["lateral_error_m"] = _r(result.lateral_error_m)
        per_frame[path.name] = stats
        errors = sum(stats[c]["fp"] + stats[c]["fn"] for c in ("green", "red"))
        if save_worst > 0 and (errors or wrong):
            worst.append((errors, wrong, path.name, frame, result, entry))
            worst.sort(key=lambda item: (-item[0], not item[1], item[2]))
            del worst[save_worst:]

    missing = sorted(reviewed - {path.name for path in frames})
    detection = {}
    for color, counts in list(totals.items()) + [("total", {k: totals["green"][k] + totals["red"][k]
                                                            for k in ("tp", "fp", "fn")})]:
        detection[color] = dict(counts,
                                precision=_r(_ratio(counts["tp"], counts["tp"] + counts["fp"])),
                                recall=_r(_ratio(counts["tp"], counts["tp"] + counts["fn"])))
    reviewed_frames = len(per_frame)
    wrong_frames = sorted(name for name, stats in per_frame.items() if stats["wrong_link"])

    def jitter(series):
        diffs = [b - a for (i, a), (j, b) in zip(series, series[1:]) if j == i + 1]
        return (_r(float(np.std(diffs)), 5), len(diffs)) if len(diffs) >= 2 else (None, len(diffs))

    tremor_m, tremor_pairs = jitter(lateral)
    tremor_px, _ = jitter(lateral_px)
    report = {
        "folder": str(folder),
        "config": {"tracking": bool(settings.tracking), "match_px": float(match_px),
                   "frame_period_s": FRAME_PERIOD_S, "row_bands": bands,
                   "calibration": calibration is not None},
        "frames_total": len(frames),
        "frames_reviewed": reviewed_frames,
        "labels_without_frame": missing,
        "detection": detection,
        "false_positives_per_frame": _r(_ratio(detection["total"]["fp"], reviewed_frames)),
        "recall_by_row": [{"y_min": int(lo), "y_max": int(hi), "labels": n, "matched": int(m),
                           "recall": _r(_ratio(m, n))} for lo, hi, n, m in (band_counts or [])],
        "frames_valid": valid_count,
        "valid_fraction": _r(_ratio(valid_count, len(frames))),
        "valid_fraction_reviewed": _r(_ratio(sum(s["valid"] for s in per_frame.values()),
                                             reviewed_frames)),
        "invalid_status": dict(invalid_status.most_common()),
        "tremor_m": tremor_m,
        "tremor_px": tremor_px,
        "tremor_pairs": tremor_pairs,
        "wrong_links": len(wrong_frames),
        "wrong_link_frames": wrong_frames,
        "frames": per_frame,
    }
    if save_worst > 0:
        report["worst_frames"] = _save_worst(folder, worst, settings)
    (folder / RESULT_FILE).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def _save_worst(folder, worst, settings):
    directory = folder / WORST_DIR
    directory.mkdir(exist_ok=True)
    for old in directory.glob("*.jpg"):
        old.unlink()  # pasta gerada por esta ferramenta: evita misturar avaliacoes
    names = []
    for rank, (errors, wrong, name, frame, result, entry) in enumerate(worst, 1):
        image = render_cones(frame, result, settings)
        for color in ("green", "red"):
            for x, y in entry[color]:
                cv2.circle(image, (int(x), int(y)), 9, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{name}  erros={errors}" + ("  LIGACAO ERRADA" if wrong else "")
        cv2.putText(image, text, (10, image.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (10, image.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        out = directory / f"{rank:02d}_{name}"
        cv2.imwrite(str(out), image)
        names.append(out.name)
    return names


def _fmt(value, pattern="{:.3f}"):
    return "-" if value is None else pattern.format(value)


def print_report(report):
    print(f"Ensaio: {report['folder']}")
    config = report["config"]
    print(f"Quadros: {report['frames_total']} | revisados: {report['frames_reviewed']} | "
          f"tracking={'sim' if config['tracking'] else 'nao'} | casamento <= {config['match_px']:g} px | "
          f"calibracao={'sim' if config['calibration'] else 'nao'}")
    if report["labels_without_frame"]:
        print(f"Aviso: {len(report['labels_without_frame'])} rotulos sem quadro em frames/")
    print()
    print(f"{'cor':<8}{'VP':>6}{'FP':>6}{'FN':>6}{'precisao':>10}{'revocacao':>11}")
    for color, label in (("green", "verde"), ("red", "vermelho"), ("total", "total")):
        d = report["detection"][color]
        print(f"{label:<8}{d['tp']:>6}{d['fp']:>6}{d['fn']:>6}"
              f"{_fmt(d['precision']):>10}{_fmt(d['recall']):>11}")
    print()
    print("Revocacao por faixa de linha (y menor = mais distante):")
    for band in report["recall_by_row"]:
        print(f"  y {band['y_min']:>3}-{band['y_max']:<3}  rotulos {band['labels']:>4}  "
              f"revocacao {_fmt(band['recall'])}")
    print()
    print(f"Falsos positivos por quadro:  {_fmt(report['false_positives_per_frame'], '{:.2f}')}")
    print(f"Quadros validos:              {report['frames_valid']}/{report['frames_total']} "
          f"({_fmt(report['valid_fraction'], '{:.1%}')}); revisados "
          f"{_fmt(report['valid_fraction_reviewed'], '{:.1%}')}")
    for status, count in list(report["invalid_status"].items())[:3]:
        print(f"  invalido {count:>4}x: {status}")
    print(f"Tremor lateral_error_m:       {_fmt(report['tremor_m'], '{:.4f} m')} "
          f"({report['tremor_pairs']} pares consecutivos)")
    if report["tremor_px"] is not None:
        print(f"Tremor lateral_error_px:      {_fmt(report['tremor_px'], '{:.2f} px')}")
    print(f"Ligacoes erradas:             {report['wrong_links']} quadros")
    if report.get("worst_frames"):
        print(f"Piores quadros em {WORST_DIR}/: {', '.join(report['worst_frames'])}")


def check_calibration(path, out=None):
    """Erro de reprojecao (m) por ponto, ordenado pela distancia x, e alcance da calibracao."""
    out = out or sys.stdout
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    calibration = load_calibration(path)
    if calibration is None:
        raise FileNotFoundError(f"Calibracao nao encontrada: {path}")
    if "image_points_px" not in data or "ground_points_m" not in data:
        print(f"{path}: so contem a homografia; sem pares pixel/piso para conferir.", file=out)
        return {"points": [], "max_distance_m": None, "max_error_m": None, "warning": None}
    pixels = np.asarray(data["image_points_px"], float)
    ground = np.asarray(data["ground_points_m"], float)
    homogeneous = np.column_stack((pixels, np.ones(len(pixels)))) @ np.asarray(calibration.homography).T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    errors = np.linalg.norm(projected - ground, axis=1)
    order = np.argsort(ground[:, 0], kind="stable")
    points = [{"pixel": pixels[i].tolist(), "ground_m": ground[i].tolist(),
               "projected_m": [_r(v) for v in projected[i]], "error_m": _r(errors[i])} for i in order]
    print(f"Calibracao: {path}", file=out)
    print(f"{'pixel (u, v)':>16}  {'piso x, y (m)':>16}  {'reprojetado (m)':>18}  {'erro (m)':>9}", file=out)
    for p in points:
        u, v = p["pixel"]
        x, y = p["ground_m"]
        px, py = p["projected_m"]
        print(f"{u:>7.1f}, {v:>6.1f}  {x:>7.3f}, {y:>6.3f}  {px:>8.3f}, {py:>7.3f}  {p['error_m']:>9.4f}",
              file=out)
    max_distance = float(ground[:, 0].max())
    max_error = float(errors.max())
    print(f"Erro medio {errors.mean():.4f} m | maximo {max_error:.4f} m | "
          f"distancia calibrada {ground[:, 0].min():.2f}-{max_distance:.2f} m", file=out)
    warning = None
    if max_distance < MIN_CALIBRATED_RANGE_M:
        warning = (f"AVISO: a maior distancia calibrada ({max_distance:.2f} m) e menor que "
                   f"{MIN_CALIBRATED_RANGE_M:.1f} m; cones alem disso tem posicao extrapolada. "
                   "Recalibre com pontos ate ~2 m.")
        print(warning, file=out)
    return {"points": points, "max_distance_m": _r(max_distance), "max_error_m": _r(max_error),
            "warning": warning}


def main(argv=None):
    parser = make_parser(__doc__)
    parser.add_argument("pasta", nargs="?", help="pasta do ensaio com frames/ e labels.json")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), metavar="ARQUIVO",
                        help="JSON de calibracao do piso (padrao: veiculo_real/calibracao/"
                             "camera_lateral.json; '' para nenhuma)")
    parser.add_argument("--no-tracking", action="store_true",
                        help="desliga o rastreamento entre quadros (padrao: ligado)")
    parser.add_argument("--match-px", type=float, default=12., metavar="PX",
                        help="distancia maxima entre base detectada e rotulo (padrao: %(default)s px)")
    parser.add_argument("--faixas", type=int, default=6, metavar="N",
                        help="faixas de linha da imagem na revocacao por distancia (padrao: %(default)s)")
    parser.add_argument("--save-worst", type=int, default=0, metavar="N",
                        help="grava os N quadros com mais erros em avaliacao_piores/")
    parser.add_argument("--check-calibration", action="store_true",
                        help="confere a reprojecao dos pontos da calibracao e o alcance em metros")
    args = parser.parse_args(argv)
    if args.pasta is None and not args.check_calibration:
        parser.error("informe a pasta do ensaio ou use --check-calibration")
    if args.check_calibration:
        if not args.calibration:
            parser.error("--check-calibration requer --calibration")
        check_calibration(args.calibration)
        if args.pasta is None:
            return 0
        print()
    report = evaluate(args.pasta, calibration=load_calibration(args.calibration),
                      settings=ConeSettings(tracking=not args.no_tracking),
                      match_px=args.match_px, bands=max(1, args.faixas), save_worst=max(0, args.save_worst))
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
