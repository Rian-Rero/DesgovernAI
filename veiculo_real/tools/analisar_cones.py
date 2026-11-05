"""Analisa logs/<ensaio>/cones_debug.jsonl gravado pelo main.py no modo cones.

Uso, no notebook, na pasta coletada pela GUI:
    python3 analisar_cones.py experimentos/verde/20261105_101500
    python3 analisar_cones.py <pasta> --frame 123        # registro completo de um quadro
    python3 analisar_cones.py <pasta> --plot 100 160     # vista de cima dos quadros 100..160

Lista episodios de perda (com o que o controle fez), trocas de lado,
bifurcacoes, saltos descartados e a sequencia de retas/curvas. Os JPGs de
eventos ficam em <pasta>/cone_frames/ com o numero do quadro no nome.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def load(folder):
    path = Path(folder)
    if path.is_dir():
        path = path / "cones_debug.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                print(f"linha {number} ilegivel (ensaio interrompido?)")
    return path.parent, records


def event_frames(folder):
    frames = {}
    for image in sorted((Path(folder) / "cone_frames").glob("f*.jpg")):
        try:
            frames.setdefault(int(image.name[1:7]), []).append(image.name)
        except ValueError:
            pass
    return frames


def runs(records, key):
    """Agrupa quadros consecutivos com o mesmo valor de key(record)."""
    output = []
    for record in records:
        value = key(record)
        if output and output[-1][0] == value:
            output[-1][2] = record
            output[-1][3] += 1
        else:
            output.append([value, record, record, 1])
    return output


def describe_loss(group, images):
    reasons = Counter(r.get("reason") or "?" for r in group)
    controls = [r.get("control") or {} for r in group]
    held = sum(bool(c.get("hold")) for c in controls)
    stopped = any(not c.get("active") for c in controls)
    factors = [c.get("hold_speed_factor") for c in controls if c.get("hold_speed_factor") is not None]
    first, last = group[0]["frame"], group[-1]["frame"]
    files = [name for frame in range(first - 5, last + 6) for name in images.get(frame, [])]
    text = (f"  quadros {first}-{last} ({len(group)}): hold em {held}"
            f"{', PAROU' if stopped else ', recuperou sem parar'}")
    if factors:
        text += f", vref x{min(factors):.2f} no minimo"
    text += "\n    motivos: " + "; ".join(f"{k} ({v})" for k, v in reasons.most_common(3))
    if files:
        text += f"\n    frames: {', '.join(files[:6])}{' ...' if len(files) > 6 else ''}"
    return text


def report(folder, records):
    images = event_frames(folder)
    total = len(records)
    valid = sum(bool(r.get("valid")) for r in records)
    print(f"Ensaio: {folder}")
    print(f"Quadros: {total} | corredor valido: {valid} ({100 * valid / max(total, 1):.0f}%)")
    summary = Path(folder) / "cones_summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        print(f"Resumo: fps={data.get('vision_fps')} deteccao media={data.get('mean_detection_s')} s"
              f" eventos={data.get('events')} descartes={data.get('dropped_records')}")

    print("\nPerdas do corredor:")
    losses = [g for g in runs(records, lambda r: bool(r.get("valid"))) if not g[0]]
    if not losses:
        print("  nenhuma")
    for _, first, last, _ in losses:
        group = [r for r in records if first["frame"] <= r["frame"] <= last["frame"]]
        print(describe_loss(group, images))

    print("\nEventos:")
    for record in records:
        for event in record.get("events") or []:
            vision = record.get("vision") or {}
            detail = ""
            if event == "troca_lado":
                sides = vision.get("sides") or {}
                detail = f"verde={'E' if sides.get('green_left') else 'D'} separacao={sides.get('separation')}"
            elif event == "bifurcacao":
                corridors = vision.get("corridors") or {}
                options = [f"#{i} custo={c.get('cost')} cont={c.get('continuity_m')}"
                           for i, c in enumerate(corridors.get("candidates") or []) if c.get("valid")]
                detail = f"escolhido #{corridors.get('chosen')} ({corridors.get('reason')}): " + ", ".join(options)
            elif event == "salto":
                detail = str((record.get("control") or {}).get("status"))
            print(f"  quadro {record['frame']}: {event} {detail}")

    print("\nTrechos (quadros validos):")
    segments = runs([r for r in records if r.get("valid")],
                    lambda r: ((r.get("vision") or {}).get("trajectory") or {}).get("segment"))
    for segment, first, last, count in segments:
        radii = [abs(((r.get("vision") or {}).get("trajectory") or {}).get("radius_m") or 0)
                 for r in records if first["frame"] <= r["frame"] <= last["frame"] and r.get("valid")]
        radii = [value for value in radii if value]
        radius = f" R~{sorted(radii)[len(radii) // 2]:.1f} m" if radii and segment != "reta" else ""
        print(f"  {segment or '?'}: quadros {first['frame']}-{last['frame']} ({count}){radius}")


def plot(records, first, last, output):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7))
    selected = [r for r in records if first <= r["frame"] <= last]
    for record in selected:
        vision = record.get("vision") or {}
        trajectory = vision.get("trajectory") or {}
        center = [p for p in trajectory.get("center_m") or [] if None not in p]
        if center:
            xs, ys = zip(*center)
            ax.plot(ys, xs, color="purple", alpha=.25)
        for candidate in (vision.get("corridors") or {}).get("candidates") or []:
            points = [p for p in candidate.get("center") or [] if None not in p]
            if points and candidate is not None:
                xs, ys = zip(*points)
                ax.plot(ys, xs, color="gray", alpha=.15, linewidth=.8)
        for color, mpl in (("green", "tab:green"), ("red", "tab:red")):
            points = [p for p in ((vision.get("detection") or {}).get(color) or {}).get("m") or [] if None not in p]
            if points:
                xs, ys = zip(*points)
                ax.scatter(ys, xs, s=8, color=mpl, alpha=.4)
    ax.invert_xaxis()  # esquerda (+y) a esquerda do grafico
    ax.set_xlabel("y [m] (esquerda +)")
    ax.set_ylabel("x [m] (frente)")
    ax.set_title(f"Quadros {first}-{last}: centro (roxo), candidatos (cinza)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    print(f"Figura salva em {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pasta", help="pasta do ensaio (ou o proprio cones_debug.jsonl)")
    parser.add_argument("--frame", type=int, help="imprime o registro completo deste quadro")
    parser.add_argument("--plot", type=int, nargs=2, metavar=("INICIO", "FIM"),
                        help="salva vista de cima dos quadros INICIO..FIM")
    args = parser.parse_args()
    folder, records = load(args.pasta)
    if not records:
        parser.exit(1, "Nenhum registro em cones_debug.jsonl\n")
    if args.frame is not None:
        match = [r for r in records if r.get("frame") == args.frame]
        print(json.dumps(match[0], indent=2, ensure_ascii=False) if match else "Quadro nao encontrado")
        return
    report(folder, records)
    if args.plot:
        plot(records, *args.plot, Path(folder) / f"cones_{args.plot[0]}_{args.plot[1]}.png")


if __name__ == "__main__":
    main()
