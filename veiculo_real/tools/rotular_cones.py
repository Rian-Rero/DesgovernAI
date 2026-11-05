#!/usr/bin/env python3
"""Rotula as bases dos cones verdes e vermelhos em quadros gravados de um ensaio.

Uso, no notebook:
    python3 rotular_cones.py <pasta_do_ensaio>
    python3 rotular_cones.py <pasta_do_ensaio> --calibration ../calibracao/camera_lateral.json

A pasta precisa conter frames/fNNNNNN.jpg (JPEG BGR 640x480). Cada quadro abre
com as bases achadas pelo detector atual como pre-rotulo; corrija e avance.

Mouse:
    clique esquerdo        adiciona base verde
    clique direito         adiciona base vermelha (Ctrl+esquerdo tambem)
    clique sobre um circulo   remove a base (raio de 8 px)
Teclado:
    n / espaco   proximo quadro (marca o atual como revisado)
    b            quadro anterior (marca o atual como revisado)
    s            marca o atual como revisado e salva labels.json
    c            apaga todas as bases do quadro
    r            refaz o pre-rotulo do detector
    q / Esc      salva e sai (o quadro atual so conta se ja estava revisado)

Saida: <pasta>/labels.json
    {"fNNNNNN.jpg": {"green": [[x, y], ...], "red": [[x, y], ...], "reviewed": true}}
Rotulos ja existentes sao preservados; so quadros "reviewed" entram na avaliacao
(avaliar_cones.py).
"""

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__:
    from .cone_path import ConePathDetector, ConeSettings
else:
    from cone_path import ConePathDetector, ConeSettings


LABELS_FILE = "labels.json"
FRAMES_DIR = "frames"
REMOVE_RADIUS_PX = 8
COLORS = ("green", "red")
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "calibracao" / "camera_lateral.json"
DRAW = {"green": (50, 220, 50), "red": (50, 50, 240)}  # BGR


class PortugueseHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        return super().add_usage(usage, actions, groups, prefix="uso: " if prefix is None else prefix)


def make_parser(description):
    """ArgumentParser com cabecalhos da ajuda em portugues."""
    parser = argparse.ArgumentParser(description=description, add_help=False,
                                     formatter_class=PortugueseHelpFormatter)
    parser._positionals.title = "argumentos"
    parser._optionals.title = "opcoes"
    parser.add_argument("-h", "--help", action="help", help="mostra esta ajuda e sai")
    return parser


# ---------------------------------------------------------------- dados

def list_frames(folder):
    """Quadros frames/f*.jpg em ordem de nome (= ordem de captura)."""
    return sorted((Path(folder) / FRAMES_DIR).glob("f*.jpg"))


def _clean_points(points):
    clean = []
    for point in ([] if points is None else points):
        x, y = (int(round(float(v))) for v in point[:2])
        clean.append([x, y])
    return clean


def load_labels(folder):
    """labels.json da pasta; {} se ainda nao existir."""
    path = Path(folder) / LABELS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Nao foi possivel ler {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} deve conter um objeto JSON por nome de quadro.")
    labels = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        labels[name] = {"green": _clean_points(entry.get("green")),
                        "red": _clean_points(entry.get("red")),
                        "reviewed": bool(entry.get("reviewed", False))}
    return labels


def save_labels(folder, labels):
    """Grava labels.json de forma atomica (temporario + rename), em ordem de nome."""
    path = Path(folder) / LABELS_FILE
    ordered = {name: {"green": _clean_points(labels[name].get("green")),
                      "red": _clean_points(labels[name].get("red")),
                      "reviewed": bool(labels[name].get("reviewed", False))}
               for name in sorted(labels)}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def empty_entry():
    return {"green": [], "red": [], "reviewed": False}


def prelabel(frame, detector=None, settings=None):
    """Bases do detector atual (sem rastreamento) como rotulo inicial."""
    detector = detector or ConePathDetector()
    settings = settings or ConeSettings(tracking=False)
    detector.reset()
    result = detector.process(frame, settings)
    entry = empty_entry()
    entry["green"] = _clean_points(np.asarray(result.green_points).reshape(-1, 2))
    entry["red"] = _clean_points(np.asarray(result.red_points).reshape(-1, 2))
    return entry


def add_point(entry, color, x, y):
    if color not in COLORS:
        raise ValueError(f"Cor desconhecida: {color}")
    entry[color].append([int(round(x)), int(round(y))])


def find_near(entry, x, y, radius=REMOVE_RADIUS_PX):
    """(cor, indice) da base mais proxima dentro do raio, ou None."""
    best = None
    for color in COLORS:
        for index, (px, py) in enumerate(entry[color]):
            distance = np.hypot(px - x, py - y)
            if distance <= radius and (best is None or distance < best[0]):
                best = (distance, color, index)
    return None if best is None else best[1:]


def remove_near(entry, x, y, radius=REMOVE_RADIUS_PX):
    """Remove a base mais proxima dentro do raio; True se removeu."""
    found = find_near(entry, x, y, radius)
    if found is None:
        return False
    color, index = found
    del entry[color][index]
    return True


def click(entry, color, x, y, radius=REMOVE_RADIUS_PX):
    """Clique sobre um circulo remove; fora de circulos adiciona base da cor."""
    if remove_near(entry, x, y, radius):
        return "removido"
    add_point(entry, color, x, y)
    return "adicionado"


class LabelSession:
    """Estado da rotulagem, sem janela: testavel sem display."""

    def __init__(self, folder, detector=None, settings=None, start=None):
        self.folder = Path(folder)
        self.frames = list_frames(self.folder)
        if not self.frames:
            raise ValueError(f"Nenhum quadro em {self.folder / FRAMES_DIR}/f*.jpg")
        self.labels = load_labels(self.folder)
        self.detector = detector or ConePathDetector()
        self.settings = settings or ConeSettings(tracking=False)
        if start is None:
            start = next((i for i, path in enumerate(self.frames)
                          if not self.labels.get(path.name, {}).get("reviewed")), 0)
        self.index = int(np.clip(start, 0, len(self.frames) - 1))
        self._image = None
        self._image_index = None

    @property
    def name(self):
        return self.frames[self.index].name

    def image(self):
        if self._image_index != self.index:
            self._image = cv2.imread(str(self.frames[self.index]), cv2.IMREAD_COLOR)
            if self._image is None:
                raise ValueError(f"Nao foi possivel ler {self.frames[self.index]}")
            self._image_index = self.index
        return self._image

    def entry(self):
        """Rotulo do quadro atual; cria a partir do detector se nao existir."""
        if self.name not in self.labels:
            self.labels[self.name] = prelabel(self.image(), self.detector, self.settings)
        return self.labels[self.name]

    def click(self, color, x, y):
        return click(self.entry(), color, x, y)

    def clear(self):
        entry = self.entry()
        entry["green"], entry["red"] = [], []

    def redo_prelabel(self):
        reviewed = self.entry()["reviewed"]
        self.labels[self.name] = prelabel(self.image(), self.detector, self.settings)
        self.labels[self.name]["reviewed"] = reviewed

    def mark_reviewed(self):
        self.entry()["reviewed"] = True

    def next(self):
        self.mark_reviewed()
        self.index = min(self.index + 1, len(self.frames) - 1)

    def previous(self):
        self.mark_reviewed()
        self.index = max(self.index - 1, 0)

    def save(self):
        return save_labels(self.folder, self.labels)

    def reviewed_count(self):
        return sum(1 for path in self.frames if self.labels.get(path.name, {}).get("reviewed"))


def draw_labels(frame, entry, scale=1.0):
    output = frame.copy()
    if scale != 1:
        output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    for color in COLORS:
        for x, y in entry[color]:
            center = (round(x * scale), round(y * scale))
            cv2.circle(output, center, round(REMOVE_RADIUS_PX * scale), DRAW[color], 2, cv2.LINE_AA)
            cv2.circle(output, center, 2, (255, 255, 255), -1, cv2.LINE_AA)
    return output


def _caption(image, text, y):
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------- janela

def run_window(session, scale=1.5):
    window = "rotular_cones"
    flags = cv2.WINDOW_AUTOSIZE | getattr(cv2, "WINDOW_GUI_NORMAL", 0)  # sem menu no botao direito
    cv2.namedWindow(window, flags)

    def on_mouse(event, x, y, mouse_flags, _):
        ix, iy = x / scale, y / scale
        if event == cv2.EVENT_LBUTTONDOWN:
            color = "red" if mouse_flags & cv2.EVENT_FLAG_CTRLKEY else "green"
            session.click(color, ix, iy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            session.click("red", ix, iy)

    cv2.setMouseCallback(window, on_mouse)
    message = ""
    try:
        while True:
            entry = session.entry()
            view = draw_labels(session.image(), entry, scale)
            state = "REVISADO" if entry["reviewed"] else "pre-rotulo"
            _caption(view, f"{session.name}  {session.index + 1}/{len(session.frames)}  {state}  "
                           f"verde={len(entry['green'])} vermelho={len(entry['red'])}", 22)
            _caption(view, message or "esq: verde  dir: vermelho  n/b: navegar  s: salvar  q: sair", 46)
            cv2.imshow(window, view)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("n"), ord(" ")):
                last = session.index == len(session.frames) - 1
                session.next()
                message = "ultimo quadro" if last else ""
            elif key == ord("b"):
                session.previous()
                message = ""
            elif key == ord("s"):
                session.mark_reviewed()
                path = session.save()
                message = f"salvo {path.name}: {session.reviewed_count()} revisados"
            elif key == ord("c"):
                session.clear()
            elif key == ord("r"):
                session.redo_prelabel()
            elif key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        path = session.save()
        cv2.destroyAllWindows()
    print(f"{path}: {session.reviewed_count()}/{len(session.frames)} quadros revisados")


def load_calibration(path):
    """GroundCalibration do arquivo, ou None (com aviso) se nao houver."""
    if path is None or str(path) == "":
        return None
    path = Path(path)
    if not path.exists():
        print(f"Aviso: calibracao {path} nao encontrada; seguindo sem calibracao.", file=sys.stderr)
        return None
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from fva_car.lateral import GroundCalibration
    return GroundCalibration.from_file(path)


def main(argv=None):
    parser = make_parser(__doc__)
    parser.add_argument("pasta", help="pasta do ensaio com frames/fNNNNNN.jpg")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), metavar="ARQUIVO",
                        help="JSON de calibracao do piso (padrao: veiculo_real/calibracao/"
                             "camera_lateral.json; '' para nenhuma)")
    parser.add_argument("--inicio", type=int, default=None, metavar="N",
                        help="indice do primeiro quadro (padrao: primeiro nao revisado)")
    parser.add_argument("--escala", type=float, default=1.5, metavar="X",
                        help="ampliacao da janela para clicar com precisao (padrao: %(default)s)")
    args = parser.parse_args(argv)
    detector = ConePathDetector(load_calibration(args.calibration))
    session = LabelSession(args.pasta, detector, start=args.inicio)
    run_window(session, max(args.escala, .25))
    return 0


if __name__ == "__main__":
    sys.exit(main())
