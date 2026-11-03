"""Cones BGR/HSV -> bordas da pista -> centro em pixels, sem depender da GUI.

Usa o contato com o chao e interpola x(y) somente entre cones observados nas
duas bordas. Nao converte pixels em metros nem envia comandos ao carrinho.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


PURPLE = (230, 40, 210)  # BGR
GREEN = (50, 220, 50)
RED = (50, 50, 240)


def _points():
    return np.empty((0, 2), np.int32)


@dataclass
class ConeSettings:
    roi_top: int = 10         # Camera baixa: os cones ficam perto do horizonte.
    min_area: int = 28        # Pixels na escala de 640 x 480.
    max_gap: int = 200        # Distancia vertical maxima, escala de 480 px.
    min_span: int = 5         # Extensao vertical minima, % da altura.
    min_width: int = 8        # Largura minima do corredor, % da imagem.
    reference_y: int = 85     # Altura da secao de medicao, % da imagem.
    camera_center: int = 50   # Coluna alinhada ao eixo do carrinho, %.
    green_left: bool = True
    green_h_min: int = 38
    green_h_max: int = 90
    red_hue: int = 12         # Vermelho nos dois extremos do H (0..179).
    s_min: int = 65           # Tolera cores desbotadas pela webcam/luz ambiente.
    v_min: int = 45
    clahe: bool = False


@dataclass
class ConePathResult:
    raw_mask: np.ndarray
    green_mask: np.ndarray
    red_mask: np.ndarray
    green_points: np.ndarray = field(default_factory=_points)
    red_points: np.ndarray = field(default_factory=_points)
    left_boundary: np.ndarray = field(default_factory=_points)
    right_boundary: np.ndarray = field(default_factory=_points)
    center: np.ndarray = field(default_factory=_points)
    left_distance_px: float | None = None
    right_distance_px: float | None = None
    lateral_error_px: float | None = None  # Positivo: centro da pista a direita.
    reference_y: int | None = None
    status: str = "Buscando cones nas duas bordas"

    @property
    def valid(self):
        return len(self.center) > 0


def _interpolate(points, rows):
    """Hermite cubica com derivadas limitadas (sem overshoot entre cones).

    Cada trecho preserva a monotonicidade local, permitindo curvas em S sem
    as oscilacoes de um unico polinomio global de grau alto.
    """
    xs, ys = points[:, 0], points[:, 1]
    gaps = np.diff(ys)
    slopes = np.diff(xs) / gaps
    tangents = np.zeros(len(points), float)
    tangents[0], tangents[-1] = slopes[0], slopes[-1]
    for i in range(1, len(points) - 1):
        a, b = slopes[i - 1], slopes[i]
        if a * b > 0:
            w1, w2 = 2 * gaps[i] + gaps[i - 1], gaps[i] + 2 * gaps[i - 1]
            tangents[i] = (w1 + w2) / (w1 / a + w2 / b)
    index = np.clip(np.searchsorted(ys, rows, side="right") - 1, 0, len(ys) - 2)
    h = gaps[index]
    t = (rows - ys[index]) / h
    return ((2*t**3 - 3*t**2 + 1)*xs[index]
            + (t**3 - 2*t**2 + t)*h*tangents[index]
            + (-2*t**3 + 3*t**2)*xs[index + 1]
            + (t**3 - t**2)*h*tangents[index + 1])


class ConePathDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))

    def reset(self):
        # Sem memoria de trajetoria: perder uma borda invalida o caminho atual.
        pass

    def segment(self, frame, settings):
        height, width = frame.shape[:2]
        roi = int(height * settings.roi_top / 100)
        hsv = cv2.cvtColor(cv2.medianBlur(frame[roi:], 3), cv2.COLOR_BGR2HSV)
        if settings.clahe:
            hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        low, high = sorted((settings.green_h_min, settings.green_h_max))
        green = cv2.inRange(hsv, (low, settings.s_min, settings.v_min), (high, 255, 255))
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, settings.s_min, settings.v_min), (settings.red_hue, 255, 255)),
            cv2.inRange(hsv, (179-settings.red_hue, settings.s_min, settings.v_min), (179, 255, 255)))
        masks = []
        for color in (green, red):
            mask = np.zeros((height, width), np.uint8)
            # Fecha pequenos buracos sem juntar cones distantes.
            mask[roi:] = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            masks.append(mask)
        return masks

    @staticmethod
    def _bases(mask, settings):
        height, width = mask.shape
        roi = int(height * settings.roi_top / 100)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        bases = []
        min_area = settings.min_area * height * width / (640 * 480)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < min_area or h < max(5, height / 100) or w < 3:
                continue
            if x <= 0 or x+w >= width or y <= roi or y+h >= height:
                continue  # Contato oculto/objeto cortado nao define uma borda.
            if not .18 <= w/h <= 1.5 or w > .2*width:
                continue
            component = labels[y:y+h, x:x+w] == label
            third = max(1, h//3)
            top_width = np.count_nonzero(np.any(component[:third], axis=0))
            bottom_width = np.count_nonzero(np.any(component[-third:], axis=0))
            if top_width >= .9*bottom_width or area < .25*w*h:
                continue  # Rejeita faixas, placas retangulares e ruido esparso.
            columns = np.flatnonzero(np.any(component[-max(2, h//10):], axis=0))
            bases.append((x + float(np.mean(columns)), float(y+h-1)))
        return np.asarray(bases, float).reshape(-1, 2)

    @staticmethod
    def _boundary(points, shape, settings):
        """Seleciona uma cadeia continua por cor, penalizando saltos de direcao."""
        if len(points) < 3:
            return np.empty((0, 2), float)
        height, width = shape
        # Limita custo em cenas com muitas manchas. Prioriza cones proximos.
        points = points[np.argsort(points[:, 1])[-80:]][::-1]
        chains, scores, tangents = [], [], []
        min_dy = max(2., .004*height)
        max_dy = settings.max_gap*height/480
        # Uma camera baixa comprime a profundidade em poucas linhas da imagem:
        # uma borda valida pode avancar muito em x e pouco em y. Normalizar a
        # inclinacao pela proporcao da nuvem preserva essas curvas sem relaxar
        # a checagem angular em cenas que ja ocupam boa extensao vertical.
        perspective_scale = float(np.clip(
            np.ptp(points[:, 0])/max(np.ptp(points[:, 1]), .03*height), 1., 6.))
        for i, point in enumerate(points):
            best, score, tangent = [i], 1., None
            for j in range(i):
                dy = points[j, 1] - point[1]
                dx = point[0] - points[j, 0]
                if not min_dy <= dy <= max_dy or abs(dx) > perspective_scale*(1.8*dy + .03*width):
                    continue
                slope = dx/(dy*perspective_scale)
                turn = 0. if tangents[j] is None else abs(np.arctan(slope) - np.arctan(tangents[j]))
                if turn > np.deg2rad(85):
                    continue
                candidate = scores[j] + 1 + dy/height - .35*turn
                if candidate > score:
                    best, score, tangent = chains[j] + [i], candidate, slope
            chains.append(best)
            scores.append(score)
            tangents.append(tangent)
        eligible = [i for i, chain in enumerate(chains) if len(chain) >= 3
                    and np.ptp(points[chain, 1]) >= height*settings.min_span/100]
        if not eligible:
            return np.empty((0, 2), float)
        chosen = max(eligible, key=lambda i: scores[i] + points[chains[i][0], 1]/height)
        return points[chains[chosen]][::-1]  # y crescente, sem duplicatas.

    def process(self, frame, settings=None, timestamp=None):
        settings = settings or ConeSettings()
        green, red = self.segment(frame, settings)
        result = ConePathResult(cv2.bitwise_or(green, red), green, red)
        gp, rp = self._bases(green, settings), self._bases(red, settings)
        result.green_points, result.red_points = gp, rp
        left, right = (gp, rp) if settings.green_left else (rp, gp)
        if len(gp) == 0 and len(rp) == 0:
            result.status = "Nenhum cone aceito: reduza Ignorar topo e confira HSV cones"
            return result
        if len(gp) < 3 or len(rp) < 3:
            result.status = f"Cones insuficientes: {len(gp)} verdes / {len(rp)} vermelhos (minimo 3 por cor)"
            return result
        left = self._boundary(left, green.shape, settings)
        right = self._boundary(right, green.shape, settings)
        if not len(left) or not len(right):
            result.status = "Bordas curtas/descontinuas: confira Extensao minima e Espaco vertical"
            return result
        height, width = green.shape
        start = int(np.ceil(max(left[0, 1], right[0, 1])))
        end = int(np.floor(min(left[-1, 1], right[-1, 1])))
        if end-start < height*settings.min_span/100:
            result.status = "Bordas sem extensao comum suficiente"
            return result
        rows = np.arange(start, end+1)
        lx, rx = _interpolate(left, rows), _interpolate(right, rows)
        widths = rx-lx
        if np.any(widths < width*settings.min_width/100):
            result.status = "Bordas cruzadas/estreitas: confira a cor do lado esquerdo"
            return result
        # O centro fica entre as bordas na MESMA linha y, mesmo se os cones
        # estiverem alternados em profundidade (nao pareia vizinhos por indice).
        cx = (lx+rx)/2
        result.left_boundary = np.c_[lx, rows].round().astype(np.int32)
        result.right_boundary = np.c_[rx, rows].round().astype(np.int32)
        result.center = np.c_[cx, rows].round().astype(np.int32)
        y = int(np.clip(round((height-1)*settings.reference_y/100), start, end))
        axis = (width-1)*settings.camera_center/100
        result.reference_y = y
        result.left_distance_px = float(axis-lx[y-start])
        result.right_distance_px = float(rx[y-start]-axis)
        result.lateral_error_px = float(cx[y-start]-axis)
        result.status = "Trajetoria por cones | " + (
            "eixo fora do corredor" if min(result.left_distance_px, result.right_distance_px) < 0
            else "centro entre as duas bordas")
        return result


def render_cones(frame, result, settings=None, *, path_only=False):
    """Bordas, bases, centro roxo e medidas, sem alterar a imagem de entrada."""
    settings = settings or ConeSettings()
    output = np.zeros_like(frame) if path_only else frame.copy()
    for points, color in ((result.green_points, GREEN), (result.red_points, RED)):
        for x, y in points:
            cv2.circle(output, (round(x), round(y)), 5, color, 2, cv2.LINE_AA)
    if not result.valid:
        return output
    lc, rc = (GREEN, RED) if settings.green_left else (RED, GREEN)
    cv2.polylines(output, [result.left_boundary], False, lc, 2, cv2.LINE_AA)
    cv2.polylines(output, [result.right_boundary], False, rc, 2, cv2.LINE_AA)
    y = result.reference_y
    index = y-int(result.center[0, 1])
    axis = round((frame.shape[1]-1)*settings.camera_center/100)
    for border, color in ((result.left_boundary, lc), (result.right_boundary, rc)):
        cv2.line(output, (axis, y), tuple(border[index]), color, 2, cv2.LINE_AA)
    cv2.drawMarker(output, (axis, y), (255, 255, 255), cv2.MARKER_CROSS, 17, 2)
    cv2.polylines(output, [result.center], False, PURPLE, 4, cv2.LINE_AA)
    caption = (f"E: {result.left_distance_px:.0f}px  D: {result.right_distance_px:.0f}px  "
               f"Erro: {result.lateral_error_px:+.0f}px  y={y}")
    cv2.putText(output, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return output
