"""Segmentacao HSV e selecao geometrica de uma faixa, sem depender da GUI.

O caminho e representado como x(y) na imagem: serve para retas e curvas que
avancam pela imagem, mas nao para retornos que se sobrepoem na mesma altura.
As medidas sao em pixels, sem calibracao metrica ou classificacao semantica.
"""

from dataclasses import dataclass, field
import time

import cv2
import numpy as np


@dataclass
class PathSettings:
    h_min: int = 18
    h_max: int = 40
    s_min: int = 65
    v_min: int = 45
    clahe: bool = True
    roi_top: int = 30          # Porcentagem da altura ignorada no topo.
    max_gap: int = 36          # Falha vertical maxima; escala base de 480 px.
    max_width: int = 18        # Porcentagem da largura da imagem.
    min_span: int = 25         # Extensao vertical minima, % da altura total.
    smoothing: int = 35        # Peso temporal; zero desativa.


@dataclass
class PathResult:
    raw_mask: np.ndarray
    path_mask: np.ndarray
    observed_mask: np.ndarray
    center: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.int32))
    support: float = 0.0
    status: str = "Buscando faixa"

    @property
    def valid(self):
        return len(self.center) > 0


class YellowPathDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.anchor = 0.5
        self.reset()

    def reset(self, anchor=None):
        self.previous = None
        self.previous_time = -float("inf")
        if anchor is not None:
            self.anchor = float(np.clip(anchor, 0, 1))

    def segment(self, frame, settings):
        hsv = cv2.cvtColor(cv2.medianBlur(frame, 3), cv2.COLOR_BGR2HSV)
        if settings.clahe:
            hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        lo, hi = sorted((settings.h_min, settings.h_max))
        strong = cv2.inRange(hsv, (lo, settings.s_min, settings.v_min), (hi, 255, 255))
        # Limiar mais tolerante recupera tinta suja/desbotada. Uma faixa ainda
        # precisa de sementes fortes e continuidade geometrica para ser aceita.
        weak = cv2.inRange(hsv,
                           (max(0, lo - 3), max(25, int(settings.s_min * .55)),
                            max(15, int(settings.v_min * .55))),
                           (min(179, hi + 3), 255, 255))
        roi = int(frame.shape[0] * settings.roi_top / 100)
        strong[:roi] = 0
        weak[:roi] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, kernel)
        return strong, weak

    @staticmethod
    def _candidates(strong, weak, settings):
        height, width = weak.shape
        step = max(3, round(height / 80))
        roi = int(height * settings.roi_top / 100)
        rows = []
        max_width = width * settings.max_width / 100
        for bottom in range(height - 1, roi + step - 1, -step):
            top = bottom - step + 1
            active = np.count_nonzero(weak[top:bottom + 1], axis=0) >= max(1, step // 3)
            changes = np.diff(np.r_[False, active, False].astype(np.int8))
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            candidates = []
            for start, end in zip(starts, ends):
                size = end - start
                if not 2 <= size <= max_width:
                    continue
                evidence = np.count_nonzero(strong[top:bottom + 1, start:end]) / (step * size)
                candidates.append(((start + end - 1) / 2, (top + bottom) / 2,
                                   float(size), float(evidence)))
            # Limita trabalho em texturas muito ruidosas sem abrir mao de
            # manter varios candidatos concorrentes para curvas/bifurcacoes.
            candidates.sort(key=lambda p: p[3] + min(p[2], 24) / 24, reverse=True)
            candidates = candidates[:24]
            rows.append(candidates)
        return rows, step

    def _trace(self, strong, weak, settings, now):
        """Programacao dinamica: sobe pela imagem, permitindo falhas limitadas.

        Cada candidato guarda a melhor cadeia ate ele. Penaliza saltos laterais,
        mudancas bruscas de direcao e largura, preferindo trajetorias extensas.
        """
        height, width = weak.shape
        rows, step = self._candidates(strong, weak, settings)
        gap_limit = settings.max_gap * height / 480 + step
        seed_y = height * (settings.roi_top / 100 + .55 * (1 - settings.roi_top / 100))
        active, paths = [], []
        recent = self.previous is not None and now - self.previous_time < .25
        for candidates in rows:
            if not candidates:
                continue
            y = candidates[0][1]
            active = [p for p in active if 0 < p[1][-1][1] - y <= gap_limit]
            new = []
            for point in candidates:
                x, y, band_width, evidence = point
                reward = step * (.35 + .65 * min(1., evidence * 2))
                best = None
                if y >= seed_y:
                    penalty = 45 * abs(x / width - self.anchor)
                    if recent:
                        expected = np.polyval(self.previous, y / height)
                        penalty += min(25., abs(x - expected) * .15)
                    best = (reward - penalty, (point,), 0.)
                for score, chain, slope in active:
                    px, py, pw, _ = chain[-1]
                    dy, dx = py - y, x - px
                    prediction = px + slope * dy
                    if abs(x - prediction) > max(12., width * .025) + .65 * dy:
                        continue
                    new_slope = dx / dy
                    if abs(new_slope) > 2.8:
                        continue
                    width_change = abs(np.log(max(band_width, 1) / max(pw, 1)))
                    if width_change > 1.25:
                        continue
                    turn_penalty = abs(new_slope - slope) if len(chain) > 1 else 0
                    cost = (.12 * abs(dx) + .22 * max(0, dy - step)
                            + step * .35 * turn_penalty + step * .25 * width_change)
                    candidate = (score + reward - cost, chain + (point,),
                                 .5 * slope + .5 * new_slope)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
                if best is not None:
                    new.append(best)
                    paths.append(best)
            active.extend(new)
        acceptable = []
        for path in paths:
            points = np.asarray(path[1])
            span = np.ptp(points[:, 1])
            if (span >= height * settings.min_span / 100
                    and span / max(1., np.median(points[:, 2])) >= 3.2
                    and len(points) * step / max(span, 1) >= .5
                    and np.mean(points[:, 3] >= .12) >= .25):
                acceptable.append(path)
        return None if not acceptable else np.asarray(max(acceptable, key=lambda p: p[0])[1])

    def process(self, frame, settings=None, timestamp=None):
        settings = PathSettings() if settings is None else settings
        now = time.monotonic() if timestamp is None else timestamp
        strong, weak = self.segment(frame, settings)
        height, width = weak.shape
        empty = np.zeros_like(weak)
        result = PathResult(strong, empty.copy(), empty.copy())
        points = self._trace(strong, weak, settings, now)
        if points is None:
            return result
        # Ajuste cubico robusto para retas, arcos e curvas em S. Sem extrapolar
        # alem do trecho observado; lacunas longas ja foram rejeitadas no trace.
        xx, yy, widths = points[:, 0], points[:, 1] / height, points[:, 2]
        keep = np.ones(len(points), dtype=bool)
        coefficients = None
        for _ in range(3):
            if np.count_nonzero(keep) < 6:
                return result
            coefficients = np.polyfit(yy[keep], xx[keep], 3)
            residual = np.abs(xx - np.polyval(coefficients, yy))
            threshold = max(6., np.median(widths) * .65,
                            3 * float(np.median(residual[keep])))
            keep = residual <= threshold
        points = points[keep]
        if (len(points) < 6 or np.ptp(points[:, 1]) < height * settings.min_span / 100
                or np.max(np.diff(np.sort(points[:, 1]))) > settings.max_gap * height / 480 + height / 80 + 1):
            return result
        coefficients = np.polyfit(points[:, 1] / height, points[:, 0], 3)
        ys = np.arange(int(np.ceil(points[:, 1].min())), int(points[:, 1].max()) + 1)
        xs = np.polyval(coefficients, ys / height)
        if self.previous is not None and now - self.previous_time < .25:
            old = np.polyval(self.previous, ys / height)
            if np.sqrt(np.mean((old - xs) ** 2)) < max(5., np.median(points[:, 2]) * .6):
                weight = settings.smoothing / 100
                xs = (1 - weight) * xs + weight * old
                coefficients = np.polyfit(ys / height, xs, 3)
        order = np.argsort(points[:, 1])
        half_width = np.interp(ys, points[order, 1], points[order, 2]) / 2
        # A faixa reconstruida conserva a largura medida; nao une todo amarelo.
        left = np.column_stack((np.clip(xs - half_width, 0, width - 1), ys))
        right = np.column_stack((np.clip(xs + half_width, 0, width - 1), ys))[::-1]
        cv2.fillPoly(result.path_mask, [np.rint(np.vstack((left, right))).astype(np.int32)], 255)
        result.observed_mask = cv2.bitwise_and(result.path_mask, weak)
        result.support = cv2.countNonZero(result.observed_mask) / max(1, cv2.countNonZero(result.path_mask))
        strong_support = cv2.countNonZero(cv2.bitwise_and(result.path_mask, strong)) / max(1, cv2.countNonZero(result.path_mask))
        # Uma cadeia em textura aleatoria pode parecer alongada. Confirma que
        # o corredor final tem suporte real, em vez de muitos pontos isolados.
        if result.support < .5 or strong_support < .12:
            return PathResult(strong, empty.copy(), empty.copy(), status="Buscando faixa com mais evidencia")
        result.center = np.rint(np.column_stack((np.clip(xs, 0, width - 1), ys))).astype(np.int32)
        result.status = "Faixa acompanhada" if result.support >= .8 else "Faixa com falhas reconstruidas"
        self.previous, self.previous_time = coefficients, now
        return result


def render_path(frame, result, settings, anchor=.5):
    """Verde = evidencia de cor; laranja = preenchimento geometrico estimado."""
    output = cv2.convertScaleAbs(frame, alpha=.60)
    if result.valid:
        observed = result.observed_mask > 0
        inferred = (result.path_mask > 0) & ~observed
        green = cv2.addWeighted(frame, .25, np.full_like(frame, (70, 245, 30)), .75, 0)
        orange = cv2.addWeighted(frame, .25, np.full_like(frame, (0, 165, 255)), .75, 0)
        output[observed], output[inferred] = green[observed], orange[inferred]
        contours, _ = cv2.findContours(result.path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 235, 90), 1)
        cv2.polylines(output, [result.center], False, (255, 255, 255), 2, cv2.LINE_AA)
    height, width = frame.shape[:2]
    roi = int(height * settings.roi_top / 100)
    for x in range(0, width, 20):
        cv2.line(output, (x, roi), (min(x + 9, width - 1), roi), (160, 160, 160), 1)
    cv2.drawMarker(output, (int(anchor * (width - 1)), height - 14),
                   (255, 220, 80), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return output
