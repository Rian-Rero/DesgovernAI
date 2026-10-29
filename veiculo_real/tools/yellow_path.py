"""Segmentacao HSV e selecao geometrica de uma faixa, sem depender da GUI.

O caminho e representado como x(y) na imagem: serve para retas e curvas que
avancam pela imagem, mas nao para retornos que se sobrepoem na mesma altura.
As medidas sao em pixels, sem calibracao metrica ou classificacao semantica.
"""

from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np


@dataclass
class PathSettings:
    h_min: int = 18
    h_max: int = 40
    s_min: int = 25
    v_min: int = 45
    clahe: bool = True
    roi_top: int = 30          # Porcentagem da altura ignorada no topo.
    max_gap: int = 36          # Falha vertical maxima; escala base de 480 px.
    max_width: int = 18        # Porcentagem da largura da imagem.
    min_span: int = 25         # Extensao vertical minima, % da altura total.
    smoothing: int = 35        # Peso temporal; zero desativa.
    dashed: bool = True        # Conecta tracos separados com tangentes compativeis.
    dash_gap: int = 120        # Gap pontilhado maximo, na escala de 480 px.


@dataclass
class PathResult:
    raw_mask: np.ndarray
    path_mask: np.ndarray
    observed_mask: np.ndarray
    center: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.int32))
    support: float = 0.0
    status: str = "Buscando faixa"
    dash_mask: np.ndarray | None = None  # Interpolacao entre tracos, sem cor observada.

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
        # O topo descartado nem entra no filtro/HSV/CLAHE: alem de poupar
        # processamento, outras linhas pintadas ao fundo nao viram evidencia.
        roi = int(frame.shape[0] * settings.roi_top / 100)
        hsv = cv2.cvtColor(cv2.medianBlur(frame[roi:], 3), cv2.COLOR_BGR2HSV)
        if settings.clahe:
            hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        lo, hi = sorted((settings.h_min, settings.h_max))
        strong = np.zeros(frame.shape[:2], np.uint8)
        weak = np.zeros_like(strong)
        strong[roi:] = cv2.inRange(hsv, (lo, settings.s_min, settings.v_min), (hi, 255, 255))
        # Limiar mais tolerante recupera tinta suja/desbotada. Uma faixa ainda
        # precisa de sementes fortes e continuidade geometrica para ser aceita.
        lower = cv2.inRange(hsv,
                            (max(0, lo - 3), max(25, int(settings.s_min * .55)),
                             max(15, int(settings.v_min * .55))),
                            (min(179, hi + 3), 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        weak[roi:] = cv2.morphologyEx(lower, cv2.MORPH_CLOSE, kernel)
        return strong, weak

    @staticmethod
    def _candidates(strong, weak, settings):
        height, width = weak.shape
        step = max(3, round(height / 80))
        roi = int(height * settings.roi_top / 100)
        rows = []
        max_width = width * settings.max_width / 100
        count = max(0, (height - roi - 1) // step)
        if count == 0:
            return rows, step
        # Conta todas as bandas de uma vez; a soma prefixada permite medir
        # o suporte de cada intervalo sem percorrer novamente seus pixels.
        shape = (count, step, width)
        active = np.count_nonzero(weak[height-count*step:].reshape(shape), axis=1)[::-1]
        evidence_columns = np.count_nonzero(strong[height-count*step:].reshape(shape), axis=1)[::-1]
        prefix = np.zeros((count, width + 1), np.int32)
        np.cumsum(evidence_columns, axis=1, dtype=np.int32, out=prefix[:, 1:])
        padded = np.zeros((count, width + 2), np.int8)
        padded[:, 1:-1] = active >= max(1, step // 3)
        all_changes = np.diff(padded, axis=1)
        for index, changes in enumerate(all_changes):
            bottom = height - 1 - index * step
            top = bottom - step + 1
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            candidates = []
            for start, end in zip(starts, ends):
                size = end - start
                if not 2 <= size <= max_width:
                    continue
                evidence = (prefix[index, end] - prefix[index, start]) / (step * size)
                candidates.append((float(start + end - 1) / 2, (top + bottom) / 2,
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
        short_limit = settings.max_gap * height / 480 + step
        gap_limit = max(short_limit, settings.dash_gap * height / 480 + step) if settings.dashed else short_limit
        seed_y = height * (settings.roi_top / 100 + .55 * (1 - settings.roi_top / 100))
        active, paths = [], []
        recent = self.previous is not None and now - self.previous_time < .25
        for row_index, candidates in enumerate(rows):
            if not candidates:
                continue
            y = candidates[0][1]
            active = [p for p in active if 0 < p[1][-1][1] - y <= gap_limit]
            # Com varios objetos amarelos, elimina em NumPy os pais que
            # falhariam nos mesmos dois primeiros testes geometricos abaixo.
            # Mantem a ordem original e todas as validacoes de pontilhado.
            parent_x = parent_dy = predictions = None
            if len(active) >= 32:
                parents = np.asarray([(p[1][-1][0], p[1][-1][1], p[2], p[0],
                                       p[1][-1][2], len(p[1])) for p in active])
                parent_x = parents[:, 0]
                parent_dy = parents[:, 1] - y
                predictions = parent_x + parents[:, 2] * parent_dy
                reach = max(12., width * .025) + .65 * parent_dy
            expected = np.polyval(self.previous, y / height) if recent and y >= seed_y else None
            new = []
            for point in candidates:
                x, y, band_width, evidence = point
                reward = step * (.35 + .65 * min(1., evidence * 2))
                best = None
                upper_tangent = None
                tangent_checked = False
                if y >= seed_y:
                    penalty = 45 * abs(x / width - self.anchor)
                    if recent:
                        penalty += min(25., abs(x - expected) * .15)
                    best = (reward - penalty, (point,), 0.)
                has_short_parent = False
                parents_to_check = reversed(active)
                if parent_x is not None:
                    new_slopes = (x - parent_x) / parent_dy
                    width_changes = np.abs(np.log(max(band_width, 1) / np.maximum(parents[:, 4], 1)))
                    compatible = ((np.abs(x - predictions) <= reach)
                                  & (np.abs(new_slopes) <= 2.8)
                                  & (width_changes <= 1.25))
                    short = np.flatnonzero(compatible & (parent_dy <= short_limit))[::-1]
                    if len(short):
                        # Todos os pais curtos ja passaram pelos mesmos testes.
                        # A versao escalar ignora os longos assim que acha um
                        # curto; calcula seus scores de uma vez em NumPy.
                        dx = x - parent_x[short]
                        turn = np.where(parents[short, 5] > 1,
                                        np.abs(new_slopes[short] - parents[short, 2]), 0.)
                        costs = (.12 * np.abs(dx) + .22 * np.maximum(0., parent_dy[short] - step)
                                 + step * .35 * turn + step * .25 * width_changes[short])
                        scores = parents[short, 3] + reward - costs
                        winner = int(np.argmax(scores))
                        index = short[winner]
                        score, chain, slope = active[index]
                        candidate = (float(scores[winner]), chain + (point,),
                                     .5 * slope + .5 * float(new_slopes[index]))
                        if best is None or candidate[0] > best[0]:
                            best = candidate
                        parents_to_check = ()
                    else:
                        parents_to_check = (active[index] for index in np.flatnonzero(compatible)[::-1])
                for score, chain, slope in parents_to_check:
                    px, py, pw, _ = chain[-1]
                    dy, dx = py - y, x - px
                    # active esta em ordem de altura. Apos encontrar um pai
                    # curto, nenhum pai mais distante pode ser escolhido.
                    if has_short_parent and dy > short_limit:
                        break
                    prediction = px + slope * dy
                    if abs(x - prediction) > max(12., width * .025) + .65 * dy:
                        continue
                    new_slope = dx / dy
                    if abs(new_slope) > 2.8:
                        continue
                    width_change = abs(math.log(max(band_width, 1) / max(pw, 1)))
                    if width_change > 1.25:
                        continue
                    if dy <= short_limit:
                        has_short_parent = True
                    if dy > short_limit:
                        if has_short_parent:
                            continue
                        # Gaps de pontilhado exigem orientacao medida nos dois
                        # lados: nao basta aproximar dois objetos amarelos.
                        if not tangent_checked:
                            upper_tangent = self._forward_tangent(point, rows[row_index + 1:row_index + 5], step)
                            tangent_checked = True
                        lower_tangent = self._tail_tangent(chain, step)
                        if (upper_tangent is None or lower_tangent is None
                                or evidence < .12 or chain[-1][3] < .12 or width_change > .65):
                            continue
                        if abs(math.atan(upper_tangent) - math.atan(lower_tangent)) > math.radians(35):
                            continue
                        mean_slope = (upper_tangent + lower_tangent) / 2
                        if abs(new_slope - mean_slope) > .22 + .4 * abs(upper_tangent - lower_tangent):
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
        # O resultado sempre foi o caminho aceito de maior score. Avaliar
        # nessa ordem permite parar no primeiro, sem validar dezenas de
        # prefixos que jamais serao selecionados. sorted conserva os empates.
        for path in sorted(paths, key=lambda p: p[0], reverse=True):
            chain = path[1]
            span = chain[0][1] - chain[-1][1]
            if span < height * settings.min_span / 100:
                continue
            points = np.asarray(path[1])
            long_gap = settings.dashed and np.any(np.abs(np.diff(points[:, 1])) > short_limit)
            if (span / max(1., np.median(points[:, 2])) >= 3.2
                    and len(points) * step / max(span, 1) >= (.28 if long_gap else .5)
                    and np.mean(points[:, 3] >= .12) >= .25):
                return points
        return None

    @staticmethod
    def _tail_tangent(chain, step):
        tail = [chain[-1]]
        for point in reversed(chain[:-1]):
            if point[1] - tail[-1][1] > 1.6 * step or len(tail) >= 6:
                break
            tail.append(point)
        if len(tail) < 3:
            return None
        return YellowPathDetector._tangent_slope(tail)

    @staticmethod
    def _forward_tangent(point, upcoming_rows, step):
        points = [point]
        for row in upcoming_rows:
            if not row:
                break
            previous = points[-1]
            candidate = min(row, key=lambda p: abs(p[0] - previous[0]))
            if (previous[1] - candidate[1] > 1.6 * step
                    or abs(candidate[0] - previous[0]) > max(12., previous[2]) + step):
                break
            points.append(candidate)
        if len(points) < 3:
            return None
        return YellowPathDetector._tangent_slope(points)

    @staticmethod
    def _tangent_slope(points):
        """Mesmo ajuste linear, centrado, sem criar uma SVD para 3 a 6 pontos."""
        mean_y = sum(p[1] for p in points) / len(points)
        mean_x = sum(p[0] for p in points) / len(points)
        numerator = sum((mean_y-p[1]) * (p[0]-mean_x) for p in points)
        denominator = sum((mean_y-p[1])**2 for p in points)
        return float(numerator / denominator)

    @staticmethod
    def _dash_mask(path_mask, observed_mask, height, settings):
        """Marca somente falhas internas entre observacoes atuais dos tracos."""
        mask = np.zeros_like(path_mask)
        if not settings.dashed:
            return mask
        total = np.count_nonzero(path_mask, axis=1)
        observed = np.count_nonzero(observed_mask, axis=1)
        missing = (total > 0) & (observed < .15 * total)
        changes = np.diff(np.r_[False, missing, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if (start > 0 and end < height and total[start - 1] > 0 and total[end] > 0
                    and observed[start - 1] > 0 and observed[end] > 0
                    and end - start >= max(6, round(12 * height / 480))):
                mask[start:end] = path_mask[start:end]
                mask[start:end][observed_mask[start:end] > 0] = 0
        return mask

    def process(self, frame, settings=None, timestamp=None):
        settings = PathSettings() if settings is None else settings
        now = time.monotonic() if timestamp is None else timestamp
        strong, weak = self.segment(frame, settings)
        height, width = weak.shape
        empty = np.zeros_like(weak)
        result = PathResult(strong, empty.copy(), empty.copy(), dash_mask=empty.copy())
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
        step = max(3, round(height / 80))
        gap = max(settings.max_gap, settings.dash_gap) if settings.dashed else settings.max_gap
        if (len(points) < 6 or np.ptp(points[:, 1]) < height * settings.min_span / 100
                or np.max(np.diff(np.sort(points[:, 1]))) > gap * height / 480 + step + 1):
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
        # Pontas arredondadas dos tracos parecem estreitas em uma unica banda.
        # Usa a largura tipica das bandas vizinhas para a ponte nao afunilar.
        nearby = np.abs(points[:, 1, None] - points[None, :, 1]) <= 5 * step
        typical = np.nanmedian(np.where(nearby, points[None, :, 2], np.nan), axis=1)
        measured_widths = np.maximum(points[:, 2], .85 * typical)
        half_width = np.interp(ys, points[order, 1], measured_widths[order]) / 2
        # A faixa reconstruida conserva a largura medida; nao une todo amarelo.
        left = np.column_stack((np.clip(xs - half_width, 0, width - 1), ys))
        right = np.column_stack((np.clip(xs + half_width, 0, width - 1), ys))[::-1]
        cv2.fillPoly(result.path_mask, [np.rint(np.vstack((left, right))).astype(np.int32)], 255)
        result.observed_mask = cv2.bitwise_and(result.path_mask, weak)
        result.support = cv2.countNonZero(result.observed_mask) / max(1, cv2.countNonZero(result.path_mask))
        strong_support = cv2.countNonZero(cv2.bitwise_and(result.path_mask, strong)) / max(1, cv2.countNonZero(result.path_mask))
        long_gap = settings.dashed and np.any(np.diff(np.sort(points[:, 1])) > settings.max_gap * height / 480 + step)
        # Uma cadeia em textura aleatoria pode parecer alongada. Confirma que
        # o corredor final tem suporte real, em vez de muitos pontos isolados.
        if result.support < (.28 if long_gap else .5) or strong_support < .12:
            return PathResult(strong, empty.copy(), empty.copy(), status="Buscando faixa com mais evidencia", dash_mask=empty.copy())
        result.center = np.rint(np.column_stack((np.clip(xs, 0, width - 1), ys))).astype(np.int32)
        result.dash_mask = self._dash_mask(result.path_mask, result.observed_mask, height, settings)
        result.status = "Faixa acompanhada" if result.support >= .8 else "Faixa com falhas reconstruidas"
        if cv2.countNonZero(result.dash_mask):
            result.status = "Faixa pontilhada: conexoes em azul"
        self.previous, self.previous_time = coefficients, now
        return result


def render_path(frame, result, settings, anchor=.5):
    """Verde = cor; laranja = buracos; azul = conexao entre tracos separados."""
    output = cv2.convertScaleAbs(frame, alpha=.60)
    if result.valid:
        observed = result.observed_mask > 0
        inferred = (result.path_mask > 0) & ~observed
        green = cv2.addWeighted(frame, .25, np.full_like(frame, (70, 245, 30)), .75, 0)
        orange = cv2.addWeighted(frame, .25, np.full_like(frame, (0, 165, 255)), .75, 0)
        output[observed], output[inferred] = green[observed], orange[inferred]
        if result.dash_mask is not None:
            blue = cv2.addWeighted(frame, .15, np.full_like(frame, (255, 95, 25)), .85, 0)
            output[result.dash_mask > 0] = blue[result.dash_mask > 0]
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
