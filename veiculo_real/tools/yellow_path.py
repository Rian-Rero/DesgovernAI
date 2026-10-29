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
        As cadeias ficam numa arvore (cada no aponta para o pai) e cada banda
        compara todos os candidatos com todos os pais de uma vez, em NumPy.
        """
        height, width = weak.shape
        rows, step = self._candidates(strong, weak, settings)
        short_limit = settings.max_gap * height / 480 + step
        gap_limit = max(short_limit, settings.dash_gap * height / 480 + step) if settings.dashed else short_limit
        seed_y = height * (settings.roi_top / 100 + .55 * (1 - settings.roi_top / 100))
        recent = self.previous is not None and now - self.previous_time < .25
        reach = max(12., width * .025)
        total = sum(len(candidates) for candidates in rows)
        # Colunas: x, y, largura, evidencia, score, inclinacao, y da raiz.
        nodes = np.empty((total, 7))
        lengths = np.empty(total, np.int32)
        parents = np.empty(total, np.int32)
        count = first_active = 0
        tails = {}

        def tail_tangent(node):
            if node not in tails:
                tail = [node]
                while len(tail) < 6 and parents[tail[-1]] >= 0:
                    parent = parents[tail[-1]]
                    if nodes[parent, 1] - nodes[tail[-1], 1] > 1.6 * step:
                        break
                    tail.append(parent)
                tails[node] = (None if len(tail) < 3 else
                               self._tangent_slope(nodes[tail, :2].tolist()))
            return tails[node]

        for row_index, candidates in enumerate(rows):
            if not candidates:
                continue
            points = np.asarray(candidates)
            xs, y, widths, evidence = points[:, 0], points[0, 1], points[:, 2], points[:, 3]
            # Nos estao em ordem de altura: os ativos formam um sufixo contiguo.
            while first_active < count and nodes[first_active, 1] - y > gap_limit:
                first_active += 1
            rewards = step * (.35 + .65 * np.minimum(1., evidence * 2))
            best_scores = np.full(len(points), -np.inf)
            best_parents = np.full(len(points), -1, np.int32)
            best_slopes = np.zeros(len(points))
            if y >= seed_y:
                penalties = 45 * np.abs(xs / width - self.anchor)
                if recent:
                    expected = np.polyval(self.previous, y / height)
                    penalties = penalties + np.minimum(25., np.abs(xs - expected) * .15)
                best_scores = rewards - penalties
            if first_active < count:
                active = slice(first_active, count)
                parent_x, parent_y, parent_w, _, parent_score, parent_slope, _ = nodes[active].T
                dy = parent_y - y
                dx = xs[:, None] - parent_x
                new_slopes = dx / dy
                width_changes = np.abs(np.log(np.maximum(widths, 1)[:, None] / np.maximum(parent_w, 1)))
                compatible = ((np.abs(xs[:, None] - (parent_x + parent_slope * dy)) <= reach + .65 * dy)
                              & (np.abs(new_slopes) <= 2.8) & (width_changes <= 1.25))
                turn = np.where(lengths[active] > 1, np.abs(new_slopes - parent_slope), 0.)
                costs = (.12 * np.abs(dx) + .22 * np.maximum(0., dy - step)
                         + step * .35 * turn + step * .25 * width_changes)
                scores = parent_score + rewards[:, None] - costs
                short = compatible & (dy <= short_limit)
                # Havendo pai curto compativel, os longos nunca sao escolhidos.
                # Em empate vence o pai mais proximo, como no laco escalar.
                masked = np.where(short, scores, -np.inf)[:, ::-1]
                picks = count - first_active - 1 - np.argmax(masked, axis=1)
                picked = scores[np.arange(len(points)), picks]
                better = short.any(axis=1) & (picked > best_scores)
                best_scores[better] = picked[better]
                best_parents[better] = picks[better] + first_active
                best_slopes[better] = (.5 * parent_slope[picks[better]]
                                       + .5 * new_slopes[better, picks[better]])
                # Gaps de pontilhado exigem orientacao medida nos dois lados:
                # nao basta aproximar dois objetos amarelos. Caso raro, escalar.
                for index in np.flatnonzero(~short.any(axis=1) & compatible.any(axis=1)):
                    if evidence[index] < .12:
                        continue
                    upper_tangent = self._forward_tangent(candidates[index],
                                                          rows[row_index + 1:row_index + 5], step)
                    if upper_tangent is None:
                        continue
                    for parent in np.flatnonzero(compatible[index])[::-1]:
                        node = parent + first_active
                        lower_tangent = tail_tangent(node)
                        if (lower_tangent is None or nodes[node, 3] < .12
                                or width_changes[index, parent] > .65):
                            continue
                        if abs(math.atan(upper_tangent) - math.atan(lower_tangent)) > math.radians(35):
                            continue
                        mean_slope = (upper_tangent + lower_tangent) / 2
                        if abs(new_slopes[index, parent] - mean_slope) > .22 + .4 * abs(upper_tangent - lower_tangent):
                            continue
                        if scores[index, parent] > best_scores[index]:
                            best_scores[index] = scores[index, parent]
                            best_parents[index] = node
                            best_slopes[index] = .5 * parent_slope[parent] + .5 * new_slopes[index, parent]
            accepted = np.flatnonzero(np.isfinite(best_scores))
            new = slice(count, count + len(accepted))
            linked = best_parents[accepted]
            nodes[new, :4] = points[accepted]
            nodes[new, 4] = best_scores[accepted]
            nodes[new, 5] = best_slopes[accepted]
            nodes[new, 6] = np.where(linked >= 0, nodes[linked, 6], y)
            lengths[new] = np.where(linked >= 0, lengths[linked] + 1, 1)
            parents[new] = linked
            count += len(accepted)
        # O resultado sempre foi o caminho aceito de maior score. Avaliar
        # nessa ordem permite parar no primeiro, sem validar dezenas de
        # prefixos que jamais serao selecionados. Ordem estavel nos empates.
        spans = nodes[:count, 6] - nodes[:count, 1]
        order = np.argsort(-nodes[:count, 4], kind='stable')
        for node in order[spans[order] >= height * settings.min_span / 100]:
            chain = [node]
            while parents[chain[-1]] >= 0:
                chain.append(parents[chain[-1]])
            points = nodes[chain[::-1], :4]
            span = spans[node]
            long_gap = settings.dashed and np.any(np.abs(np.diff(points[:, 1])) > short_limit)
            if (span / max(1., np.median(points[:, 2])) >= 3.2
                    and len(points) * step / max(span, 1) >= (.28 if long_gap else .5)
                    and np.mean(points[:, 3] >= .12) >= .25):
                return points
        return None

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
