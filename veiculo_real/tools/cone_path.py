"""Cones BGR/HSV -> bordas da pista -> centro em pixels, sem depender da GUI.

Rastreia contatos com o chao em um mapa local com expiracao. Sem calibracao,
interpola x(y) na imagem. Com calibracao, estima o centro entre curvas por
secoes normais no piso e tolera uma borda ausente pela largura aprendida.
Nao envia comandos ao carrinho.
"""

from dataclasses import asdict, dataclass, field
import math
import time

import cv2
import numpy as np

if __package__:
    from .cone_tracking import ConeTracker, FloorMotion, ego_transform, transform_points
    from .cone_geometry import image_homography, metric_corridor, normals, project_curve, virtual_boundary
else:
    from cone_tracking import ConeTracker, FloorMotion, ego_transform, transform_points
    from cone_geometry import image_homography, metric_corridor, normals, project_curve, virtual_boundary


PURPLE = (230, 40, 210)  # BGR
GREEN = (50, 220, 50)
RED = (50, 50, 240)


def _points():
    return np.empty((0, 2), np.int32)


@dataclass
class ConeSettings:
    roi_top: int = 0         # Camera baixa: os cones ficam perto do horizonte.
    min_area: int = 28        # Pixels na escala de 640 x 480.
    max_gap: int = 200        # Distancia vertical maxima, escala de 480 px.
    min_span: int = 5         # Extensao vertical minima, % da altura.
    min_width: int = 8        # Largura minima do corredor, % da imagem.
    reference_y: int = 85     # Altura da secao de medicao, % da imagem.
    camera_center: int = 50   # Coluna alinhada ao eixo do carrinho, %.
    green_left: bool = True
    green_h_min: int = 38
    green_h_max: int = 90
    red_hue: int = 18         # Vermelho nos dois extremos do H (0..179).
    red_s_min: int = 45       # Vermelho escuro/desbotado, independente do verde.
    red_v_min: int = 35
    s_min: int = 65           # Tolera cores desbotadas pela webcam/luz ambiente.
    v_min: int = 45
    clahe: bool = False
    tracking: bool = True
    memory_ms: int = 350
    smoothing: int = 35
    min_confidence: int = 20
    width_memory_ms: int = 1500
    margin_cm: int = 10
    min_radius_cm: int = 30
    min_width_cm: int = 40
    max_width_cm: int = 500


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
    confidence: float = 0.
    estimated: bool = False
    source: str = "imagem"
    motion_source: str = "sem odometria"
    ground_center: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    left_distance_m: float | None = None
    right_distance_m: float | None = None
    lateral_error_m: float | None = None  # Positivo: centro a direita do carro.
    heading_rad: float | None = None     # Positivo: curva orientada a esquerda.
    curvature_m_inv: float | None = None
    corridor_width_m: float | None = None
    measurement_pixels: np.ndarray = field(default_factory=_points)

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
    def __init__(self, calibration=None):
        self.clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
        self.calibration = calibration
        self.tracker = ConeTracker()
        self.motion = FloorMotion()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.motion.reset()
        self.signature = None
        self.learned_width = None
        self.width_time = -math.inf

    def set_calibration(self, calibration):
        self.calibration = calibration
        self.reset()

    def segment(self, frame, settings):
        height, width = frame.shape[:2]
        roi = int(height * settings.roi_top / 100)
        hsv = cv2.cvtColor(cv2.medianBlur(frame[roi:], 3), cv2.COLOR_BGR2HSV)
        if settings.clahe:
            hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        low, high = sorted((settings.green_h_min, settings.green_h_max))
        green = cv2.inRange(hsv, (low, settings.s_min, settings.v_min), (high, 255, 255))
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, settings.red_s_min, settings.red_v_min), (settings.red_hue, 255, 255)),
            cv2.inRange(hsv, (179-settings.red_hue, settings.red_s_min, settings.red_v_min), (179, 255, 255)))
        masks = []
        for color in (green, red):
            mask = np.zeros((height, width), np.uint8)
            # Fecha pequenos buracos sem juntar cones distantes.
            mask[roi:] = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            masks.append(mask)
        return masks

    @staticmethod
    def _bases(mask, settings, split=True):
        height, width = mask.shape
        roi = int(height * settings.roi_top / 100)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        bases = []
        min_area = settings.min_area * height * width / (640 * 480)
        def contact(component, x, y):
            rows, columns = np.nonzero(component)
            if not len(rows):
                return None
            left, top, right, bottom = columns.min(), rows.min(), columns.max(), rows.max()
            component = component[top:bottom+1, left:right+1]
            x, y = x+int(left), y+int(top)
            h, w = component.shape
            area = np.count_nonzero(component)
            if area < min_area or h < max(5, height/100) or w < 3:
                return None
            # Exige contato inteiro no piso, mesmo ao recuperar uma regiao unida.
            foot = np.flatnonzero(np.any(component[-max(2, h//10):], axis=0))
            if (x+foot.min() <= 0 or x+foot.max() >= width-1
                    or y <= roi or y+h >= height):
                return None
            if not .18 <= w/h <= 1.5 or w > .2*width:
                return None
            third = max(1, h//3)
            top_width = np.count_nonzero(np.any(component[:third], axis=0))
            bottom_width = np.count_nonzero(np.any(component[-third:], axis=0))
            if top_width >= .9*bottom_width or area < .25*w*h:
                return None
            return x+float(np.mean(foot)), float(y+h-1)

        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < min_area or y <= roi or y+h >= height:
                continue
            component = labels[y:y+h, x:x+w] == label
            base = contact(component, x, y)
            if base is None:
                # Uma sobreposicao pode unir um cone inteiro a outro cortado na
                # lateral. Tenta somente o cone inferior, cuja base e observada.
                # Nao cria contatos para os cones ocultos do mesmo componente.
                for fraction in (.2, .35, .5):
                    cut = int(h*fraction)
                    lower = component[cut:]
                    n, pieces, piece_stats, _ = cv2.connectedComponentsWithStats(lower.astype(np.uint8), 8)
                    candidates = []
                    for piece in range(1, n):
                        candidate = contact(pieces == piece, x, y+cut)
                        if candidate is not None and candidate[1] == y+h-1:
                            candidates.append(candidate)
                    if candidates:
                        base = max(candidates, key=lambda point: point[1])
                        break
            recovered = []
            if split and w > 12 and h > 12:
                # Separa os nucleos de cones que se tocam. Cada nucleo e
                # restaurado separadamente e limitado a cor realmente observada.
                # A dilatacao conjunta voltaria a fundir os cones.
                local = component.astype(np.uint8)
                for nominal in (3, 5, 7, 9, 11, 13, 15, 17):
                    size = max(3, int(round(nominal*height/480)) | 1)
                    kernel = np.ones((size, size), np.uint8)
                    padded = np.pad(local, size)
                    eroded = cv2.erode(padded, kernel)
                    n, cores, core_stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
                    eligible = [i for i in range(1, n) if core_stats[i, -1] >= max(4, min_area*.2)]
                    if not 2 <= len(eligible) <= 12:
                        continue
                    candidates = []
                    for i in eligible:
                        restored = cv2.dilate((cores == i).astype(np.uint8), kernel)
                        restored = restored[size:size+h, size:size+w] & local
                        full = np.zeros_like(mask)
                        full[y:y+h, x:x+w] = restored*255
                        candidates.extend(ConePathDetector._bases(full, settings, split=False))
                    if len(candidates) > len(recovered):
                        recovered = candidates
                if len(recovered) >= 2:
                    if base is not None:
                        closest = int(np.argmin([np.linalg.norm(np.asarray(p)-base) for p in recovered]))
                        if np.linalg.norm(np.asarray(recovered[closest])-base) < .04*width:
                            recovered[closest] = base
                    bases.extend(recovered)
                    continue
            if base is not None:
                bases.append(base)
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

    def process(self, frame, settings=None, timestamp=None, *, motion=None):
        """motion opcional: delta (frente m, esquerda m, yaw rad), desde o frame anterior.

        Com calibracao, aceita odometria externa ou estima movimento pelo piso.
        Sem calibracao, a memoria permanece em pixels e usa fluxo optico.
        """
        settings = settings or ConeSettings()
        now = time.monotonic() if timestamp is None else float(timestamp)
        green, red = self.segment(frame, settings)
        result = ConePathResult(cv2.bitwise_or(green, red), green, red)
        gp, rp = self._bases(green, settings), self._bases(red, settings)
        result.green_points, result.red_points = gp, rp
        if not math.isfinite(now):
            self.reset()
            result.status = "Horario de captura invalido"
            return result
        signature = (tuple(asdict(settings).items()), green.shape)
        if self.signature != signature or self.tracker.timestamp is not None and now <= self.tracker.timestamp:
            self.reset()
        self.signature = signature
        matrix = None
        try:
            ground_h = None if self.calibration is None else image_homography(self.calibration, green.shape)
            if motion is not None and ground_h is None:
                raise ValueError('Odometria em metros requer calibracao.')
            optical = self.motion.update(frame) if settings.tracking else None
            if motion is not None:
                matrix = ego_transform(motion)
                result.motion_source = 'odometria'
            elif optical is not None:
                matrix = optical if ground_h is None else ground_h @ optical @ np.linalg.inv(ground_h)
                result.motion_source = 'fluxo optico'
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            self.reset()
            result.status = str(error)
            return result
        observations = {'green': gp, 'red': rp}
        if ground_h is not None:
            observations = {color: self._ground_points(points, ground_h) for color, points in observations.items()}
            result.source = 'solo'
        ttl = settings.memory_ms/1000 if settings.tracking else 0.
        if not settings.tracking:
            self.tracker.reset()
        self.tracker.update(observations, now, ttl, matrix=matrix,
                            metric=ground_h is not None, smoothing=settings.smoothing)
        age = self.tracker.age(now)
        result.estimated = age > 1e-6
        result.confidence = math.exp(-age/max(ttl, .001))
        if result.estimated and matrix is None:
            result.confidence *= .7
        mapped = {color: self.tracker.points(color) for color in observations}
        if ground_h is None:
            pixel_points = mapped
        else:
            inverse = np.linalg.inv(ground_h)
            pixel_points = {color: transform_points(points, inverse) for color, points in mapped.items()}
            pixel_points = {color: points[np.all(np.isfinite(points), axis=1)]
                            for color, points in pixel_points.items()}
        left_points, right_points = ((pixel_points['green'], pixel_points['red']) if settings.green_left
                                    else (pixel_points['red'], pixel_points['green']))
        left = self._boundary(left_points, green.shape, settings)
        right = self._boundary(right_points, green.shape, settings)
        if ground_h is not None:
            return self._metric_result(result, left, right, ground_h, settings, now, observations)
        if not len(pixel_points['green']) and not len(pixel_points['red']):
            result.confidence = 0.
            result.status = "Nenhum cone aceito: reduza Ignorar topo e confira HSV cones"
            return result
        if not len(left) or not len(right):
            result.status = "Bordas curtas/descontinuas: confira Extensao minima e Espaco vertical"
            return result
        if result.confidence < settings.min_confidence/100:
            result.status = "Memoria sem confianca suficiente; aguardando cones atuais"
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
        result.measurement_pixels = np.array([[lx[y-start], y], [axis, y], [rx[y-start], y]]).round().astype(np.int32)
        result.status = ("ESTIMADA por memoria (px) | " if result.estimated else "Curvas observadas (px) | ") + (
            "eixo fora do corredor" if min(result.left_distance_px, result.right_distance_px) < 0
            else "centro entre as duas bordas")
        return result

    @staticmethod
    def _ground_points(points, homography):
        points = transform_points(points, homography)
        return points[np.all(np.isfinite(points), axis=1) & (points[:, 0] > .03)
                      & (points[:, 0] < 10.) & (abs(points[:, 1]) < 10.)]

    def _metric_result(self, result, left, right, homography, settings, now, observations):
        # A cadeia da imagem esta do fundo para perto; no piso queremos avancar.
        left = self._ground_points(left[::-1], homography)
        right = self._ground_points(right[::-1], homography)
        virtual = False
        if (settings.tracking and (not len(left) or not len(right))
                and self.learned_width is not None):
            width_age = now-self.width_time
            width_ttl = settings.width_memory_ms/1000
            left_color = 'green' if settings.green_left else 'red'
            right_color = 'red' if settings.green_left else 'green'
            if width_age <= width_ttl and width_ttl > 0:
                if len(left) >= 3 and len(observations[left_color]) >= 3 and not len(right):
                    right = virtual_boundary(left, self.learned_width, True)
                    virtual = True
                elif len(right) >= 3 and len(observations[right_color]) >= 3 and not len(left):
                    left = virtual_boundary(right, self.learned_width, False)
                    virtual = True
                if virtual:
                    result.estimated = True
                    result.confidence = min(result.confidence, .6*math.exp(-width_age/width_ttl))
        if len(left) < 3 or len(right) < 3:
            result.status = "Sem curvas suficientes no piso; confira cones e calibracao"
            return result
        if result.confidence < settings.min_confidence/100:
            result.status = "Estimativa expirada ou com baixa confianca"
            return result
        geometry = metric_corridor(left, right, margin=settings.margin_cm/100,
                                   min_width=settings.min_width_cm/100,
                                   max_width=settings.max_width_cm/100,
                                   min_radius=settings.min_radius_cm/100)
        if geometry is None:
            result.status = "Curvas no piso inconsistentes: largura, margem ou raio minimo"
            return result
        center, left, right, widths, curvature = geometry
        try:
            inverse = np.linalg.inv(homography)
            cp = project_curve(center, inverse, result.raw_mask.shape)
            lp = project_curve(left, inverse, result.raw_mask.shape)
            rp = project_curve(right, inverse, result.raw_mask.shape)
        except (ValueError, np.linalg.LinAlgError) as error:
            result.status = str(error)
            return result
        result.center, result.left_boundary, result.right_boundary = cp, lp, rp
        result.ground_center = center
        width = float(np.median(widths))
        if not result.estimated and not virtual:
            self.learned_width = width if self.learned_width is None else .7*self.learned_width+.3*width
            self.width_time = now
        index = int(np.argmin(np.linalg.norm(center, axis=1)))
        normal = normals(center)[index]
        offset = float(center[index] @ normal)
        result.left_distance_m = float(widths[index]/2+offset)
        result.right_distance_m = float(widths[index]/2-offset)
        result.lateral_error_m = -offset
        tangent = np.array([normal[1], -normal[0]])
        result.heading_rad = float(np.arctan2(tangent[1], tangent[0]))
        result.curvature_m_inv = float(curvature[index])
        result.corridor_width_m = width
        axis = center[index]-offset*normal
        try:
            result.measurement_pixels = project_curve(np.array([left[index], axis, right[index]]),
                                                      inverse, result.raw_mask.shape)
        except ValueError:
            pass  # A secao no piso pode cruzar o eixo fora do campo da camera.
        result.status = ("ESTIMADA: uma borda + largura aprendida (m)" if virtual else
                         "ESTIMADA: mapa temporal (m)" if result.estimated else
                         "Centro entre curvas no piso (m)")
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
    if len(result.measurement_pixels) == 3:
        left, axis, right = result.measurement_pixels
        cv2.line(output, tuple(axis), tuple(left), lc, 2, cv2.LINE_AA)
        cv2.line(output, tuple(axis), tuple(right), rc, 2, cv2.LINE_AA)
        cv2.drawMarker(output, tuple(axis), (255, 255, 255), cv2.MARKER_CROSS, 17, 2)
    if result.estimated:
        for start in range(0, len(result.center)-1, 12):
            cv2.polylines(output, [result.center[start:start+8]], False, PURPLE, 4, cv2.LINE_AA)
    else:
        cv2.polylines(output, [result.center], False, PURPLE, 4, cv2.LINE_AA)
    if result.source == 'solo':
        caption = (f"E:{result.left_distance_m:.2f}m D:{result.right_distance_m:.2f}m "
                   f"Erro:{result.lateral_error_m:+.2f}m")
    else:
        caption = (f"E:{result.left_distance_px:.0f}px D:{result.right_distance_px:.0f}px "
                   f"Erro:{result.lateral_error_px:+.0f}px y={result.reference_y}")
    cv2.putText(output, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    caption = f"{'ESTIMADA' if result.estimated else 'OBSERVADA'}  {result.confidence:.0%}  {result.motion_source}"
    cv2.putText(output, caption, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, caption, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, .45, PURPLE, 1, cv2.LINE_AA)
    return output
