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
    from .cone_track_path import (blend_paths, chain_boundary, curve_geometry, fit_path,
                                  gate_midpoints, virtual_midpoints)
else:
    from cone_tracking import ConeTracker, FloorMotion, ego_transform, transform_points
    from cone_geometry import image_homography, metric_corridor, normals, project_curve, virtual_boundary
    from cone_track_path import (blend_paths, chain_boundary, curve_geometry, fit_path,
                                 gate_midpoints, virtual_midpoints)


PURPLE = (230, 40, 210)  # BGR
GREEN = (50, 220, 50)
RED = (50, 50, 240)


def _points():
    return np.empty((0, 2), np.int32)


def floor_color(frame):
    """Mediana BGR da metade de baixo (quase so piso; os cones sao poucos pixels)."""
    sample = frame[frame.shape[0]//2::4, ::4].reshape(-1, 3)
    return np.median(sample, axis=0).astype(np.float32)


def normalize_light(frame, settings, roi=0):
    """(imagem, ganhos BGR) com o piso levado a cor de referencia de settings.

    Ganho comum (brilho) e razao entre canais (cor da luz) separados, cada um
    com zona morta: so o excesso alem de light_band / tint_band % e corrigido,
    e a imagem de referencia passa intacta. Ganho limitado a 0,5-3,5x e cor a
    +-50%, para um piso de outra cor nao virar outra coisa; e o ganho nunca
    estoura os pixels mais claros (percentil 99): canal saturado muda o matiz
    (vermelho estourado vira laranja). roi: linhas acima dela (horizonte, fora
    do alcance dos cones) sao copiadas sem corrigir.
    """
    if not settings.auto_light:
        return frame, None
    reference = np.array([settings.floor_b, settings.floor_g, settings.floor_r], np.float32)
    ratio = np.log(np.maximum(reference, 1.)/np.maximum(floor_color(frame), 1.))
    gain = float(np.mean(ratio))
    tint = ratio - gain
    soft = lambda value, band: np.sign(value)*np.maximum(np.abs(value) - np.log1p(band/100), 0.)
    gain = float(np.clip(soft(gain, settings.light_band), math.log(.5), math.log(3.5)))
    tint = np.clip(soft(tint, settings.tint_band), -math.log(1.5), math.log(1.5))
    scale = np.exp(gain + tint)
    if np.max(scale) > 1.:
        # So pixels coloridos: janela ou reflexo branco estourado nao mudam matiz.
        sample = frame[::4, ::4].reshape(-1, 3).astype(np.int16)
        top, bottom = sample.max(axis=1), sample.min(axis=1)
        colored = top[(top - bottom > .3*top) & (top > 30)]
        if len(colored) >= 20:
            bright = float(np.percentile(colored, 99))
            scale = scale*min(1., max(250./bright, 1.)/float(np.max(scale)))
    if np.all(np.abs(scale - 1.) < 1e-3):
        return frame, scale
    table = np.clip(np.arange(256, dtype=np.float32)[:, None]*scale[None, :], 0, 255)
    table = table.astype(np.uint8).reshape(1, 256, 3)
    roi = int(np.clip(roi, 0, frame.shape[0]-1))
    if not roi:
        return cv2.LUT(frame, table), scale
    output = np.empty_like(frame)
    output[:roi] = frame[:roi]
    output[roi:] = cv2.LUT(frame[roi:], table)
    return output, scale


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
    odometry_memory_ms: int = 2500  # Com odometria o mapa e confiavel por mais tempo.
    confirm_hits: int = 2           # Quadros seguidos para um cone novo entrar nas bordas.
    smoothing: int = 35
    min_confidence: int = 20
    width_memory_ms: int = 1500
    margin_cm: int = 10
    min_radius_cm: int = 30
    min_width_cm: int = 40
    max_width_cm: int = 500
    auto_sides: bool = True   # Decide pela geometria qual cor fica a esquerda.
    side_flip_frames: int = 3  # Quadros seguidos de evidencia contraria para trocar.
    path_prior_ms: int = 1000  # Idade maxima da trajetoria anterior usada como prioridade.
    max_range_cm: int = 300    # Cones usados ate esta distancia; acima do horizonte, ignora.
    cone_height_cm: int = 20   # Cone da prova: define a margem da ROI e o tamanho esperado.
    cone_base_cm: int = 15
    track_width_cm: int = 90   # Largura esperada do corredor (checagem da calibracao).
    cone_spacing_cm: int = 70  # Espacamento tipico entre cones da mesma borda.
    # Matiz em Lab: angulo de (a*, b*) em graus, pouco sensivel ao brilho. No
    # ensaio real: cones vermelhos 14-19 graus, placa 51, porta laranja 65,
    # reflexos no piso 77; cones verdes 153-169; piso sem croma (~2).
    # Vermelho saturado puro fica em ~40 graus; HSV H=8 (quase laranja) em 46.
    lab_filter: bool = True
    chroma_min: int = 12      # porta marrom: croma 7-13 (mediana 9); cones: ~29
    # Verde escuro/azulado na sombra (ensaio de 23/10: croma 4-12, angulo 180-250)
    # fica abaixo do limite do vermelho; o verde tem limite proprio.
    green_chroma_min: int = 12
    partial_cones: bool = True  # Aceita cone cortado so pela lateral, com base visivel.
    red_angle_low: int = 355   # faixa vermelha cruza o zero: 355..360 e 0..47
    red_angle_high: int = 47
    green_angle_low: int = 115  # HSV H=38 (limite verde) ~117 graus
    green_angle_high: int = 200
    hysteresis: bool = True   # Cresce cones fracos (sombra/longe) a partir de pixels fortes.
    size_check: bool = True   # Tamanho metrico do cone (so com calibracao confiavel).
    gate_path: bool = True    # Centro pelos portoes no piso; False: curvas por secoes normais.
    single_border_recovery: bool = False  # Aceita a largura nominal com uma borda recente.
    path_smoothing: int = 40  # % do caminho anterior fundido ao novo (0 desliga).
    min_gap_cm: int = 20      # Espacamento admissivel entre cones da mesma borda.
    max_gap_cm: int = 160
    # Iluminacao: leva o piso (mediana da metade de baixo) a cor de referencia
    # antes da segmentacao, com zona morta. Nos quadros dos ensaios de 23/10,
    # escurecidos/clareados/esquentados/esfriados, recupera 93-95% dos cones
    # (sem: 5-89%); dentro da zona morta a imagem nao muda. floor_* e o piso
    # dos ensaios de 23/10 (BGR); outra pista: tools/ajustar_camera.py --piso.
    auto_light: bool = True
    floor_b: int = 148
    floor_g: int = 154
    floor_r: int = 157
    light_band: int = 15      # % de brilho sem correcao (+-)
    tint_band: int = 2        # % de cor sem correcao (+-)


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
    recovery: bool = False
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
    green_left: bool | None = None       # Lado usado neste quadro (auto ou manual).
    segment: str | None = None           # "reta", "curva_esq" ou "curva_dir".
    curve_ahead_m: float | None = None   # Arco ate |kappa| passar do limiar; 0: ja na curva.
    branch: bool = False                 # Mais de um corredor valido e divergente.
    candidate_centers: list = field(default_factory=list)  # Centros rejeitados (px).
    debug: dict = field(default_factory=dict)  # Serializavel em JSON.

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


def _apexes(component):
    """Apices no perfil superior do blob: cones unidos (lado a lado ou em
    profundidade) mostram dois ou mais picos com vale entre eles."""
    h = component.shape[0]
    filled = np.any(component, axis=0)
    if np.count_nonzero(filled) < 3:
        return 1
    top = np.argmax(component, axis=0).astype(float)
    top[~filled] = h
    prominence = max(3., .12*h)
    # Minimos proeminentes de "top" (linha do topo por coluna) = apices.
    count, low, high, seeking_low = 0, top[0], top[0], True
    for value in top[1:]:
        if seeking_low:
            if value < low:
                low = value
            elif value - low >= prominence:
                count += 1
                seeking_low, high = False, value
        else:
            if value > high:
                high = value
            elif high - value >= prominence:
                seeking_low, low = True, value
    if seeking_low and high - low >= prominence:
        count += 1  # ultimo apice junto a borda direita do blob
    return max(count, 1)


def _grow(strong, weak):
    """Componentes da mascara fraca que contem algum pixel forte."""
    count, labels = cv2.connectedComponents(cv2.bitwise_or(weak, strong), connectivity=8)
    if count <= 1:
        return strong
    keep = np.zeros(count, np.uint8)
    keep[labels[strong > 0]] = 255  # atribuicao repetida dispensa np.unique (ordenacao)
    keep[0] = 0
    return keep[labels]


# Base observada / cone_base_cm nos cones do ensaio de 23/10 (vermelhos: ~0,7).
PARTIAL_BASE_FRACTION = .7
CLASS_GREEN, CLASS_RED = 1, 4  # bits da tabela de matiz/croma Lab
STRAIGHT_MAX_CURVATURE = .25  # 1/m (raio 4 m): mesmo limiar de busca do Stanley.
SIDE_MIN_SEPARATION_M = .15
BRANCH_DIVERGENCE_M = .3
MAX_CORRIDOR_EVALUATIONS = 3  # metric_corridor por quadro: limita custo no Pi 4.


def _r(value, digits=3):
    """Float JSON: arredondado, None para nao finito."""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _polyline(points, count=10, digits=3):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) > count:
        points = points[np.linspace(0, len(points)-1, count).round().astype(int)]
    return [[_r(x, digits), _r(y, digits)] for x, y in points]


class SideEstimator:
    """Decide qual cor fica a esquerda, com histerese entre quadros.

    Evidencia em [-1, 1]: positiva quando o verde esta a esquerda. Somente a
    evidencia com as duas cores (forte) pode trocar um lado ja decidido, e
    apenas apos flip_frames quadros seguidos contrarios.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.green_left = None
        self.score = 0.
        self.contrary = 0
        self.flips = 0

    def update(self, evidence, strong, flip_frames):
        """Retorna True quando trocou de lado neste quadro."""
        if evidence is None or not math.isfinite(evidence) or evidence == 0:
            return False
        weight = .4 if strong else .15
        self.score = float(np.clip((1-weight)*self.score + weight*evidence, -1., 1.))
        vote = evidence > 0
        if self.green_left is None:
            self.green_left = vote
            self.score = float(evidence)
            return False
        if vote == self.green_left:
            self.contrary = 0
            return False
        if not strong:
            return False
        self.contrary += 1
        if self.contrary >= max(1, flip_frames) and (self.score > 0) == vote:
            self.green_left = vote
            self.contrary = 0
            self.flips += 1
            return True
        return False


def _project_visible(points, inverse, shape):
    """Projeta no quadro so os pontos visiveis (o mapa guarda cones fora da imagem)."""
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points):
        return np.empty((0, 2), np.int32)
    pixels = transform_points(points, inverse)
    h, w = shape
    inside = (np.all(np.isfinite(pixels), axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    return np.rint(pixels[inside]).astype(np.int32)


def _distinct(chain, chosen):
    """Cadeias alternativas: nao contidas em outra e com ao menos 2 cones diferentes."""
    a = set(chain)
    for other in chosen:
        b = set(other)
        if a <= b or b <= a or len(a ^ b) < 2:
            return False
    return True


def _curve_profile(center, curvature, step=.25):
    """Curvatura amostrada a cada step m de arco a partir do ponto mais proximo."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    start = int(np.argmin(np.linalg.norm(center, axis=1)))
    arc = arc - arc[start]
    samples = np.arange(0., arc[-1] + 1e-9, step)
    if not len(samples):
        samples = np.array([0.])
    values = np.interp(samples, arc, curvature)
    return samples, values, arc, start


def classify_segment(center, curvature, near_m=.5):
    """Reta ou curva no trecho proximo + distancia ate a proxima curva."""
    samples, values, arc, start = _curve_profile(center, curvature)
    near = values[samples <= near_m]
    mean = float(np.mean(near)) if len(near) else float(values[0])
    if abs(mean) < STRAIGHT_MAX_CURVATURE:
        segment = "reta"
    else:
        segment = "curva_esq" if mean > 0 else "curva_dir"
    ahead = np.flatnonzero(abs(curvature[start:]) >= STRAIGHT_MAX_CURVATURE)
    curve_ahead = float(arc[start + ahead[0]]) if len(ahead) else None
    return segment, mean, curve_ahead, samples, values


class ConePathDetector:
    def __init__(self, calibration=None, classifier=None):
        """classifier: cone_classifier.ConeClassifier opcional (confirma candidatos)."""
        self.clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
        self.calibration = calibration
        self.classifier = classifier
        self.tracker = ConeTracker()
        self.motion = FloorMotion()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.motion.reset()
        self.signature = None
        self.learned_width = None
        self.width_time = -math.inf
        self.sides = SideEstimator()
        self.prev_center = None  # Centro no piso do ultimo corredor valido.
        self.prev_center_time = -math.inf
        self.geometry_samples = []  # (t, espacamento m, largura m) p/ checar a calibracao
        self.calibration_suspect = False
        self.prev_left = self.prev_right = None  # bordas anteriores (sementes)
        self.switch_count = 0

    def set_calibration(self, calibration):
        self.calibration = calibration
        self.reset()

    def roi_row(self, shape, settings):
        """Primeira linha processada: horizonte do alcance util, pela homografia.

        Acima de max_range (mais a altura de um cone ali) so ha parede, portas e
        placas: no ensaio real eram falsos positivos e tempo de processamento.
        Sem calibracao, usa somente roi_top (%).
        """
        height, width = shape
        manual = int(height * settings.roi_top / 100)
        key = (shape, manual, settings.max_range_cm, settings.cone_height_cm, id(self.calibration))
        cached = getattr(self, '_roi_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        row = manual
        if self.calibration is not None:
            try:
                inverse = np.linalg.inv(image_homography(self.calibration, shape))
                far = settings.max_range_cm/100
                pixels = transform_points([[far, -.5], [far, 0.], [far, .5]], inverse)
                if np.all(np.isfinite(pixels)) and np.all((pixels[:, 1] >= 0) & (pixels[:, 1] < height)):
                    # Cone em pe tem altura ~ escala lateral daquela linha (camera baixa).
                    cone = settings.cone_height_cm/100*abs(pixels[2, 0]-pixels[0, 0])
                    row = max(manual, int(np.floor(pixels[:, 1].min() - 1.5*cone)))
            except (ValueError, np.linalg.LinAlgError):
                pass
        row = int(np.clip(row, 0, height-2))
        self._roi_cache = (key, row)
        return row

    def calibration_trusted(self, settings):
        """Calibracao cobre a faixa dos cones e nao foi desmentida em corrida."""
        if getattr(self, 'calibration_suspect', False):
            return False
        points = getattr(self.calibration, 'ground_points_m', None)
        if points is None:
            return True  # so homografia: sem como saber; confia
        return float(np.max(np.asarray(points, float)[:, 0])) >= min(1.5, .6*settings.max_range_cm/100)

    def row_scale(self, shape, settings):
        """Pixels por metro (lateral) em cada linha, pela homografia; None sem calibracao."""
        if self.calibration is None:
            return None
        key = (shape, id(self.calibration))
        cached = getattr(self, '_scale_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        height, width = shape
        scale = np.full(height, np.nan)
        try:
            homography = image_homography(self.calibration, shape)
            inverse = np.linalg.inv(homography)
            rows = np.arange(height, dtype=float)
            ground = transform_points(np.c_[np.full(height, (width-1)/2), rows], homography)
            ahead = np.isfinite(ground[:, 0]) & (ground[:, 0] > .03)
            left = transform_points(ground + [0., .5], inverse)
            right = transform_points(ground - [0., .5], inverse)
            px = np.linalg.norm(left-right, axis=1)
            scale[ahead] = px[ahead]
        except (ValueError, np.linalg.LinAlgError):
            pass
        self._scale_cache = (key, scale)
        return scale

    def _chroma_lut(self, settings):
        """(a*, b*) do Lab uint8 -> bits CLASS_GREEN/CLASS_RED de matiz e croma.

        Mesmo calculo de antes (cartToPolar em float32), feito uma vez para as
        65536 combinacoes: por quadro restam um indice uint16 e uma consulta.
        """
        key = (settings.chroma_min, settings.green_chroma_min, settings.red_angle_low,
               settings.red_angle_high, settings.green_angle_low, settings.green_angle_high)
        cached = getattr(self, '_chroma_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        a, b = np.meshgrid(np.arange(256, dtype=np.float32) - 128,
                           np.arange(256, dtype=np.float32) - 128, indexing='ij')
        magnitude, angle = cv2.cartToPolar(a, b, angleInDegrees=True)
        low_r, high_r = settings.red_angle_low % 360, settings.red_angle_high % 360
        red_hue = (((angle >= low_r) | (angle <= high_r)) if low_r > high_r
                   else ((angle >= low_r) & (angle <= high_r)))
        green_hue = (angle >= settings.green_angle_low) & (angle <= settings.green_angle_high)
        lut = ((green_hue & (magnitude >= settings.green_chroma_min)) * np.uint8(CLASS_GREEN)
               | (red_hue & (magnitude >= settings.chroma_min)) * np.uint8(CLASS_RED)
               ).astype(np.uint8).reshape(-1)
        self._chroma_cache = (key, lut)
        return lut

    def segment(self, frame, settings, roi=None):
        height, width = frame.shape[:2]
        roi = int(height * settings.roi_top / 100) if roi is None else roi
        region = cv2.medianBlur(frame[roi:], 3)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        if settings.clahe:
            hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        low, high = sorted((settings.green_h_min, settings.green_h_max))

        def hsv_masks(factor):
            s_min, v_min = int(settings.s_min*factor), int(settings.v_min*factor)
            red_s, red_v = int(settings.red_s_min*factor), int(settings.red_v_min*factor)
            green = cv2.inRange(hsv, (low, s_min, v_min), (high, 255, 255))
            red = cv2.bitwise_or(
                cv2.inRange(hsv, (0, red_s, red_v), (settings.red_hue, 255, 255)),
                cv2.inRange(hsv, (179-settings.red_hue, red_s, red_v), (179, 255, 255)))
            return green, red

        green, red = hsv_masks(1.)
        green_chroma = red_chroma = None
        if settings.lab_filter:
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            index = (lab[:, :, 1].astype(np.uint16) << 8) | lab[:, :, 2]
            classes = np.take(self._chroma_lut(settings), index)
            green_chroma = cv2.compare(cv2.bitwise_and(classes, CLASS_GREEN), 0, cv2.CMP_GT)
            red_chroma = cv2.compare(cv2.bitwise_and(classes, CLASS_RED), 0, cv2.CMP_GT)
            green, red = cv2.bitwise_and(green, green_chroma), cv2.bitwise_and(red, red_chroma)
        if settings.hysteresis:
            # Mascara fraca (S/V 40% menores, mesma cromaticidade) so sobrevive
            # quando toca pixels fortes: recupera cones na sombra e distantes
            # sem trazer manchas novas.
            weak = hsv_masks(.6)
            if green_chroma is not None:
                weak = (cv2.bitwise_and(weak[0], green_chroma), cv2.bitwise_and(weak[1], red_chroma))
            green, red = (_grow(strong, loose) for strong, loose in zip((green, red), weak))
        masks = []
        for color in (green, red):
            mask = np.zeros((height, width), np.uint8)
            # Fecha pequenos buracos sem juntar cones distantes.
            mask[roi:] = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            masks.append(mask)
        return masks

    @staticmethod
    def _bases(mask, settings, split=True, stats=None, origin=(0, 0), shape=None, roi=None, scale=None,
               size_check=False, _tables=None):
        """stats opcional: conta componentes rejeitados por motivo (sem custo por pixel).

        origin/shape: mask e um recorte em (x, y) de uma imagem com shape (h, w).
        Os criterios usam a geometria da imagem inteira sem alocar a mascara toda.
        scale: pixels por metro (lateral) em cada linha da imagem, pela
        homografia; habilita a area minima por distancia. size_check tambem
        exige o tamanho metrico do cone, so com calibracao confiavel: com a
        calibracao errada do ensaio de 23/10 a escala errava 2-6x.
        """
        ox, oy = origin
        height, width = mask.shape if shape is None else shape
        roi = int(height * settings.roi_top / 100) if roi is None else roi
        count, labels, component_stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        bases = []
        min_area = settings.min_area * height * width / (640 * 480)
        cone_h, cone_w = settings.cone_height_cm/100, settings.cone_base_cm/100
        if _tables is None:
            # (px/m, area minima) por linha da imagem, calculados uma vez por
            # mascara: antes np.clip por componente custava ~10% da deteccao.
            if scale is None:
                px_rows = np.full(height, math.nan)
            else:
                px_rows = np.asarray(scale, float)
                px_rows = np.where(np.isfinite(px_rows) & (px_rows > 0), px_rows, math.nan)
            with np.errstate(invalid='ignore'):
                # Cones longe sao pequenos: a area minima acompanha o tamanho esperado.
                min_rows = np.maximum(4., np.minimum(min_area, .25*.5*cone_h*cone_w*px_rows*px_rows))
            min_rows = np.where(np.isfinite(px_rows), min_rows, min_area)
            _tables = (px_rows, min_rows, px_rows.tolist(), min_rows.tolist())
        px_rows, min_rows, px_list, min_list = _tables
        last_row = len(px_list)-1

        def expected(bottom):
            """(px/m, area minima) para um cone com base na linha bottom."""
            row = min(max(int(bottom), 0), last_row)
            return px_list[row], min_list[row]

        reason = [None]
        def contact(component, x, y):
            rows, columns = np.nonzero(component)
            if not len(rows):
                return None
            left, top, right, bottom = columns.min(), rows.min(), columns.max(), rows.max()
            component = component[top:bottom+1, left:right+1]
            x, y = x+int(left), y+int(top)
            h, w = component.shape
            if h >= 8:
                # Piso brilhante: o reflexo invertido abaixo do cone se estreita.
                # A base e a linha mais larga mais baixa; o que vem abaixo e reflexo.
                widths = np.count_nonzero(component, axis=1)
                base_row = int(np.flatnonzero(widths >= .85*widths.max()).max())
                if base_row <= h-3 and base_row >= 4:
                    component = component[:base_row+1]
                    columns = np.flatnonzero(np.any(component, axis=0))
                    component = component[:, columns.min():columns.max()+1]
                    x += int(columns.min())
                    h, w = component.shape
            area = np.count_nonzero(component)
            px, local_min = expected(y+h-1)
            if area < local_min or h < (4 if math.isfinite(px) else max(5, height/100)) or w < 3:
                reason[0] = reason[0] or "pequeno"
                return None
            # Exige contato inteiro no piso, mesmo ao recuperar uma regiao unida.
            foot = np.flatnonzero(np.any(component[-max(2, h//10):], axis=0))
            at_left, at_right = x+foot.min() <= 0, x+foot.max() >= width-1
            # Cone proximo cortado so pela lateral da imagem: a base (linha de
            # baixo) e vista e a altura confirma o tamanho. Eram os cones mais
            # proximos, descartados como "tamanho"/"cortado_borda" (23/10).
            partial = (settings.partial_cones and (at_left != at_right) and math.isfinite(px)
                       and y > roi and y+h < height and h >= .6*cone_h*px)
            if size_check and math.isfinite(px):
                # Tamanho metrico: base ~cone_base_cm e altura ~cone_height_cm.
                # Largura pelo terco inferior: as 2 linhas de baixo de um cone na
                # sombra sao finas e recusavam cones reais.
                lower = np.flatnonzero(np.any(component[-max(2, h//3):], axis=0))
                foot_m = (lower.max()-lower.min()+1)/px
                foot_ok = foot_m <= 2.2*cone_w and (partial or foot_m >= .4*cone_w)
                if not foot_ok or not .3*cone_h*px <= h <= 2.5*cone_h*px:
                    reason[0] = reason[0] or "tamanho"
                    return None
            if (at_left or at_right) and not partial or y <= roi or y+h >= height:
                reason[0] = reason[0] or "cortado_borda"
                return None
            if partial:
                # Centro da base estimado pela borda visivel e meia base esperada.
                half = .5*cone_w*px*PARTIAL_BASE_FRACTION
                visible = x+foot.max()-half if at_left else x+foot.min()+half
                return float(visible), float(y+h-1)
            if not .18 <= w/h <= 1.5 or w > .2*width:
                reason[0] = reason[0] or "proporcao"
                return None
            # Quarto de cima mais estreito que o de baixo. Com tercos, cones
            # distantes borrados (13 x 23 px no ensaio real) falhavam: 0,92.
            quarter = max(1, h//4)
            top_width = np.count_nonzero(np.any(component[:quarter], axis=0))
            bottom_width = np.count_nonzero(np.any(component[-quarter:], axis=0))
            if top_width >= .85*bottom_width or area < .25*w*h:
                reason[0] = reason[0] or "nao_triangular"
                return None
            return x+float(np.mean(foot)), float(y+h-1)

        # Filtro vetorizado antes do laco em Python: a maioria das componentes
        # e ruido (centenas de "pequeno" por ensaio) e nao chega a contact().
        blobs = component_stats[1:]
        bottoms = np.clip(blobs[:, 1]+blobs[:, 3]-1+oy, 0, last_row)
        small = blobs[:, 4] < min_rows[bottoms]
        # Blob que toca a linha da ROI segue: o cone pode estar na parte de
        # baixo, unido a parede/porta (ensaio real), e a separacao o recupera.
        cut = blobs[:, 1]+oy+blobs[:, 3] >= height
        # Menos de 4 linhas ou 3 colunas: contact() recusaria e nao ha o que separar.
        tiny = (blobs[:, 3] < 4) | (blobs[:, 2] < 3)
        if stats is not None:
            for key, flags in (("pequeno", small | (tiny & ~cut)), ("cortado_borda", cut & ~small)):
                total = int(np.count_nonzero(flags))
                if total:
                    stats[key] = stats.get(key, 0) + total
        for label in (np.flatnonzero(~(small | cut | tiny)) + 1).tolist():
            x, y, w, h, area = component_stats[label].tolist()
            component = labels[y:y+h, x:x+w] == label
            x, y = x+ox, y+oy
            local_min = expected(y+h-1)[1]
            reason[0] = None
            base = contact(component, x, y)
            first_reason = reason[0]
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
            # Separar nucleos custa 8 erosoes + componentes: so quando o blob nao
            # forma um cone (sem base) ou e largo demais para um so (cones lado a
            # lado). Antes rodava em todo blob > 12 px: 67% do tempo no ensaio real.
            if split and w > 12 and h > 12 and (base is None or w > 1.1*h or _apexes(component) >= 2):
                # Separa os nucleos de cones que se tocam. Cada nucleo e
                # restaurado separadamente e limitado a cor realmente observada.
                # A dilatacao conjunta voltaria a fundir os cones.
                local = component.astype(np.uint8)
                # Erosao por quadrado de lado 2r+1 == distancia de tabuleiro > r:
                # uma transformada de distancia serve aos 8 niveis (antes: 8 pad+erode).
                depth = cv2.distanceTransform(np.pad(local, 1), cv2.DIST_C, 3)[1:-1, 1:-1]
                previous = -1
                for nominal in (3, 5, 7, 9, 11, 13, 15, 17):
                    size = max(3, int(round(nominal*height/480)) | 1)
                    kernel = np.ones((size, size), np.uint8)
                    eroded = (depth > size//2).astype(np.uint8)
                    remaining = int(np.count_nonzero(eroded))
                    if remaining < max(4, local_min*.2):
                        break  # niveis maiores so encolhem os nucleos: nenhum seria elegivel
                    if remaining == previous:
                        continue  # mesmos nucleos do nivel anterior: mesmos candidatos
                    previous = remaining
                    n, cores, core_stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
                    eligible = [i for i in range(1, n) if core_stats[i, -1] >= max(4, local_min*.2)]
                    # Sem base, um unico nucleo basta: cone preso a um ruido fino.
                    if not (1 if base is None else 2) <= len(eligible) <= 12:
                        continue
                    candidates = []
                    for i in eligible:
                        restored = cv2.dilate((cores == i).astype(np.uint8), kernel) & local
                        candidates.extend(ConePathDetector._bases(
                            restored*255, settings, split=False, origin=(x, y), shape=(height, width),
                            roi=roi, scale=scale, size_check=size_check, _tables=_tables))
                    if len(candidates) > len(recovered):
                        recovered = candidates
                if len(recovered) >= (1 if base is None else 2):
                    if base is not None:
                        closest = int(np.argmin([np.linalg.norm(np.asarray(p)-base) for p in recovered]))
                        if np.linalg.norm(np.asarray(recovered[closest])-base) < .04*width:
                            recovered[closest] = base
                    bases.extend(recovered)
                    if stats is not None:
                        stats["separados"] = stats.get("separados", 0) + 1
                    continue
            if base is not None:
                bases.append(base)
            elif stats is not None:
                key = first_reason or "sem_base"
                stats[key] = stats.get(key, 0) + 1
        return np.asarray(bases, float).reshape(-1, 2)

    @staticmethod
    def _boundary(points, shape, settings):
        """Seleciona uma cadeia continua por cor, penalizando saltos de direcao."""
        chains = ConePathDetector._chains(points, shape, settings, top_k=1)
        return chains[0][0] if chains else np.empty((0, 2), float)

    @staticmethod
    def _chains(points, shape, settings, top_k=3):
        """Ate top_k cadeias distintas por cor: [(pontos y crescente, pontuacao)].

        A primeira e a mesma de _boundary. As demais sao alternativas (por
        exemplo, os dois ramos de uma bifurcacao) avaliadas pelo corredor.
        """
        if len(points) < 3:
            return []
        height, width = shape
        # Limita custo em cenas com muitas manchas. Prioriza cones proximos.
        points = points[np.argsort(points[:, 1])[-80:]][::-1]
        min_dy = max(2., .004*height)
        max_dy = settings.max_gap*height/480
        # Uma camera baixa comprime a profundidade em poucas linhas da imagem:
        # uma borda valida pode avancar muito em x e pouco em y. Normalizar a
        # inclinacao pela proporcao da nuvem preserva essas curvas sem relaxar
        # a checagem angular em cenas que ja ocupam boa extensao vertical.
        perspective_scale = float(np.clip(
            np.ptp(points[:, 0])/max(np.ptp(points[:, 1]), .03*height), 1., 6.))
        def ranked_chains(index):
            """Cadeias (indices em points, pontuacao) das melhores para as piores."""
            chains, scores, tangents = [], [], []
            for a, i in enumerate(index):
                point = points[i]
                best, score, tangent = [i], 1., None
                for b in range(a):
                    j = index[b]
                    dy = points[j, 1] - point[1]
                    dx = point[0] - points[j, 0]
                    if not min_dy <= dy <= max_dy or abs(dx) > perspective_scale*(1.8*dy + .03*width):
                        continue
                    slope = dx/(dy*perspective_scale)
                    turn = 0. if tangents[b] is None else abs(np.arctan(slope) - np.arctan(tangents[b]))
                    if turn > np.deg2rad(85):
                        continue
                    candidate = scores[b] + 1 + dy/height - .35*turn
                    if tangents[b] is None:
                        # Primeiro elo sem curva anterior: com duas fileiras de cones
                        # da mesma cor, prefere a vizinha na propria fileira em vez de
                        # cruzar para a outra (empate decidido pelo menor salto lateral).
                        candidate -= .05*abs(np.arctan(slope))
                    if candidate > score:
                        best, score, tangent = chains[b] + [i], candidate, slope
                chains.append(best)
                scores.append(score)
                tangents.append(tangent)
            eligible = [a for a, chain in enumerate(chains) if len(chain) >= 3
                        and np.ptp(points[chain, 1]) >= height*settings.min_span/100]
            eligible.sort(key=lambda a: scores[a] + points[chains[a][0], 1]/height, reverse=True)
            return [(chains[a], float(scores[a])) for a in eligible]

        ranked = ranked_chains(list(range(len(points))))
        if not ranked:
            return []
        chosen = [ranked[0]]
        if top_k > 1:
            # Alternativa disjunta: refaz a busca sem os cones da melhor cadeia.
            # Recupera uma segunda fileira da mesma cor (bifurcacao, outro trecho).
            used = set(ranked[0][0])
            rest = [k for k in range(len(points)) if k not in used]
            if len(rest) >= 3:
                disjoint = ranked_chains(rest)
                if disjoint:
                    chosen.append(disjoint[0])
            # Variantes com cones em comum (ramos que partem do mesmo trecho).
            for chain in ranked[1:]:
                if len(chosen) >= top_k:
                    break
                if _distinct(chain[0], [c[0] for c in chosen]):
                    chosen.append(chain)
        # y crescente, sem duplicatas.
        return [(points[chain][::-1], score) for chain, score in chosen[:top_k]]

    def candidates(self, frame, settings):
        """Bases candidatas por cor, antes do classificador e do rastreamento.

        Usado por process() e pela extracao do conjunto de treino
        (extrair_recortes_cones.py): o modelo aprende com os mesmos candidatos
        que vai julgar no carro. 'frame' devolve a imagem com a luz normalizada,
        de onde saem os recortes do classificador.
        """
        roi = self.roi_row(frame.shape[:2], settings)
        frame, light = normalize_light(frame, settings, roi)
        green, red = self.segment(frame, settings, roi)
        rejected = {'green': {}, 'red': {}}
        scale = self.row_scale(frame.shape[:2], settings)
        size_check = settings.size_check and scale is not None and self.calibration_trusted(settings)
        return {
            'green': self._bases(green, settings, stats=rejected['green'], roi=roi, scale=scale,
                                 size_check=size_check),
            'red': self._bases(red, settings, stats=rejected['red'], roi=roi, scale=scale,
                               size_check=size_check),
            'masks': (green, red), 'rejected': rejected, 'roi': roi, 'scale': scale,
            'size_check': size_check, 'frame': frame, 'light': light}

    def process(self, frame, settings=None, timestamp=None, *, motion=None):
        """motion opcional: delta (frente m, esquerda m, yaw rad), desde o frame anterior.

        Com calibracao, aceita odometria externa ou estima movimento pelo piso.
        Sem calibracao, a memoria permanece em pixels e usa fluxo optico.
        """
        settings = settings or ConeSettings()
        now = time.monotonic() if timestamp is None else float(timestamp)
        found = self.candidates(frame, settings)
        green, red = found['masks']
        gp, rp = found['green'], found['red']
        rejected = found['rejected']
        result = ConePathResult(cv2.bitwise_or(green, red), green, red)
        result.green_left = settings.green_left
        detection = {'roi_row': found['roi'], 'size_check': bool(found['size_check'])}
        if found['light'] is not None:
            detection['light'] = [_r(v, 2) for v in found['light']]
        frame = found['frame']
        if self.classifier is not None and found['scale'] is not None:
            # A CNN confirma cada candidato pelo recorte em volta da base.
            started = time.perf_counter()
            verdict = self.classifier.classify(frame, {'green': gp, 'red': rp}, found['scale'],
                                               settings.cone_height_cm/100)
            for color in ('green', 'red'):
                keep = verdict[color][0]
                dropped = int(np.count_nonzero(~keep))
                if dropped:
                    rejected[color]['classificador'] = rejected[color].get('classificador', 0) + dropped
            candidates = len(gp) + len(rp)
            gp, rp = gp[verdict['green'][0]], rp[verdict['red'][0]]
            detection['classifier'] = {'candidates': candidates, 'kept': len(gp) + len(rp),
                                       'ms': _r(1000*(time.perf_counter()-started), 2)}
        result.green_points, result.red_points = gp, rp
        result.debug = {'detection': {
            'green': {'bases': len(gp), 'rejected': rejected['green'], 'px': _polyline(gp, 20, 1)},
            'red': {'bases': len(rp), 'rejected': rejected['red'], 'px': _polyline(rp, 20, 1)},
            **detection}}
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
            if motion is not None:
                self.motion.reset()  # proximo fluxo optico parte de um par novo
                optical = None
            else:
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
            observations = {color: self._ground_points(points, ground_h, settings.max_range_cm/100)
                            for color, points in observations.items()}
            result.source = 'solo'
        odometry = motion is not None and ground_h is not None
        memory = settings.odometry_memory_ms if odometry else settings.memory_ms
        ttl = memory/1000 if settings.tracking else 0.
        if not settings.tracking:
            self.tracker.reset()
        self.tracker.update(observations, now, ttl, matrix=matrix,
                            metric=ground_h is not None, smoothing=settings.smoothing,
                            confirm_hits=settings.confirm_hits if settings.tracking else 1,
                            behind_x=-.4 if odometry else None)
        # Com odometria o mapa guarda cones fora da imagem: a confianca vem do
        # cone visto mais recentemente, nao do mais antigo.
        age = self.tracker.age(now, freshest=odometry)
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
        if ground_h is not None and self.prev_center is not None and matrix is not None:
            # A trajetoria anterior acompanha o carro, como os cones do mapa.
            moved = transform_points(self.prev_center, matrix)
            self.prev_center = moved if np.all(np.isfinite(moved)) else None
            for name in ('prev_left', 'prev_right'):
                boundary = getattr(self, name)
                if boundary is not None:
                    moved = transform_points(boundary, matrix)
                    setattr(self, name, moved if np.all(np.isfinite(moved)) else None)
        self._debug_tracking(result, observations, mapped, matrix, ground_h, now, age)
        top_k = 3 if ground_h is not None else 1
        chains = {color: self._chains(pixel_points[color], green.shape, settings, top_k)
                  for color in ('green', 'red')}
        result.green_left = self._decide_sides(result, chains, mapped, ground_h, green.shape, settings)
        left_chains, right_chains = ((chains['green'], chains['red']) if result.green_left
                                     else (chains['red'], chains['green']))
        result.debug['chains'] = {color: [{'cones': len(c), 'span_px': _r(np.ptp(c[:, 1]), 1),
                                           'score': _r(score), 'chosen': k == 0}
                                          for k, (c, score) in enumerate(items)]
                                  for color, items in chains.items()}
        if ground_h is not None:
            suspect = self._check_calibration(result, chains, ground_h, settings, now)
            gated = (self._gate_result(result, mapped, ground_h, settings, now, observations)
                     if settings.gate_path else None)
            result = gated if gated is not None else self._metric_result(
                result, left_chains, right_chains, ground_h, settings, now, observations)
            if suspect:
                result.status = suspect + " | " + result.status
            return result
        left = left_chains[0][0] if left_chains else np.empty((0, 2), float)
        right = right_chains[0][0] if right_chains else np.empty((0, 2), float)
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
    def _ground_points(points, homography, max_x=10.):
        points = transform_points(points, homography)
        return points[np.all(np.isfinite(points), axis=1) & (points[:, 0] > .03)
                      & (points[:, 0] < max_x) & (abs(points[:, 1]) < 10.)]

    def _debug_tracking(self, result, observations, mapped, matrix, ground_h, now, age):
        detection = result.debug['detection']
        if ground_h is not None:
            for color in ('green', 'red'):
                detection[color]['m'] = _polyline(observations[color], 20)
        tracked = [{'c': 'g' if t.color == 'green' else 'r', 'x': _r(t.point[0]), 'y': _r(t.point[1]),
                    'age_s': _r(now-t.last_seen), 'ok': t.confirmed} for t in self.tracker.tracks[:40]]
        tracking = {'motion': result.motion_source, 'age_s': _r(age), 'confidence': _r(result.confidence),
                    'tracked': tracked,
                    'tentative': sum(not t.confirmed for t in self.tracker.tracks),
                    'tracked_count': {color: len(points) for color, points in mapped.items()}}
        if matrix is not None and ground_h is not None:
            # Delta do carro desde o quadro anterior (aprox. rigido): frente, esquerda, yaw.
            rotation = np.asarray(matrix)[:2, :2]
            yaw = math.atan2(rotation[0, 1], rotation[0, 0])
            delta = -rotation.T @ np.asarray(matrix)[:2, 2]
            tracking['motion_m'] = [_r(delta[0]), _r(delta[1]), _r(yaw, 4)]
        result.debug['tracking'] = tracking

    def _decide_sides(self, result, chains, mapped, ground_h, shape, settings):
        """Qual cor e a borda esquerda: geometria das cadeias + histerese."""
        height, width = shape
        evidence, strong, source, separation = None, False, 'sem evidencia', None
        green = chains['green'][0][0] if chains['green'] else None
        red = chains['red'][0][0] if chains['red'] else None
        if ground_h is not None:
            g = None if green is None else self._ground_points(green, ground_h)
            r = None if red is None else self._ground_points(red, ground_h)
            if g is not None and r is not None and len(g) >= 2 and len(r) >= 2:
                g, r = g[np.argsort(g[:, 0])], r[np.argsort(r[:, 0])]
                start, end = max(g[0, 0], r[0, 0]), min(g[-1, 0], r[-1, 0])
                if end-start >= .1:
                    x = np.linspace(start, end, 8)
                    separation = float(np.median(np.interp(x, g[:, 0], g[:, 1]) - np.interp(x, r[:, 0], r[:, 1])))
                    strong, source = True, 'duas cores'
                else:
                    separation = float(np.median(g[:, 1]) - np.median(r[:, 1]))
                    source = 'duas cores sem faixa comum'
                if abs(separation) >= SIDE_MIN_SEPARATION_M:
                    evidence = float(np.clip(separation/.3, -1., 1.))
                else:
                    strong = False
            else:
                for color, sign in (('green', 1.), ('red', -1.)):
                    points = mapped[color]
                    near = points[(points[:, 0] > 0) & (points[:, 0] < 1.2)] if len(points) else points
                    if len(near) >= 2 and abs(np.median(near[:, 1])) >= SIDE_MIN_SEPARATION_M:
                        separation = float(np.median(near[:, 1]))
                        evidence = sign*math.copysign(.5, separation)
                        source = 'uma cor (' + color + ')'
                        break
        else:
            if green is not None and red is not None:
                start, end = max(green[0, 1], red[0, 1]), min(green[-1, 1], red[-1, 1])
                if end-start >= .03*height:
                    rows = np.linspace(start, end, 8)
                    separation = float(np.median(_interpolate(red, rows) - _interpolate(green, rows)))
                    strong, source = True, 'duas cores'
                else:
                    separation = float(np.median(red[:, 0]) - np.median(green[:, 0]))
                    source = 'duas cores sem faixa comum'
                if abs(separation) >= .05*width:
                    evidence = float(np.clip(separation/(.15*width), -1., 1.))
                else:
                    strong = False
            else:
                axis = (width-1)*settings.camera_center/100
                for color, sign in (('green', 1.), ('red', -1.)):
                    points = mapped[color]
                    near = points[points[:, 1] > .6*height] if len(points) else points
                    if len(near) >= 2 and abs(axis-np.median(near[:, 0])) >= .1*width:
                        separation = float(axis-np.median(near[:, 0]))
                        evidence = sign*math.copysign(.5, separation)
                        source = 'uma cor (' + color + ')'
                        break
        flipped = False
        if settings.auto_sides:
            flipped = self.sides.update(evidence, strong, settings.side_flip_frames)
            green_left = settings.green_left if self.sides.green_left is None else self.sides.green_left
        else:
            green_left = settings.green_left
        result.debug['sides'] = {'auto': bool(settings.auto_sides), 'green_left': bool(green_left),
                                 'score': _r(self.sides.score), 'evidence': _r(evidence),
                                 'separation': _r(separation), 'strong': strong, 'source': source,
                                 'contrary': self.sides.contrary, 'flipped': flipped,
                                 'flips': self.sides.flips}
        return bool(green_left)

    def _check_calibration(self, result, chains, ground_h, settings, now):
        """Espacamento e largura medidos muito longe do esperado por >= 1 s:
        a homografia nao corresponde a camera (ensaio de 23/10: cones a 0,5-1 m
        de distancia entre si apareciam a 3-8 cm, todo o piso comprimido)."""
        spacings = []
        for color in ('green', 'red'):
            if chains[color]:
                ground = self._ground_points(chains[color][0][0], ground_h)
                if len(ground) >= 2:
                    spacings.extend(np.linalg.norm(np.diff(ground, axis=0), axis=1))
        sides = result.debug.get('sides') or {}
        width = abs(sides['separation']) if sides.get('strong') and sides.get('separation') is not None else None
        spacing = float(np.median(spacings)) if spacings else None
        if spacing is not None or width is not None:
            self.geometry_samples.append((now, spacing, width))
        self.geometry_samples = [item for item in self.geometry_samples if now-item[0] <= 2.]
        report = {'spacing_m': _r(spacing), 'width_m': _r(width), 'suspect': False,
                  'expected_spacing_m': settings.cone_spacing_cm/100,
                  'expected_width_m': settings.track_width_cm/100}
        result.debug['calibration'] = report
        samples = self.geometry_samples
        if len(samples) < 5 or samples[-1][0]-samples[0][0] < 1.:
            return None
        spacing_values = [item[1] for item in samples if item[1] is not None]
        width_values = [item[2] for item in samples if item[2] is not None]
        spacing_ratio = (np.median(spacing_values)/(settings.cone_spacing_cm/100)
                         if spacing_values and settings.cone_spacing_cm else None)
        width_ratio = (np.median(width_values)/(settings.track_width_cm/100)
                       if width_values and settings.track_width_cm else None)
        report.update(spacing_ratio=_r(spacing_ratio), width_ratio=_r(width_ratio))
        self.calibration_suspect = False
        # A pista muda de largura de um dia para outro (0,9-2,5 m): so um erro
        # grosseiro (homografia de outra camera: fator ~10) marca a calibracao.
        if ((spacing_ratio is not None and not .35 <= spacing_ratio <= 2.5)
                or (width_ratio is not None and not .4 <= width_ratio <= 2.5)):
            report['suspect'] = True
            self.calibration_suspect = True
            return (f"Calibracao incompativel com a camera? espaco {np.median(spacing_values or [math.nan]):.2f} m"
                    f" (esperado {settings.cone_spacing_cm/100:.2f}), largura"
                    f" {np.median(width_values or [math.nan]):.2f} m (esperada {settings.track_width_cm/100:.2f})")
        return None

    def _continuity(self, center, now, settings):
        """(distancia media m, diferenca de rumo rad) ate a trajetoria anterior, ou None."""
        if (self.prev_center is None or len(self.prev_center) < 3
                or now-self.prev_center_time > settings.path_prior_ms/1000):
            return None
        previous = self.prev_center
        low, high = previous[:, 0].min()-.2, previous[:, 0].max()+.2
        sample = center[(center[:, 0] >= low) & (center[:, 0] <= high)]
        if len(sample) < 3:
            return None
        sample = sample[np.linspace(0, len(sample)-1, min(12, len(sample))).round().astype(int)]
        distance = np.min(np.linalg.norm(sample[:, None, :]-previous[None, :, :], axis=2), axis=1)

        def heading(curve):
            i = int(np.clip(np.argmin(np.linalg.norm(curve, axis=1)), 0, len(curve)-2))
            d = curve[i+1]-curve[i]
            return math.atan2(d[1], d[0])
        turn = abs((heading(center)-heading(previous)+math.pi) % (2*math.pi)-math.pi)
        return float(np.mean(distance)), float(turn)

    def _recovery_border(self, observations, gaps, now, settings):
        """Seleciona a fileira proxima pelo piso, independentemente da cor."""
        candidates = []
        isolated = []
        for color, points in observations.items():
            if not len(points):
                continue
            # Um cone pode sumir de um quadro para o outro. So aproveita
            # vizinhos confirmados da mesma cor vistos ha pouco.
            if len(points) < 2:
                tracks = [t for t in self.tracker.tracks if t.color == color
                          and t.confirmed and now-t.last_seen <= settings.memory_ms/1000]
                if not any(t.last_seen == now for t in tracks):
                    continue
                if len(tracks) == 1:
                    track = tracks[0]
                    x, y = track.point
                    other = observations['red' if color == 'green' else 'green']
                    if (track.hits >= 2 and .35 <= x <= 1.15
                            and max(.12, settings.margin_cm/100) <= abs(y) < settings.track_width_cm/200
                            and not np.any(other[:, 0] <= x+.35)):
                        # Um cone proximo e persistente so informa deslocamento.
                        # Projeta um trecho paralelo ao carro, sem inventar curva.
                        boundary = np.array([track.point, track.point + [.6, 0.]])
                        isolated.append((x, y, color, boundary, True))
                points = np.asarray([t.point for t in tracks])
            boundary = chain_boundary(points, behind=0., **gaps)
            if not len(boundary):
                continue
            near = boundary[0]
            following = boundary[boundary[:, 0]-near[0] >= .30]
            if not len(following):
                continue
            step = following[0] - near
            if not (0 <= near[0] <= 1.65 and abs(step[1]) <= step[0]):
                continue
            normal = np.array([-step[1], step[0]]) / np.linalg.norm(step)
            offset = float(near @ normal)
            candidates.append((near[0], offset, color, np.array([near, following[0]]), False))
        candidates = candidates or isolated
        candidates.sort(key=lambda candidate: candidate[0])
        if not candidates or (len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .35):
            return None
        _, offset, color, boundary, single_cone = candidates[0]
        # Perto do eixo, o lado da fileira nao e observavel com uma cor so.
        if abs(offset) < settings.margin_cm/100:
            return None
        return color, boundary, offset, single_cone

    def _gate_result(self, result, mapped, homography, settings, now, observations):
        """Centro pelos portoes no piso (metodo principal); None para usar o antigo."""
        left_color = 'green' if result.green_left else 'red'
        right_color = 'red' if result.green_left else 'green'
        prior = now-self.prev_center_time <= settings.path_prior_ms/1000
        gaps = dict(min_gap=settings.min_gap_cm/100, max_gap=settings.max_gap_cm/100)
        left = chain_boundary(mapped[left_color], self.prev_left if prior else None, **gaps)
        right = chain_boundary(mapped[right_color], self.prev_right if prior else None, **gaps)
        min_w, max_w = settings.min_width_cm/100, settings.max_width_cm/100
        mids, widths = gate_midpoints(left, right, min_width=min_w, max_width=max_w)
        gates = {'method': 'portoes', 'left_cones': len(left), 'right_cones': len(right),
                 'gates': len(mids), 'prior': bool(prior)}
        result.debug['gates'] = gates
        need = 2 if prior else 3
        virtual = False
        recovery = False
        sparse_gates = (settings.single_border_recovery and len(mids) == 2
                        and all(len(points) >= 2 for points in observations.values())
                        and np.linalg.norm(mids[1]-mids[0]) >= .30)
        if sparse_gates:
            # Dois portoes bem separados ja definem um segmento entre as bordas.
            need = 2
            result.estimated = True
        nominal_width = settings.track_width_cm/100
        if (len(mids) < need and settings.single_border_recovery
                and min_w <= nominal_width <= max_w):
            border = self._recovery_border(observations, gaps, now, settings)
            if border is not None and abs(border[2]) <= max_w/2:
                color, boundary, offset, single_cone = border
                # Sem largura medida, nao aproxima o carro de uma fileira
                # que ja esta mais longe do que a meia largura nominal.
                width = (self.learned_width if self.learned_width is not None
                         and now-self.width_time <= settings.width_memory_ms/1000
                         else min(max_w, max(nominal_width, 2*abs(offset))))
                is_left = offset > 0
                mids = virtual_midpoints(boundary, width, is_left)
                widths = np.full(len(mids), width)
                empty = np.empty((0, 2))
                left, right = (boundary, empty) if is_left else (empty, boundary)
                gates['recovery_color'] = color
                gates['recovery_single_cone'] = single_cone
                virtual = recovery = True
                need = 2
                result.estimated = True
                result.confidence = min(result.confidence, .5)
        if (len(mids) < need and settings.tracking and self.learned_width is not None
                and now-self.width_time <= settings.width_memory_ms/1000):
            width_age = now-self.width_time
            if len(left) >= 2 and len(right) < 2 and len(observations[left_color]) >= 2:
                mids, virtual = virtual_midpoints(left, self.learned_width, True), True
            elif len(right) >= 2 and len(left) < 2 and len(observations[right_color]) >= 2:
                mids, virtual = virtual_midpoints(right, self.learned_width, False), True
            if virtual:
                widths = np.full(len(mids), self.learned_width)
                result.estimated = True
                result.confidence = min(result.confidence, .6*math.exp(-width_age/max(settings.width_memory_ms/1000, 1e-3)))
        gates['virtual'] = virtual
        if len(mids) < need:
            return None
        if result.confidence < settings.min_confidence/100:
            result.status = "Estimativa expirada ou com baixa confianca"
            return result
        fit = fit_path(mids)
        if fit is None:
            return None
        curve, curvature, info = fit
        gates.update(mids=_polyline(mids, 12), outliers=_polyline(info['outlier_points'], 4),
                     fit_rms_m=_r(info['rms_m']), degree=info['degree'],
                     width_m={'min': _r(np.min(widths)), 'median': _r(np.median(widths)), 'max': _r(np.max(widths))})
        center = curve
        continuity = self._continuity(curve, now, settings) if prior and not recovery else None
        gates['raw_center'] = _polyline(curve, 8)
        if continuity is not None and continuity[0] > BRANCH_DIVERGENCE_M:
            # Caminho novo longe do anterior: so troca se persistir (2 quadros).
            self.switch_count += 1
            gates['switch_pending'] = self.switch_count
            if self.switch_count < 2 and self.prev_center is not None and len(self.prev_center) >= 3:
                center = self.prev_center.copy()
                result.estimated = True
        elif not recovery:
            self.switch_count = 0
            if prior and self.prev_center is not None and settings.path_smoothing > 0:
                center, deviation = blend_paths(curve, self.prev_center, settings.path_smoothing/100)
                gates['blend_deviation_m'] = _r(deviation)
        if center is not curve:
            curvature = curve_geometry(center)
        if center is curve or self.switch_count == 0:
            self.switch_count = self.switch_count if center is not curve else 0
        gates['continuity_m'] = None if continuity is None else _r(continuity[0])
        limit = 1/max(settings.min_radius_cm/100, .01)
        curvature = np.clip(curvature, -limit, limit)
        # Ramo alternativo: as bordas sem a semente anterior seguem outro caminho?
        if prior and not recovery:
            for cones, chosen in ((mapped[left_color], left), (mapped[right_color], right)):
                free = chain_boundary(cones, None, **gaps)
                if len(free) >= 2 and len(chosen) >= 2 and np.linalg.norm(free[-1]-chosen[-1]) > BRANCH_DIVERGENCE_M:
                    result.branch = True
        result.debug['corridors'] = {'virtual': virtual, 'candidates': [], 'chosen': None,
                                     'branch': result.branch, 'reason': 'portoes'}
        inverse = np.linalg.inv(homography)
        shape = result.raw_mask.shape
        result.center = _project_visible(center, inverse, shape)
        result.left_boundary = _project_visible(left, inverse, shape)
        result.right_boundary = _project_visible(right, inverse, shape)
        if not len(result.center):
            result.status = "Centro fora da imagem/calibracao"
            return result
        result.ground_center = center
        result.recovery = sparse_gates or settings.single_border_recovery and virtual
        width = float(np.median(widths))
        if not result.estimated and not virtual:
            self.learned_width = width if self.learned_width is None else .7*self.learned_width+.3*width
            self.width_time = now
        if not recovery:
            self.prev_center = center[np.linspace(0, len(center)-1, min(30, len(center))).round().astype(int)].copy()
            self.prev_center_time = now
            if len(left) >= 2:
                self.prev_left = left.copy()
            if len(right) >= 2:
                self.prev_right = right.copy()
        index = int(np.argmin(np.linalg.norm(center, axis=1)))
        normal = normals(center)[index] if len(center) >= 2 else np.array([0., 1.])
        offset = float(center[index] @ normal)
        result.left_distance_m = float(width/2+offset)
        result.right_distance_m = float(width/2-offset)
        result.lateral_error_m = -offset
        tangent = np.array([normal[1], -normal[0]])
        result.heading_rad = float(np.arctan2(tangent[1], tangent[0]))
        result.curvature_m_inv = float(curvature[index])
        result.corridor_width_m = width
        segment, near_curvature, curve_ahead, samples, profile = classify_segment(center, curvature)
        result.segment, result.curve_ahead_m = segment, curve_ahead
        axis = center[index]-offset*normal
        result.measurement_pixels = _project_visible(
            np.array([axis+normal*width/2, axis, axis-normal*width/2]), inverse, shape)
        if len(result.measurement_pixels) != 3:
            result.measurement_pixels = np.empty((0, 2), np.int32)
        result.status = ("Recuperacao de centro por uma borda de cones" if recovery else
                         "Recuperacao de centro por dois portoes de cones" if sparse_gates else
                         "ESTIMADA: uma borda + largura aprendida (m)" if virtual else
                         "ESTIMADA: mapa temporal (m)" if result.estimated else
                         "Centro pelos portoes de cones (m)")
        result.debug['trajectory'] = {
            'segment': segment,
            'radius_m': None if abs(near_curvature) < 1e-3 else _r(1/near_curvature, 2),
            'curvature_near': _r(near_curvature), 'curve_ahead_m': _r(curve_ahead, 2),
            'profile_s_m': [_r(v, 2) for v in samples], 'profile_k': [_r(v) for v in profile],
            'length_m': _r(np.sum(np.linalg.norm(np.diff(center, axis=0), axis=1)), 2),
            'width_m': {'min': _r(np.min(widths)), 'median': _r(width), 'max': _r(np.max(widths)),
                        'learned': _r(self.learned_width)},
            'lateral_error_m': _r(result.lateral_error_m), 'heading_rad': _r(result.heading_rad, 4),
            'curvature_m_inv': _r(result.curvature_m_inv),
            'left_m': _r(result.left_distance_m), 'right_m': _r(result.right_distance_m),
            'center_m': _polyline(center), 'left_border_m': _polyline(left, 8),
            'right_border_m': _polyline(right, 8),
            'estimated': bool(result.estimated), 'virtual': virtual, 'confidence': _r(result.confidence),
            'method': 'portoes'}
        return result

    def _metric_result(self, result, left_chains, right_chains, homography, settings, now, observations):
        # A cadeia da imagem esta do fundo para perto; no piso queremos avancar.
        lefts = [(self._ground_points(c[::-1], homography), score) for c, score in left_chains]
        rights = [(self._ground_points(c[::-1], homography), score) for c, score in right_chains]
        lefts = [item for item in lefts if len(item[0]) >= 3]
        rights = [item for item in rights if len(item[0]) >= 3]
        virtual = False
        if (settings.tracking and (not lefts or not rights)
                and self.learned_width is not None):
            width_age = now-self.width_time
            width_ttl = settings.width_memory_ms/1000
            left_color = 'green' if result.green_left else 'red'
            right_color = 'red' if result.green_left else 'green'
            if width_age <= width_ttl and width_ttl > 0:
                if lefts and len(observations[left_color]) >= 3 and not rights:
                    rights = [(virtual_boundary(l, self.learned_width, True), score) for l, score in lefts]
                    virtual = True
                elif rights and len(observations[right_color]) >= 3 and not lefts:
                    lefts = [(virtual_boundary(r, self.learned_width, False), score) for r, score in rights]
                    virtual = True
                if virtual:
                    result.estimated = True
                    result.confidence = min(result.confidence, .6*math.exp(-width_age/width_ttl))
        corridors = {'virtual': virtual, 'candidates': [], 'chosen': None, 'branch': False, 'reason': None}
        result.debug['corridors'] = corridors
        if not lefts or not rights or any(len(item[0]) < 3 for item in lefts + rights):
            result.status = "Sem curvas suficientes no piso; confira cones e calibracao"
            return result
        if result.confidence < settings.min_confidence/100:
            result.status = "Estimativa expirada ou com baixa confianca"
            return result
        # Pares de bordas candidatas. A borda virtual so combina com a sua origem.
        pairs = ([(i, i) for i in range(len(lefts))] if virtual else
                 [(i, j) for i in range(len(lefts)) for j in range(len(rights))])
        prior = self.prev_center is not None and now-self.prev_center_time <= settings.path_prior_ms/1000
        ordered = []
        for i, j in pairs:
            l, r = lefts[i][0], rights[j][0]
            pre = .3*(i+j)
            lo, hi = max(l[:, 0].min(), r[:, 0].min()), min(l[:, 0].max(), r[:, 0].max())
            if hi-lo >= .2:
                x = np.linspace(lo, hi, 8)
                lorder, rorder = np.argsort(l[:, 0]), np.argsort(r[:, 0])
                yl = np.interp(x, l[lorder, 0], l[lorder, 1])
                yr = np.interp(x, r[rorder, 0], r[rorder, 1])
                width = float(np.median(yl-yr))
                if width <= 0 or width > 2*settings.max_width_cm/100:
                    continue  # Bordas trocadas ou distantes demais: nem avalia.
                if prior:
                    estimate = self._continuity(np.c_[x, (yl+yr)/2], now, settings)
                    pre += 0. if estimate is None else 2*estimate[0]/.3
            ordered.append((pre, i, j))
        ordered.sort()
        valid = []
        for _, i, j in ordered[:MAX_CORRIDOR_EVALUATIONS]:
            geometry = metric_corridor(lefts[i][0], rights[j][0], margin=settings.margin_cm/100,
                                       min_width=settings.min_width_cm/100,
                                       max_width=settings.max_width_cm/100,
                                       min_radius=settings.min_radius_cm/100)
            entry = {'left': i, 'right': j, 'valid': geometry is not None}
            corridors['candidates'].append(entry)
            if geometry is None:
                continue
            center, _, _, widths, _ = geometry
            index = int(np.argmin(np.linalg.norm(center, axis=1)))
            offset = float(center[index] @ normals(center)[index])
            width = float(np.median(widths))
            cost = .3*(i+j)
            continuity = self._continuity(center, now, settings) if prior else None
            if continuity is not None:
                entry['continuity_m'], entry['continuity_rad'] = _r(continuity[0]), _r(continuity[1])
                cost += 2*(continuity[0]/.3 + continuity[1]/.5)
            else:
                # Sem historico: prefere o corredor que contem o carro.
                cost += .5*abs(offset)/max(width, .1)
            if self.learned_width is not None:
                deviation = abs(width-self.learned_width)/max(self.learned_width, .1)
                entry['width_dev'] = _r(deviation)
                cost += .5*deviation
            entry.update(cost=_r(cost), width_m=_r(width), offset_m=_r(-offset),
                         center=_polyline(center, 6))
            valid.append((cost, len(corridors['candidates'])-1, geometry))
        if not valid:
            result.status = "Curvas no piso inconsistentes: largura, margem ou raio minimo"
            return result
        valid.sort(key=lambda item: item[0])
        _, chosen, geometry = valid[0]
        corridors['chosen'] = chosen
        corridors['reason'] = ('unico' if len(valid) == 1 else
                               'continuidade' if corridors['candidates'][chosen].get('continuity_m') is not None
                               else 'sem historico')
        center, left, right, widths, curvature = geometry
        for _, _, other in valid[1:]:
            far = min(center[:, 0].max(), other[0][:, 0].max())
            a = center[np.argmin(abs(center[:, 0]-far))]
            b = other[0][np.argmin(abs(other[0][:, 0]-far))]
            if np.linalg.norm(a-b) > BRANCH_DIVERGENCE_M:
                result.branch = corridors['branch'] = True
                break
        try:
            inverse = np.linalg.inv(homography)
            cp = project_curve(center, inverse, result.raw_mask.shape)
            lp = project_curve(left, inverse, result.raw_mask.shape)
            rp = project_curve(right, inverse, result.raw_mask.shape)
        except (ValueError, np.linalg.LinAlgError) as error:
            result.status = str(error)
            return result
        for _, _, other in valid[1:]:
            try:
                result.candidate_centers.append(project_curve(other[0], inverse, result.raw_mask.shape))
            except ValueError:
                pass
        result.center, result.left_boundary, result.right_boundary = cp, lp, rp
        result.ground_center = center
        result.recovery = settings.single_border_recovery and virtual
        width = float(np.median(widths))
        if not result.estimated and not virtual:
            self.learned_width = width if self.learned_width is None else .7*self.learned_width+.3*width
            self.width_time = now
        self.prev_center = center[np.linspace(0, len(center)-1, min(30, len(center))).round().astype(int)].copy()
        self.prev_center_time = now
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
        segment, near_curvature, curve_ahead, samples, profile = classify_segment(center, curvature)
        result.segment, result.curve_ahead_m = segment, curve_ahead
        axis = center[index]-offset*normal
        try:
            result.measurement_pixels = project_curve(np.array([left[index], axis, right[index]]),
                                                      inverse, result.raw_mask.shape)
        except ValueError:
            pass  # A secao no piso pode cruzar o eixo fora do campo da camera.
        result.status = ("ESTIMADA: uma borda + largura aprendida (m)" if virtual else
                         "ESTIMADA: mapa temporal (m)" if result.estimated else
                         "Centro entre curvas no piso (m)")
        result.debug['trajectory'] = {
            'segment': segment,
            'radius_m': None if abs(near_curvature) < 1e-3 else _r(1/near_curvature, 2),
            'curvature_near': _r(near_curvature), 'curve_ahead_m': _r(curve_ahead, 2),
            'profile_s_m': [_r(v, 2) for v in samples], 'profile_k': [_r(v) for v in profile],
            'length_m': _r(np.sum(np.linalg.norm(np.diff(center, axis=0), axis=1)), 2),
            'width_m': {'min': _r(np.min(widths)), 'median': _r(width), 'max': _r(np.max(widths)),
                        'learned': _r(self.learned_width)},
            'lateral_error_m': _r(result.lateral_error_m), 'heading_rad': _r(result.heading_rad, 4),
            'curvature_m_inv': _r(result.curvature_m_inv),
            'left_m': _r(result.left_distance_m), 'right_m': _r(result.right_distance_m),
            'center_m': _polyline(center), 'left_border_m': _polyline(left, 6),
            'right_border_m': _polyline(right, 6),
            'estimated': bool(result.estimated), 'virtual': virtual, 'confidence': _r(result.confidence)}
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
    green_left = settings.green_left if result.green_left is None else result.green_left
    lc, rc = (GREEN, RED) if green_left else (RED, GREEN)
    cv2.polylines(output, [result.left_boundary], False, lc, 2, cv2.LINE_AA)
    cv2.polylines(output, [result.right_boundary], False, rc, 2, cv2.LINE_AA)
    if len(result.measurement_pixels) == 3:
        left, axis, right = result.measurement_pixels
        cv2.line(output, tuple(axis), tuple(left), lc, 2, cv2.LINE_AA)
        cv2.line(output, tuple(axis), tuple(right), rc, 2, cv2.LINE_AA)
        cv2.drawMarker(output, tuple(axis), (255, 255, 255), cv2.MARKER_CROSS, 17, 2)
    for candidate in result.candidate_centers:
        cv2.polylines(output, [candidate], False, (150, 150, 150), 1, cv2.LINE_AA)
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
