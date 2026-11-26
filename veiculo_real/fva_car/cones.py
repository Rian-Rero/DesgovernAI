"""Corredor de cones observado pela camera BGR calibrada do veiculo real.

Nao usa posicoes do simulador: projeta somente as bases visiveis no piso.
Convencao metrica: frente +x, esquerda +y, origem no eixo dianteiro.
"""
from dataclasses import dataclass, replace, asdict
import math
import time

import cv2
import numpy as np

from .corridor import CorridorEstimator, CorridorSettings, sample_centerline
from .lateral import GroundCalibration, LineObservation, fit_line_errors
from tools.cone_classifier import load_classifier
from tools.cone_geometry import image_homography
from tools.cone_path import ConePathDetector, ConeSettings, _r, normalize_light, render_cones
from tools.cone_tracking import transform_points


# m/s. Em 26/11, 55% dos segundos com vref <= 0,35 "engasgavam" (atrito
# estatico); 0,45 piorou a pista virtual (largura nao aprendida, volta mais
# lenta). Fica 0,35; o engasgo foi tratado no PI de velocidade (car.py).
CONE_RECOVERY_SPEED = .35
# Com o FOV atual, um cone a ~0,9 m de lado so entra na imagem a x >~ 1,6 m:
# no ensaio parado (logs 20261023_080354) o primeiro portao ficava a 1,9-2,3 m
# e o carro nunca partia. Trecho distante so e extrapolado ate o eixo em reta;
# em curva a reta secante inventa erro lateral (quadro 101 de 20261023_080437).
CONE_FAR_START_M = 2.6
CONE_FAR_MAX_CURVATURE = .25  # 1/m, raio de 4 m (mesmo limiar de reta/curva)
# Lado de cada cor para o estimador: verde e vermelho a mesma distancia a frente
# (pares com |dx| < SIDE_PAIR_DX_M, ate SIDE_MAX_RANGE_M) separados de pelo menos
# SIDE_MIN_SEPARATION_M; troca so apos SIDE_FLIP_FRAMES quadros contrarios.
# No ensaio de 10/11 o vermelho estava a esquerda e o detector, sem cadeias,
# ficou no padrao (verde a esquerda): largura "negativa" e erros sem sentido.
SIDE_PAIR_DX_M = .8
SIDE_MAX_RANGE_M = 4.
SIDE_MIN_SEPARATION_M = .3
SIDE_FLIP_FRAMES = 2
# Com o corredor ja estimado, o lado sai das coordenadas do proprio corredor (de
# que lado da linha central cada cor esta), que valem em qualquer curva. Comparar
# cones a mesma distancia a frente falha em curva fechada: a borda de fora da a
# volta pela frente do carro (ensaios de 10/11: 4 trocas erradas em grampos, e o
# carro saiu da pista em todas). Trocar com o corredor estimado exige evidencia
# forte (separacao >= meia largura) por SIDE_FLIP_FRAMES_TRACKING quadros seguidos.
SIDE_FLIP_FRAMES_TRACKING = 6
# Profundidade dos cones longe. O detector roda na imagem reduzida (vision_width,
# 320 px): ali 1 linha vale 6 cm a 2 m e 14 cm a 3 m (a 640 px, metade), e a
# distancia sai da linha da base. Em cones sinteticos com a calibracao real,
# o erro de distancia a 2-3 m foi 5-10 cm (desvio) a 320 px e 1-2 cm a 640 px.
# Por isso a base de cada cone alem de REFINE_MIN_DISTANCE_M e refeita na imagem
# CHEIA, so numa janela em volta dele (mesma segmentacao de cor e mesma regra de
# base do detector): a precisao de 640 px pelo custo de 320 px + ~0,1 ms/cone.
REFINE_MIN_DISTANCE_M = 1.3
REFINE_BASE_M = .15        # largura da base do cone (janela de busca)


def refine_bases(frame, points, color, detector, settings, scale, light=None, ground_from_pixel=None,
                 min_distance=REFINE_MIN_DISTANCE_M):
    """Bases (N, 2) em px da imagem reduzida -> mesmas bases refeitas na imagem cheia.

    frame: BGR na resolucao calibrada; scale: px cheios por px reduzido; color: 0
    verde, 1 vermelho (ordem de ConePathDetector.segment); light: ganhos BGR da
    normalizacao de luz do quadro; ground_from_pixel: homografia da imagem cheia
    (decide quem esta longe e o tamanho da janela). Base sem mascara na janela,
    ou que andaria mais de 1,5 px reduzido, fica como estava. Devolve (bases,
    quantas mudaram).
    """
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points) or scale <= 1 or ground_from_pixel is None:
        return points, 0
    height, width = frame.shape[:2]
    full = (points + .5)*scale - .5
    ground = transform_points(full, ground_from_pixel)
    inverse = np.linalg.inv(ground_from_pixel)
    out, changed = points.copy(), 0
    table = None
    if light is not None and np.any(np.abs(np.asarray(light, float) - 1.) > 1e-3):
        table = np.clip(np.arange(256, dtype=np.float32)[:, None]*np.asarray(light, np.float32)[None, :], 0, 255)
        table = table.astype(np.uint8).reshape(1, 256, 3)
    for k, ((x, y), (gx, gy)) in enumerate(zip(full, ground)):
        if not (np.isfinite(gx) and gx >= min_distance):
            continue
        edges = transform_points(np.array([[gx, gy - REFINE_BASE_M/2], [gx, gy + REFINE_BASE_M/2]]), inverse)
        half = max(3., abs(edges[1, 0] - edges[0, 0])/2)
        x0, x1 = int(max(0, math.floor(x - 1.3*half) - 2)), int(min(width, math.ceil(x + 1.3*half) + 3))
        y0, y1 = int(max(0, math.floor(y - 4*scale) - 2)), int(min(height, math.ceil(y + 3*scale) + 3))
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        crop = frame[y0:y1, x0:x1]
        if table is not None:
            crop = cv2.LUT(crop, table)
        mask = detector.segment(np.ascontiguousarray(crop), settings, roi=0)[color]
        band = mask[:, max(0, int(x - 1.1*half) - x0):int(x + 1.1*half) - x0 + 1] > 0
        widths = np.count_nonzero(band, axis=1)
        rows = np.flatnonzero(widths)
        if len(rows) < 2 or widths.max() < 2:
            continue
        # Mesma regra do detector: base = linha mais larga mais baixa (o reflexo
        # no piso brilhante, abaixo, se estreita).
        base = int(np.flatnonzero(widths >= .85*widths.max()).max())
        columns = np.flatnonzero(band[base]) + max(0, int(x - 1.1*half) - x0)
        new = np.array([x0 + (columns.min() + columns.max())/2, y0 + base], float)
        moved = (new + .5)/scale - .5
        if np.all(np.abs(moved - points[k]) <= 1.5):
            out[k] = moved
            changed += 1
    return out, changed


# So votam no lado de cada cor os cones NA FAIXA de uma borda: distancia a linha
# central entre estas fracoes da largura (a borda fica em 0,5). Numa rotatoria, de
# dentro do anel o carro ve pelas aberturas os cones das vias de entrada e saida
# BEM fora do anel (do lado de fora, a direita, com a cor da esquerda); antes
# eles votavam e trocavam as cores 4-10 vezes por volta na pista virtual.
SIDE_BORDER_BAND = (.2, .85)


def path_separation(green, red, state, max_curvature=2., width=None):
    """Distancia a linha central (esquerda +) mediana do verde - a do vermelho, no corredor `state`.

    So com cones na faixa das bordas (SIDE_BORDER_BAND) da largura `width` (padrao:
    a do estado; o SideTracker passa a nominal, porque um estado montado com as
    cores trocadas tem largura sem sentido); None sem cones assim das duas cores.
    Uma cor so nao vota: com uma borda so, um corredor ja montado com as cores
    trocadas se confirma sozinho (testado na pista virtual: nao ajudou)."""
    from .corridor import offsets
    if width is None or not math.isfinite(width) or width <= 0:
        width = float(state[4]) if len(state) > 4 and math.isfinite(state[4]) and state[4] > 0 else 1.
    low, high = SIDE_BORDER_BAND[0]*width, SIDE_BORDER_BAND[1]*width
    medians = []
    for points in (green, red):
        points = np.asarray(points, float).reshape(-1, 2)
        points = points[(points[:, 0] > .2) & (points[:, 0] <= SIDE_MAX_RANGE_M)] if len(points) else points
        if not len(points):
            return None
        d, along, _, _ = offsets(points, state, max_curvature=max_curvature)
        keep = (along > -.6) & (along < 4.) & (np.abs(d) >= low) & (np.abs(d) <= high)
        if not np.any(keep):
            return None
        medians.append(float(np.median(d[keep])))
    return medians[0] - medians[1]


class SideTracker:
    """Qual cor e a borda esquerda, para o estimador, com histerese.

    Sem corredor estimado: cones das duas cores a mesma distancia a frente
    (side_evidence). Com corredor: lado de cada cor em relacao a linha central.
    """
    def __init__(self):
        self.green_left = None
        self.contrary = 0

    def update(self, green, red, estimator, fallback):
        """(verde a esquerda?, debug, trocou?)."""
        state = None if estimator is None else estimator.x
        if state is not None:
            separation = path_separation(green, red, state, estimator.settings.max_curvature,
                                         getattr(estimator, 'nominal_width', None))
            paired, source = True, 'corredor'
            threshold = max(SIDE_MIN_SEPARATION_M, .5*float(state[4]))
            needed = SIDE_FLIP_FRAMES_TRACKING
        else:
            separation, paired = side_evidence(green, red)
            source, threshold, needed = 'pares', SIDE_MIN_SEPARATION_M, SIDE_FLIP_FRAMES
        vote = None if separation is None or abs(separation) < threshold else separation > 0
        flipped = False
        if self.green_left is None:
            if vote is not None:
                self.green_left = vote
                flipped = vote != fallback  # o estado anterior usou o lado do detector
        elif vote is not None and vote != self.green_left:
            self.contrary += 1
            if self.contrary >= needed:
                self.green_left, flipped = vote, True
        if vote is None or vote == self.green_left or flipped:
            self.contrary = 0
        green_left = fallback if self.green_left is None else self.green_left
        return green_left, {'green_left': bool(green_left), 'separation_m': None if separation is None
                            else round(separation, 3), 'paired': bool(paired), 'flipped': bool(flipped),
                            'source': 'detector' if self.green_left is None else source,
                            'contrary': self.contrary}, flipped


def handle_side_flip(estimator, info, cone_map=None):
    """As cores trocaram de lado: o que fazer com o estimador do corredor.

    Decidido pelo proprio corredor (fonte 'corredor'), a GEOMETRIA continua certa
    e so a cor de cada borda mudou (ilha de rotatoria de outra cor, trecho com as
    cores invertidas): mantem o estado e a largura, so abre a incerteza do
    deslocamento. Antes recomecava do zero: estado PERDIDO, carro parado ate
    juntar cones de novo (o "engasgo"). Sem corredor (pares de cones na partida),
    o estado foi montado com as cores trocadas: recomeca, como antes."""
    if info.get('source') == 'corredor' and estimator.x is not None:
        estimator.relax()
        info['kept_state'] = True
        return False
    estimator.reset()
    estimator.forget_width()
    if cone_map is not None:
        cone_map.reset()
    info['kept_state'] = False
    return True


def side_evidence(green, red, max_range=SIDE_MAX_RANGE_M):
    """Separacao lateral verde - vermelho (m, + = verde a esquerda) e se veio de pares.

    Compara cones das duas cores a mesma distancia a frente (vale em curva);
    sem pares, compara as medianas. None sem as duas cores.
    """
    keep = lambda p: p[(p[:, 0] > .2) & (p[:, 0] <= max_range)] if len(p) else p
    green, red = keep(np.asarray(green, float).reshape(-1, 2)), keep(np.asarray(red, float).reshape(-1, 2))
    if not len(green) or not len(red):
        return None, False
    dx = np.abs(green[:, None, 0] - red[None, :, 0])
    pairs = dx < SIDE_PAIR_DX_M
    if np.any(pairs):
        dy = green[:, None, 1] - red[None, :, 1]
        return float(np.median(dy[pairs])), True
    return float(np.median(green[:, 1]) - np.median(red[:, 1])), False


@dataclass(frozen=True)
class ConeObservation(LineObservation):
    reference_source: str = "cones"
    red_pixels: tuple = ()
    green_pixels: tuple = ()
    center_pixels: tuple = ()
    corridor_width_m: float = math.nan
    cone_result: object = None
    cone_settings: object = None
    corridor: object = None   # CorridorEstimate do quadro (estimador ligado)
    debug: dict | None = None  # Detalhes serializaveis do quadro (cones_debug.jsonl).
    # Placas da ultima busca ArUco, com o horario do quadro que as mediu.
    markers: tuple = ()
    marker_timestamp: float | None = None
    marker_scan: int | None = None



class RealConeDetector:
    """Adaptador metrico do MESMO detector utilizado pela interface HSV."""
    def __init__(self, calibration, settings=None, render_preview=False, processing_width=0, odometry=None,
                 classifier=None, corridor=None, line_fusion=False, refine_bases=True, cone_map=False,
                 aruco_every=0, aruco_marker_m=0., aruco_marker_height_m=None):
        """classifier: caminho do modelo .onnx (com o .json ao lado) ou None.

        corridor: estimador do corredor (fva_car/corridor.py). None/False usa o
        ajuste quadro a quadro; True ou dicionario de CorridorSettings liga o
        filtro, que passa a dar os erros do Stanley.

        Com classificador, os ajustes de cor usados para gerar o conjunto de
        treino (candidate_settings do .json) valem por padrao; chaves dadas
        explicitamente em settings tem prioridade.

        cone_map: mapa de cones (SLAM, fva_car/cone_slam.py) com tracado pelo SVM
        (fva_car/svm_path.py) quando o estimador perde o corredor; "substituir"
        usa o tracado do mapa sempre (experimento). Precisa do estimador.

        aruco_every: procura placas ArUco a cada N quadros, na imagem cheia
        (~1,3 ms a 640 px no notebook); 0 desliga. Entre buscas, o quadro repete
        as placas da ultima busca. aruco_marker_m: lado do quadrado preto do
        marcador (m); com ele, cada placa ganha a posicao no chao (camera_pose).
        """
        if not isinstance(calibration, GroundCalibration):
            raise ValueError('Cones requerem calibracao metrica da camera fixa.')
        self.calibration = calibration
        self.classifier = load_classifier(classifier)
        values = asdict(ConeSettings())
        changes = {} if settings is None else dict(settings)
        if self.classifier is not None:
            changes = {**self.classifier.candidate_settings, **changes}
        if set(changes)-set(values):
            raise ValueError('Ajuste de cones desconhecido: '+str(set(changes)-set(values)))
        values.update(changes)
        for name, value in values.items():
            default = getattr(ConeSettings(), name)
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, float):  # medidas do cone, cm
                valid = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
            else:
                valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
            if not valid:
                raise ValueError('cone_settings.'+name+' tem valor invalido.')
        for name in ('roi_top', 'min_span', 'min_width', 'reference_y', 'camera_center', 'smoothing', 'min_confidence'):
            if values[name] > (99 if name == 'roi_top' else 100):
                raise ValueError('cone_settings.'+name+' fora do intervalo percentual.')
        if not 0 <= values['green_h_min'] <= values['green_h_max'] <= 179 or values['red_hue'] > 89:
            raise ValueError('Limites H dos cones invalidos.')
        if max(values['s_min'], values['v_min'], values['red_s_min'], values['red_v_min'],
               values['floor_b'], values['floor_g'], values['floor_r']) > 255:
            raise ValueError('S e V devem estar entre 0 e 255.')
        if not 0 < values['min_width_cm'] < values['max_width_cm'] or values['min_radius_cm'] <= 0:
            raise ValueError('Limites metricos dos cones invalidos.')
        self.settings = ConeSettings(**values)
        if (isinstance(processing_width, bool) or not isinstance(processing_width, int)
                or processing_width != 0 and not 160 <= processing_width <= calibration.image_size[0]):
            raise ValueError('vision_width deve ser 0 ou largura entre 160 e a resolucao calibrada.')
        w, h = calibration.image_size
        self.processing_size = (w, h) if not processing_width else (processing_width, round(h*processing_width/w))
        self.render_preview = render_preview
        # Base dos cones longe refeita na imagem cheia (refine_bases acima).
        self.refine = bool(refine_bases) and self.processing_size != (w, h)
        self.full_homography = np.asarray(calibration.homography, float)
        self.refined = 0
        # Um unico detector: os lados sao decididos pela geometria, com histerese,
        # sem reprocessar o quadro inteiro com as cores trocadas.
        self.detector = ConePathDetector(calibration, classifier=self.classifier)
        self.green_left = self.settings.green_left
        # PoseHistory (fva_car/odometry.py) do laco de controle: leva o mapa de
        # cones entre quadros; sem ela, o detector usa fluxo optico no piso.
        self.odometry = odometry
        self.last_timestamp = None
        self.corridor = None
        self.sides = SideTracker()  # lado decidido pelos proprios cones (estimador)
        if corridor not in (None, False):
            corridor_settings = CorridorSettings.from_value(
                corridor, track_width_m=self.settings.track_width_cm/100,
                max_range_m=min(3., self.settings.max_range_cm/100))
            self.corridor = CorridorEstimator(
                corridor_settings, image_homography(calibration, self.processing_size[::-1]),
                pixel_scale=self.processing_size[0]/w)
        self.map = None
        if cone_map and self.corridor is not None:
            from .map_path import MapPath
            self.map = MapPath(wheelbase=self.corridor.settings.pose_offset_m,
                               default_width=self.corridor.settings.track_width_m,
                               replace=cone_map == 'substituir')
        # Linha amarela no centro do corredor (se houver): entra no estimador como
        # linha central junto com os cones. Ela fica perto de y = 0, onde a
        # calibracao e boa; os cones (+-0,6 m) dependem da extrapolacao.
        self.line = None
        if line_fusion and self.corridor is not None:
            from .lateral import BirdsEyeView
            from tools.yellow_path import PathSettings, YellowPathDetector
            self.line = (BirdsEyeView(calibration), YellowPathDetector(),
                         PathSettings(roi_top=0, min_span=15, h_max=35, s_min=25, dashed=True, dash_gap=120))
            self.line[1].reset(anchor=.5)
        self.line_frames = 0
        self.aruco_every = int(aruco_every or 0)
        self.aruco = None
        self.aruco_frames = 0
        self.markers = ()
        self.marker_timestamp = None
        self.marker_scan = 0
        self.aruco_marker_m = float(aruco_marker_m or 0.)
        # Altura do CENTRO do marcador (m); None: so pelo tamanho.
        self.aruco_marker_height_m = None if aruco_marker_height_m is None else float(aruco_marker_height_m)
        self.camera_pose = None
        if self.aruco_every > 0:
            from .aruco import ArucoDetector
            from .camera_pose import camera_pose
            self.aruco = ArucoDetector()
            self.camera_pose = camera_pose(self.full_homography, calibration.image_size)

    def _find_markers(self, frame):
        """Busca ArUco a cada aruco_every quadros; None fora dos quadros de busca.
        Com o processo auxiliar, roda no processo de visao, enquanto o auxiliar
        prepara o quadro seguinte: a etapa com estado leva ~3 ms e a sem estado
        ~6 ms, e a busca (~2,6 ms) equilibra as duas."""
        if self.aruco is None:
            return None
        self.aruco_frames += 1
        if (self.aruco_frames - 1) % self.aruco_every:
            return None
        try:
            return tuple(self.aruco.detect(frame))
        except cv2.error:
            return ()

    def _detect_markers(self, found, timestamp):
        """Etapa com estado: posicao das placas achadas neste quadro (se buscou)."""
        if found is None:
            return
        self.marker_timestamp = timestamp
        self.marker_scan += 1
        from dataclasses import replace as with_fields
        self.markers = tuple(with_fields(marker, **self._marker_position(marker)).to_tuple() for marker in found)

    def _marker_position(self, marker):
        """Posicao da placa no chao: pela altura do centro (se dada) ou pelo tamanho.

        Pelo tamanho, uma placa "abaixo do chao" (altura < -3 cm) e impossivel: o
        marcador e menor que marker_m (ensaio de 24/11: -10 cm em todas as
        distancias, ~1,4x longe demais). Ai vale o raio cortado no chao (ou na
        altura dada), e o lado implicito sai no terminal para corrigir marker_m."""
        from .camera_pose import implied_side, marker_at_height, marker_ground
        pixel = (marker.cx, marker.cy)
        point, source = None, ""
        if self.aruco_marker_height_m is not None:
            point, source = marker_at_height(self.camera_pose, self.full_homography, pixel,
                                             self.aruco_marker_height_m), "altura"
        if point is None and self.aruco_marker_m > 0:
            point, source = marker_ground(self.camera_pose, self.full_homography, pixel, marker.side_px,
                                          self.aruco_marker_m), "tamanho"
            if point is not None and point[2] < -.03:
                point, source = marker_at_height(self.camera_pose, self.full_homography, pixel, 0.), "chao"
        if point is None:
            return {}
        return dict(x=point[0], y=point[1], z=point[2], source=source,
                    size_m=implied_side(self.camera_pose, pixel, marker.side_px, point))

    def prepare(self, frame, markers=True):
        """Etapa SEM estado do quadro: reducao, candidatos, classificador e refino
        das bases longe na imagem cheia. So depende deste quadro: roda em outro
        processo enquanto observe_frame() trata o quadro anterior
        (fva_car/vision_process.py). ~57% do tempo de um quadro (perfil de 26/11),
        markers=False (auxiliar): sem a busca ArUco, que fica no observe_frame()."""
        markers = self._find_markers(frame) if markers else False
        started = time.perf_counter()
        small = self._small(frame)
        found = self.detector.detect(small, self.settings)
        refined = None
        if self.refine:
            scale = frame.shape[1]/small.shape[1]
            # Mesma luz arredondada que o debug guardava (refino identico ao de antes).
            light = None if found['light'] is None else [_r(v, 2) for v in found['light']]
            refined = [refine_bases(frame, found[color], index, self.detector, self.settings, scale, light,
                                    self.full_homography)
                       for index, color in enumerate(('green', 'red'))]
        prepared = {'found': found, 'refined': refined, 'seconds': time.perf_counter() - started}
        if markers is not False:
            prepared['markers'] = markers
        return prepared

    def _small(self, frame):
        return (frame if frame.shape[1::-1] == self.processing_size
                else cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA))

    def _unslim(self, found, frame, motion):
        """Desfaz vision_process.slim_prepared: mascaras vazias do mesmo formato
        (o resto so usa o formato, e o envio as descarta) e a imagem reduzida
        com a luz normalizada (as mesmas contas de candidates()) so se o fluxo
        optico for usar (sem odometria)."""
        empty = np.zeros(found['mask_shape'], np.uint8)
        small = None
        if motion is None and self.settings.tracking:
            small = self._small(frame)
            small, _ = normalize_light(small, self.settings, self.detector.roi_row(small.shape[:2], self.settings))
        return dict(found, masks=(empty, empty), frame=small)

    def observe_frame(self, frame, timestamp, prepared=None):
        """prepared: resultado de prepare() deste quadro, ja calculado em paralelo;
        None calcula aqui (mesmo resultado)."""
        metadata = dict(processing_size=self.processing_size)
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            return ConeObservation(timestamp, reason='Horario de captura invalido', **metadata)
        if frame is None:
            return ConeObservation(timestamp, reason='Camera sem imagem', **metadata)
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return ConeObservation(timestamp, reason='Imagem deve ser BGR uint8', **metadata)
        metadata['source_frame'] = frame
        if frame.shape[1::-1] != self.calibration.image_size:
            return ConeObservation(timestamp, reason='Resolucao diferente da calibracao da camera', **metadata)
        if prepared is None:
            prepared = self.prepare(frame)
        if self.aruco is not None:
            found = prepared.get('markers', False)
            self._detect_markers(self._find_markers(frame) if found is False else found, timestamp)
            metadata.update(markers=self.markers, marker_timestamp=self.marker_timestamp,
                            marker_scan=self.marker_scan)
        started = time.perf_counter()
        motion = None
        if self.odometry is not None and self.last_timestamp is not None:
            motion = self.odometry.delta(self.last_timestamp, timestamp)
        self.last_timestamp = timestamp
        if self.corridor is not None:  # largura aprendida (muda ao longo da pista), nao a fixa
            self.detector.nominal_width_m = self.corridor.learned_width
        found = prepared['found']
        if found.get('masks') is None:
            found = self._unslim(found, frame, motion)
        result = self.detector.process(None, self.settings, timestamp=timestamp, motion=motion,
                                       found=found)
        if prepared['refined'] is not None:
            changed = 0
            for name, (points, count) in zip(('green_points', 'red_points'), prepared['refined']):
                setattr(result, name, points)
                changed += count
            self.refined = changed
            result.debug.setdefault('detection', {})['refined_bases'] = changed
        self.green_left = bool(result.green_left)
        # O desenho usa o lado decidido neste quadro.
        settings = replace(self.settings, green_left=self.green_left)
        metadata['detection_seconds'] = prepared['seconds'] + time.perf_counter()-started
        # O detector compartilha a homografia com correcao da resolucao reduzida.
        scale = np.asarray(self.calibration.image_size)/self.processing_size
        original = lambda pixels: tuple(map(tuple, (np.asarray(pixels)+.5)*scale-.5))
        metadata.update(red_pixels=original(result.red_points), green_pixels=original(result.green_points),
                        center_pixels=original(result.center), cone_result=result, cone_settings=settings)
        started = time.perf_counter()
        errors = None
        recovery = result.recovery
        if result.valid and result.source == 'solo':
            try:
                # Stanley precisa dos erros junto ao eixo dianteiro, nao apenas
                # do deslocamento na primeira secao visivel usado na interface.
                errors = fit_line_errors(result.ground_center, [0., 0.], [1., 0.],
                                         with_curvature=True, limit_curve_extrapolation=True)
                if (errors is None and self.settings.single_border_recovery
                        and (recovery or not result.estimated and not result.branch)
                        and np.min(result.ground_center[:, 0]) > 1.5
                        and abs(result.curvature_m_inv or 0.) < CONE_FAR_MAX_CURVATURE):
                    # Um trecho distante e reto ainda da rumo para avancar devagar.
                    errors = fit_line_errors(result.ground_center, [0., 0.], [1., 0.],
                                             with_curvature=True, limit_curve_extrapolation=True,
                                             max_near_distance=CONE_FAR_START_M)
                    recovery = errors is not None
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                pass
        valid = errors is not None and np.all(np.isfinite(errors))
        reason = result.status if valid or not result.valid else 'Cones sem cobertura metrica junto ao eixo dianteiro'
        if valid and recovery and not result.recovery:
            reason = 'Recuperacao de centro por trecho distante de cones'
        legacy = [round(float(e), 4) for e in errors] if valid else None
        estimate = None
        speed_factor = 1.
        path_dkappa = math.nan
        speed_cap = math.inf
        one_border_curve = False
        min_lookahead = 0.
        sigma_e = math.nan
        if self.corridor is not None:
            center = None
            if self.line is not None:
                # Um quadro sim, um nao: a fita e complemento dos cones e a busca
                # custa ~8 ms no notebook (mais que os cones a 320 px).
                self.line_frames += 1
                if self.line_frames % 2:
                    center = self._line_points(frame, timestamp)
            estimate = self._corridor_step(result, timestamp, motion, center)
            valid = estimate.valid
            errors = estimate.errors
            recovery = estimate.slow
            reason = estimate.reason
            speed_factor = estimate.speed_factor if valid else 1.
            if valid:
                path_dkappa = float(estimate.state_vector[3])
                if estimate.reference_state in ('SO_EXTERNA', 'SO_INTERNA', 'UMA_BORDA_RETA'):
                    speed_cap = self.corridor.settings.one_border_speed
                one_border_curve = estimate.state in ('SO_EXTERNA', 'SO_INTERNA')
                sigma_e = float(estimate.sigma[0])
                if estimate.near_uncertain:
                    min_lookahead = self.corridor.settings.check_distance_m
            metadata['corridor'] = estimate
        metadata['geometry_seconds'] = time.perf_counter()-started
        debug = dict(result.debug)
        if self.aruco is not None:
            debug.update(aruco_scan=self.marker_scan, aruco_timestamp=self.marker_timestamp)
            if self.markers:
                debug['aruco'] = [list(m) for m in self.markers]
        if estimate is not None:
            debug.update(corridor=dict(estimate.debug(), sides=self.corridor_sides,
                                       line_points=int(self.corridor.used_center)), legacy_errors=legacy)
            if self.map is not None:
                debug['mapa'] = dict(self.map.last)
                # Cones do mapa perto do carro (inclusive fora da imagem): o filtro
                # do ultrassom (obstacle.py) confere se o eco e um deles.
                near = [p for points in self.map.slam.landmarks().values() for p in points
                        if -.2 < p[0] < 2.5 and abs(p[1]) < 1.5]
                debug['map_cones'] = [[round(float(x), 3), round(float(y), 3)] for x, y in near]
        debug.update(status=reason, valid=bool(valid), estimated=bool(result.estimated),
                     recovery=bool(valid and recovery),
                     confidence=round(float(result.confidence), 3), branch=bool(result.branch),
                     detection_s=round(metadata['detection_seconds'], 4),
                     geometry_s=round(metadata['geometry_seconds'], 4),
                     stanley_errors=[round(float(e), 4) for e in errors] if valid else None)
        if estimate is not None:
            width = estimate.width_m if valid else math.nan
        else:
            width = result.corridor_width_m if valid else math.nan
        observation = ConeObservation(timestamp,
            errors=tuple(float(e) for e in errors) if valid else None,
            speed_limit=CONE_RECOVERY_SPEED if valid and recovery else speed_cap,
            speed_factor=speed_factor, path_dkappa=path_dkappa, one_border_curve=one_border_curve,
            reason=reason, debug=debug, min_lookahead_m=min_lookahead, lateral_sigma_m=sigma_e,
            corridor_width_m=width, **metadata)
        return self.with_preview(observation) if self.render_preview else observation

    def _corridor_sides(self, green, red, fallback):
        """Verde a esquerda? Pela geometria dos cones deste quadro, com histerese."""
        green_left, info, flipped = self.sides.update(green, red, self.corridor, fallback)
        if flipped:
            handle_side_flip(self.corridor, info, self.map)
        return green_left, info

    def _line_points(self, frame, timestamp):
        """Pontos da fita amarela no chao (vista de cima), ou vazio se nao ha fita confiavel."""
        from .lateral import LINE_MIN_SATURATION, line_samples, line_saturation, mask_cones
        view, detector, settings = self.line
        image = view.warp(mask_cones(frame))
        result = detector.process(image, settings, timestamp=timestamp)
        self.line_seen = bool(result.valid) and line_saturation(image, result) >= LINE_MIN_SATURATION
        if not self.line_seen:
            return np.empty((0, 2))
        return line_samples(view.to_ground(result.center))

    def _corridor_step(self, result, timestamp, motion, center=None):
        """Cones deste quadro (ja confirmados pelo classificador) -> estimador."""
        homography = self.corridor.homography
        detector_side = self.green_left if result.green_left is None else bool(result.green_left)
        ground = {}
        for color, points in (('green', result.green_points), ('red', result.red_points)):
            mapped = transform_points(points, homography) if len(points) else np.empty((0, 2))
            ground[color] = mapped[np.all(np.isfinite(mapped), axis=1)]
        green_left, self.corridor_sides = self._corridor_sides(ground['green'], ground['red'], detector_side)
        left, right = (ground['green'], ground['red']) if green_left else (ground['red'], ground['green'])
        pose_frame = motion is not None
        if motion is None:
            # Sem odometria: o deslocamento estimado pelo fluxo optico do piso,
            # ja no referencial dos cones.
            motion = (result.debug.get('tracking') or {}).get('motion_m')
        estimate = self.corridor.step(float(timestamp), left, right, motion, pose_frame, center=center)
        if self.map is not None:
            estimate = self.map.step(float(timestamp), ground['green'], ground['red'], green_left, motion,
                                     pose_frame, estimate, self.corridor._point_covariance)
        return estimate

    def roi_row(self):
        """Primeira linha util na grade de processamento: acima, so o que fica alem
        do alcance dos cones (horizonte). A deteccao ja ignora essa faixa."""
        return self.detector.roi_row(self.processing_size[::-1], self.settings)

    def preview_bgr(self, observation):
        if observation is None or observation.source_frame is None:
            return None
        source = observation.source_frame
        source = (source.copy() if source.shape[1::-1] == tuple(self.processing_size)
                  else cv2.resize(source, self.processing_size, interpolation=cv2.INTER_AREA))
        if observation.cone_result is None:
            return source
        image = render_cones(source, observation.cone_result, observation.cone_settings)
        estimate = getattr(observation, 'corridor', None)
        if estimate is not None and estimate.state_vector is not None and self.corridor is not None:
            # Corredor do estimador: bordas em ciano, centro branco (tracejado sem validade).
            inverse = np.linalg.inv(self.corridor.homography)
            arcs = np.linspace(0., 3., 31)
            vector = estimate.state_vector
            for offset, color in ((vector[4]/2, (255, 255, 0)), (-vector[4]/2, (255, 255, 0)), (0., (255, 255, 255))):
                ground = sample_centerline(vector, arcs, offset)
                ground = ground[ground[:, 0] > .3]
                pixels = transform_points(ground, inverse)
                pixels = pixels[np.all(np.isfinite(pixels) & (np.abs(pixels) < 1e4), axis=1)]
                if len(pixels) < 2:
                    continue
                pixels = pixels.round().astype(np.int32)
                if estimate.valid:
                    cv2.polylines(image, [pixels], False, color, 2, cv2.LINE_AA)
                else:
                    for start in range(0, len(pixels)-1, 4):
                        cv2.polylines(image, [pixels[start:start+2]], False, color, 1, cv2.LINE_AA)
            cv2.putText(image, estimate.state, (6, image.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        # Placas ArUco: quadrado magenta com o ID (coordenadas da imagem cheia).
        scale = image.shape[1]/float(self.calibration.image_size[0])
        for marker_id, cx, cy, side, *_ in getattr(observation, 'markers', ()) or ():
            half = side*scale/2
            top_left = (int(cx*scale - half), int(cy*scale - half))
            bottom_right = (int(cx*scale + half), int(cy*scale + half))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 255), 2)
            cv2.putText(image, f'ID {marker_id}', (top_left[0], max(12, top_left[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1, cv2.LINE_AA)
        return image

    def with_preview(self, observation):
        if observation is None or observation.frame is not None:
            return observation
        image = self.preview_bgr(observation)
        return observation if image is None else replace(observation, frame=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
