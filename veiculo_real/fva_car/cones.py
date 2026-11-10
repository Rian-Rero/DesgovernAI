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
from tools.cone_path import ConePathDetector, ConeSettings, render_cones
from tools.cone_tracking import transform_points


CONE_RECOVERY_SPEED = .35  # m/s
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



class RealConeDetector:
    """Adaptador metrico do MESMO detector utilizado pela interface HSV."""
    def __init__(self, calibration, settings=None, render_preview=False, processing_width=0, odometry=None,
                 classifier=None, corridor=None):
        """classifier: caminho do modelo .onnx (com o .json ao lado) ou None.

        corridor: estimador do corredor (fva_car/corridor.py). None/False usa o
        ajuste quadro a quadro; True ou dicionario de CorridorSettings liga o
        filtro, que passa a dar os erros do Stanley.

        Com classificador, os ajustes de cor usados para gerar o conjunto de
        treino (candidate_settings do .json) valem por padrao; chaves dadas
        explicitamente em settings tem prioridade.
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
        # Um unico detector: os lados sao decididos pela geometria, com histerese,
        # sem reprocessar o quadro inteiro com as cores trocadas.
        self.detector = ConePathDetector(calibration, classifier=self.classifier)
        self.green_left = self.settings.green_left
        # PoseHistory (fva_car/odometry.py) do laco de controle: leva o mapa de
        # cones entre quadros; sem ela, o detector usa fluxo optico no piso.
        self.odometry = odometry
        self.last_timestamp = None
        self.corridor = None
        self.corridor_green_left = None  # lado decidido pelos proprios cones (estimador)
        self.side_contrary = 0
        if corridor not in (None, False):
            corridor_settings = CorridorSettings.from_value(
                corridor, track_width_m=self.settings.track_width_cm/100,
                max_range_m=min(3., self.settings.max_range_cm/100))
            self.corridor = CorridorEstimator(
                corridor_settings, image_homography(calibration, self.processing_size[::-1]),
                pixel_scale=self.processing_size[0]/w)

    def observe_frame(self, frame, timestamp):
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
        started = time.perf_counter()
        small = frame if frame.shape[1::-1] == self.processing_size else cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        motion = None
        if self.odometry is not None and self.last_timestamp is not None:
            motion = self.odometry.delta(self.last_timestamp, timestamp)
        self.last_timestamp = timestamp
        result = self.detector.process(small, self.settings, timestamp=timestamp, motion=motion)
        self.green_left = bool(result.green_left)
        # O desenho usa o lado decidido neste quadro.
        settings = replace(self.settings, green_left=self.green_left)
        metadata['detection_seconds'] = time.perf_counter()-started
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
        if self.corridor is not None:
            estimate = self._corridor_step(result, timestamp, motion)
            valid = estimate.valid
            errors = estimate.errors
            recovery = estimate.slow
            reason = estimate.reason
            speed_factor = estimate.speed_factor if valid else 1.
            if valid:
                path_dkappa = float(estimate.state_vector[3])
                if estimate.state in ('SO_EXTERNA', 'SO_INTERNA', 'UMA_BORDA_RETA'):
                    speed_cap = self.corridor.settings.one_border_speed
                one_border_curve = estimate.state in ('SO_EXTERNA', 'SO_INTERNA')
            metadata['corridor'] = estimate
        metadata['geometry_seconds'] = time.perf_counter()-started
        debug = dict(result.debug)
        if estimate is not None:
            debug.update(corridor=dict(estimate.debug(), sides=self.corridor_sides), legacy_errors=legacy)
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
            reason=reason, debug=debug,
            corridor_width_m=width, **metadata)
        return self.with_preview(observation) if self.render_preview else observation

    def _corridor_sides(self, green, red, fallback):
        """Verde a esquerda? Pela geometria dos cones deste quadro, com histerese."""
        separation, paired = side_evidence(green, red)
        vote = None if separation is None or abs(separation) < SIDE_MIN_SEPARATION_M else separation > 0
        flipped = False
        if self.corridor_green_left is None:
            if vote is not None:
                self.corridor_green_left = vote
                flipped = vote != fallback  # o estado anterior usou o lado do detector
        elif vote is not None and vote != self.corridor_green_left:
            self.side_contrary += 1
            if self.side_contrary >= SIDE_FLIP_FRAMES:
                self.corridor_green_left, flipped = vote, True
        if vote is None or vote == self.corridor_green_left:
            self.side_contrary = 0
        if flipped:
            # Estado montado com as cores trocadas nao serve: recomeca, sem a largura.
            self.side_contrary = 0
            self.corridor.reset()
            self.corridor.learned_width = None
        green_left = fallback if self.corridor_green_left is None else self.corridor_green_left
        return green_left, {'green_left': bool(green_left), 'separation_m': None if separation is None
                            else round(separation, 3), 'paired': bool(paired), 'flipped': bool(flipped),
                            'source': 'detector' if self.corridor_green_left is None else 'cones'}

    def _corridor_step(self, result, timestamp, motion):
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
        return self.corridor.step(float(timestamp), left, right, motion, pose_frame)

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
        return image

    def with_preview(self, observation):
        if observation is None or observation.frame is not None:
            return observation
        image = self.preview_bgr(observation)
        return observation if image is None else replace(observation, frame=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
