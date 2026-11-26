"""Stanley metrico para o carro real, sem importar sensores ou atuadores.

A homografia leva pixels da camera fixa ao plano do piso: x para frente e y
para esquerda, em metros, com origem no centro do eixo dianteiro. A selecao
da faixa e exatamente a implementacao compartilhada de tools.yellow_path.
"""
from collections import deque
from dataclasses import dataclass, field, fields, replace
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from tools.yellow_path import PathResult, PathSettings, YellowPathDetector, render_path

LINE_RECOVERY_SPEED = .35  # m/s: linha so pela memoria do estimador ou incerta


@dataclass(frozen=True)
class LineObservation:
    timestamp: float
    errors: tuple | None = None
    frame: np.ndarray | None = None  # RGB, apenas para exibicao.
    reason: str = ''
    color_support: float = math.nan  # Fracao do corredor com evidencia de cor.
    dash_fraction: float = 0.0  # Fracao do corredor interpolada entre trechos.
    source_frame: np.ndarray | None = None  # BGR original para diagnostico.
    capture_seconds: float = math.nan
    processing_seconds: float = math.nan
    detection_seconds: float = math.nan
    geometry_seconds: float = math.nan
    path_result: PathResult | None = None
    processing_size: tuple | None = None
    sequence: int = 0
    reference_source: str = "line"
    speed_limit: float = math.inf  # Teto opcional da referencia visual, em m/s.
    speed_factor: float = 1.  # Fracao da vref pela confianca da referencia (estimador de cones).
    # Variacao da curvatura por metro a frente (clotoide do estimador de cones),
    # 1/m^2; com errors, descreve o caminho ate ~3 m para o pure pursuit e para
    # a previa de curva. nan: so a curvatura local (linha amarela).
    path_dkappa: float = math.nan
    # Curva vista por uma cor so (estimador em SO_EXTERNA/SO_INTERNA): a outra
    # borda saiu da imagem; o seguidor vira mais e mais rapido (one_border_curve_*).
    one_border_curve: bool = False
    # Estimador de cones: alvo minimo do pure pursuit (eixo ainda incerto, caminho
    # a frente conhecido) e desvio do deslocamento no eixo, m (alarga o portao de
    # salto enquanto o estimador ainda converge). 0 / nan: sem efeito.
    min_lookahead_m: float = 0.
    lateral_sigma_m: float = math.nan
    # De onde veio o horario da foto: "v4l2" (marcado pelo driver ao chegar o
    # quadro) ou "leitura" (antes do read(), inclui a espera pelo quadro).
    capture_clock: str = ''

    @property
    def valid(self):
        try:
            errors = np.asarray(self.errors, dtype=float)
            return errors.shape == (3,) and bool(np.all(np.isfinite(errors)))
        except (TypeError, ValueError):
            return False


def fit_ground_homography(pixels, ground):
    """Homografia pixel -> chao ajustada no espaco da IMAGEM (chao -> pixel, invertida).

    O ruido esta no clique (pixels), nao nas medidas: ajustar direto pixel -> chao
    minimiza metros, e os pontos longe (1 px = 7 cm a 3 m) puxam o ajuste. Na
    calibracao de 17/11 (sem a fileira de 1,5 m), o erro de cada ponto deixado
    fora do ajuste foi de 6,5 para 3,2 px (mediana) e de 16 para 7,7 px (p90).
    """
    matrix, _ = cv2.findHomography(np.asarray(ground, np.float64), np.asarray(pixels, np.float64), method=0)
    if matrix is None or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
        return None
    return np.linalg.inv(matrix)


@dataclass(frozen=True)
class GroundCalibration:
    image_size: tuple
    homography: np.ndarray
    # Pares medidos (metros), quando a calibracao veio deles: diz ate onde ela vale.
    ground_points_m: tuple | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        try:
            size = np.asarray(self.image_size, dtype=float)
            matrix = np.asarray(self.homography, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('Resolucao e homografia devem conter valores numericos.') from error
        if (size.shape != (2,) or not np.all(np.isfinite(size))
                or np.any(size <= 0) or np.any(size != np.floor(size))):
            raise ValueError('image_size deve conter largura e altura inteiras positivas.')
        if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
                or np.linalg.matrix_rank(matrix) < 3 or np.linalg.cond(matrix) > 1e12):
            raise ValueError('Homografia deve ser uma matriz 3x3 finita e invertivel.')
        matrix = matrix.copy() / np.max(np.abs(matrix))
        matrix.setflags(write=False)
        object.__setattr__(self, 'image_size', tuple(int(x) for x in size))
        object.__setattr__(self, 'homography', matrix)

    @classmethod
    def from_file(cls, path):
        try:
            path = Path(path)
        except (TypeError, ValueError) as error:
            raise ValueError('Informe o caminho do JSON de calibracao da camera.') from error
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            raise FileNotFoundError(f'Calibracao da camera nao encontrada: {path}') from None
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f'Nao foi possivel ler a calibracao {path}: {error}') from error
        if not isinstance(data, dict) or 'image_size' not in data:
            raise ValueError('Calibracao requer image_size e correspondencias pixel/chao.')
        if 'homography' in data:
            return cls(data['image_size'], data['homography'])
        try:
            pixels = np.asarray(data['image_points_px'], dtype=float)
            ground = np.asarray(data['ground_points_m'], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Calibracao requer image_points_px e ground_points_m numericos.') from error
        if (pixels.ndim != 2 or pixels.shape[1:] != (2,) or len(pixels) < 4
                or ground.shape != pixels.shape
                or not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(ground))):
            raise ValueError('Calibracao requer pelo menos 4 pares finitos Nx2, em pixels e metros.')
        normalized = []
        for points in (pixels, ground):
            centered = points - points.mean(axis=0)
            scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
            if scale <= 0 or np.linalg.matrix_rank(centered / scale) < 2:
                raise ValueError('Os pontos de calibracao nao podem ser colineares ou repetidos.')
            normalized.append(centered / scale)
        # Rank oito evita conjuntos de quatro pontos que nao definem uma
        # homografia unica, mesmo quando cada nuvem tem alguma area.
        design = []
        for (u, v), (x, y) in zip(*normalized):
            design.extend(([-u, -v, -1, 0, 0, 0, x*u, x*v, x],
                           [0, 0, 0, -u, -v, -1, y*u, y*v, y]))
        if np.linalg.matrix_rank(np.asarray(design)) < 8:
            raise ValueError('Correspondencias de calibracao degeneradas; distribua pontos no piso.')
        try:
            matrix = fit_ground_homography(pixels, ground)
        except cv2.error as error:
            raise ValueError('Nao foi possivel calcular a homografia da camera.') from error
        if matrix is None:
            raise ValueError('Correspondencias nao definem uma homografia valida.')
        result = cls(data['image_size'], matrix, tuple(map(tuple, ground.tolist())))
        if len(result.project_pixels(pixels)) != len(pixels):
            raise ValueError('A homografia projeta pontos de calibracao para o infinito.')
        return result

    def project_pixels(self, pixels):
        pixels = np.asarray(pixels, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1:] != (2,):
            raise ValueError('Pixels devem ter formato Nx2.')
        homogeneous = np.column_stack((pixels, np.ones(len(pixels)))) @ self.homography.T
        valid = (np.all(np.isfinite(homogeneous), axis=1)
                 & (np.abs(homogeneous[:, 2]) > 1e-12))
        points = homogeneous[valid, :2] / homogeneous[valid, 2, None]
        return points[np.all(np.isfinite(points), axis=1)]


def _path_settings(value):
    if isinstance(value, PathSettings):
        value = {field.name: getattr(value, field.name) for field in fields(PathSettings)}
    try:
        settings = PathSettings(**({} if value is None else value))
    except (TypeError, ValueError) as error:
        raise ValueError('path_settings deve usar os campos de tools.yellow_path.PathSettings.') from error
    limits = {'h_min': (0, 179), 'h_max': (0, 179), 's_min': (0, 255),
              'v_min': (0, 255), 'roi_top': (0, 99), 'max_gap': (0, 480),
              'dash_gap': (0, 480),
              'max_width': (1, 100), 'min_span': (1, 100), 'smoothing': (0, 100)}
    for name, (lower, upper) in limits.items():
        number = getattr(settings, name)
        if (isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, float, np.number)) or not math.isfinite(number)
                or number != int(number) or not lower <= number <= upper):
            raise ValueError(f'path_settings.{name} deve ser inteiro entre {lower} e {upper}.')
        setattr(settings, name, int(number))
    for name in ('clahe', 'dashed'):
        if not isinstance(getattr(settings, name), (bool, np.bool_)):
            raise ValueError(f'path_settings.{name} deve ser booleano.')
    return settings


class BirdsEyeView:
    """Vista de cima do chao a partir da homografia: x de near a far m a frente
    (para cima na imagem), y de +-half_width m (esquerda a esquerda), 1 px =
    resolution m. Com a camera alta (para ver cones), a linha amarela aparece
    deitada e comprimida perto do horizonte; vista de cima ela volta a ficar
    como com a camera inclinada para baixo, onde o detector foi ajustado."""

    def __init__(self, calibration, near=.5, far=3., half_width=1., resolution=.01, edge_px=9):
        """half_width 1 m: a fita fica a +-0,6 m do carro; alem disso entram as
        paredes e portas (madeira, quase amarela). edge_px: faixa descartada na
        borda do campo de visao, onde a borda da imagem vira uma "linha"."""
        homography = np.asarray(calibration.homography, float)
        self.size = (int(round(2*half_width/resolution)), int(round((far-near)/resolution)))
        self.ground_to_view = np.array([[0., -1/resolution, half_width/resolution],
                                        [-1/resolution, 0., far/resolution],
                                        [0., 0., 1.]])
        self.matrix = self.ground_to_view @ homography  # pixel da camera -> pixel da vista
        self.view_to_ground = np.linalg.inv(self.ground_to_view)
        width, height = calibration.image_size
        seen = cv2.warpPerspective(np.full((height, width), 255, np.uint8), self.matrix, self.size,
                                   flags=cv2.INTER_NEAREST)
        self.valid = cv2.erode(seen, np.ones((edge_px, edge_px), np.uint8)) if edge_px else seen

    def warp(self, frame):
        view = cv2.warpPerspective(frame, self.matrix, self.size, flags=cv2.INTER_LINEAR)
        view[self.valid == 0] = 0
        return view

    def to_ground(self, pixels):
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        return (np.c_[pixels, np.ones(len(pixels))] @ self.view_to_ground.T)[:, :2]


def mask_cones(frame, dilate=7):
    """Pinta os cones (verde e vermelho saturados) com a cor do piso, na camera.

    Na vista de cima um cone em pe vira uma faixa longa; na borda dela o verde
    misturado ao piso morno cai no matiz do amarelo e o detector seguia essa
    faixa em vez da fita (ensaios de 10/11)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cones = (cv2.inRange(hsv, (38, 45, 30), (95, 255, 255))
             | cv2.inRange(hsv, (0, 70, 40), (12, 255, 255))
             | cv2.inRange(hsv, (165, 70, 40), (179, 255, 255)))
    if not cv2.countNonZero(cones):
        return frame
    cones = cv2.dilate(cones, np.ones((dilate, dilate), np.uint8))
    floor = np.median(frame[frame.shape[0]//2::4, ::4].reshape(-1, 3), axis=0)
    output = frame.copy()
    output[cones > 0] = floor.astype(np.uint8)
    return output


LINE_MIN_SATURATION = 36  # S mediana ao longo da fita: reflexo do piso ~25-30, fita ~49


def line_saturation(view, result):
    """Saturacao mediana dos pixels amarelos da faixa achada (0 sem pixels)."""
    pixels = (result.path_mask > 0) & (result.raw_mask > 0)
    if not np.any(pixels):
        return 0.
    return float(np.median(cv2.cvtColor(view, cv2.COLOR_BGR2HSV)[..., 1][pixels]))


def line_samples(points, spacing=.15, near=.3, far=3.):
    """Pontos da linha (Nx2, chao) espacados ~spacing m ao longo dela, para o
    estimador: dezenas de pixels vizinhos seriam a mesma medida repetida."""
    points = np.asarray(points, float).reshape(-1, 2)
    points = points[(points[:, 0] > near) & (points[:, 0] <= far)]
    if len(points) < 2:
        return points
    points = points[np.argsort(points[:, 0])]
    keep, last = [0], points[0]
    for index in range(1, len(points)):
        if np.hypot(*(points[index] - last)) >= spacing:
            keep.append(index)
            last = points[index]
    return points[keep]


class RealLineDetector:
    def __init__(self, calibration, path_settings=None, anchor=.5, render_preview=True,
                 processing_width=0, birdseye=False, estimator=None):
        """birdseye: detecta a linha na vista de cima do chao (camera alta, a dos
        cones). estimator: True/dict de CorridorSettings para seguir a linha com
        o estimador do corredor (clotoide + odometria); None usa o ajuste antigo."""
        if not isinstance(calibration, GroundCalibration):
            raise ValueError('RealLineDetector requer GroundCalibration em metros.')
        if not isinstance(anchor, (int, float)) or not math.isfinite(anchor) or not 0 <= anchor <= 1:
            raise ValueError('path_anchor deve estar entre 0 e 1.')
        self.calibration = calibration
        self.settings = _path_settings(path_settings)
        self.anchor = float(anchor)
        self.render_preview = render_preview
        if (isinstance(processing_width, bool) or not isinstance(processing_width, int)
                or processing_width != 0 and not 160 <= processing_width <= calibration.image_size[0]):
            raise ValueError('vision_width deve ser 0 (original) ou inteiro entre 160 e a largura calibrada.')
        width, height = calibration.image_size
        self.processing_size = ((width, height) if processing_width == 0 else
                                (processing_width, max(1, round(height * processing_width / width))))
        self.view = BirdsEyeView(calibration) if birdseye else None
        self.detect_settings = self.settings
        if self.view is not None:
            self.processing_size = self.view.size
            # Na vista de cima nao ha horizonte (roi 0) e a linha visivel e curta
            # em curva fechada (a camera ve so ~0,5 m dela).
            self.detect_settings = replace(self.settings, roi_top=0, min_span=min(self.settings.min_span, 15),
                                           h_max=min(self.settings.h_max, 35))
        self.detector = YellowPathDetector()
        self.detector.reset(anchor=self.anchor)
        self.odometry = None  # PoseHistory do laco de controle (como nos cones)
        self.last_timestamp = None
        self.estimator = None
        if estimator not in (None, False):
            from .corridor import CorridorEstimator, CorridorSettings
            self.estimator = CorridorEstimator(
                CorridorSettings.from_value(estimator, track_width_m=1., max_range_m=3.),
                calibration.homography, 1.)

    def observe_frame(self, frame, timestamp):
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            return LineObservation(timestamp, reason='Horario de captura invalido')
        if frame is None:
            return LineObservation(timestamp, reason='Camera sem imagem')
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return LineObservation(timestamp, reason='Imagem deve ser BGR uint8 com tres canais')
        if (frame.shape[1], frame.shape[0]) != self.calibration.image_size:
            return LineObservation(timestamp, reason='Resolucao diferente da calibracao da camera',
                                   source_frame=frame)
        detection_at = time.perf_counter()
        motion = None
        if self.odometry is not None and self.last_timestamp is not None:
            motion = self.odometry.delta(self.last_timestamp, timestamp)
        self.last_timestamp = timestamp
        processing_frame = frame
        if self.view is not None:
            processing_frame = self.view.warp(mask_cones(frame))
        elif self.processing_size != self.calibration.image_size:
            processing_frame = cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        result = self.detector.process(processing_frame, self.detect_settings, timestamp=timestamp)
        metadata = {'source_frame': frame, 'path_result': result,
                    'processing_size': self.processing_size,
                    'detection_seconds': time.perf_counter() - detection_at}
        display = None
        if self.render_preview:
            display = self._preview(frame, result)
        # A validacao de evidencia pertence ao detector compartilhado. O modo
        # pontilhado pode aceitar suporte < 50% apos validar as duas tangentes.
        detected = (result.valid and math.isfinite(result.support) and 0 <= result.support <= 1)
        if detected and self.view is not None and line_saturation(processing_frame, result) < LINE_MIN_SATURATION:
            # Faixa fraca demais para ser a fita (reflexo, resto de cone): sem
            # resposta; o estimador segue pela memoria. Nos quadros de 10/11:
            # responde em ~53% com 86% de acerto, contra 97% com 47%.
            detected = False
            result.status = 'Faixa fraca (reflexo?): seguindo a memoria'
        if self.estimator is not None:
            return self._estimate(timestamp, result, detected, motion, display, metadata)
        if not detected:
            return LineObservation(timestamp, frame=display, reason=result.status, **metadata)
        geometry_at = time.perf_counter()
        path_pixels = cv2.countNonZero(result.path_mask)
        bridge_pixels = (0 if result.dash_mask is None else
                         cv2.countNonZero(cv2.bitwise_and(result.dash_mask, result.path_mask)))
        metadata.update(color_support=float(result.support),
                        dash_fraction=bridge_pixels / max(1, path_pixels))
        # cv2.resize alinha centros de pixel: volta a grade original ANTES da
        # homografia; na vista de cima, a conversao e direta para o chao.
        points = self._ground_points(result)
        try:
            errors = fit_line_errors(points, [0., 0.], [1., 0.], with_curvature=True)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            errors = None
        metadata['geometry_seconds'] = time.perf_counter() - geometry_at
        if errors is None or not np.all(np.isfinite(errors)):
            return LineObservation(timestamp, frame=display,
                                   reason='Faixa sem cobertura metrica suficiente junto ao eixo dianteiro',
                                   **metadata)
        return LineObservation(timestamp, tuple(float(x) for x in errors), display, result.status,
                               **metadata)

    def _ground_points(self, result):
        pixels = result.center
        if self.view is not None:
            return self.view.to_ground(pixels)
        if self.processing_size != self.calibration.image_size:
            scale = np.asarray(self.calibration.image_size) / self.processing_size
            pixels = (pixels.astype(float) + .5) * scale - .5
        return self.calibration.project_pixels(pixels)

    def _estimate(self, timestamp, result, detected, motion, display, metadata):
        """Linha como linha central do estimador do corredor; segue pela memoria
        (odometria) em quadros sem deteccao."""
        geometry_at = time.perf_counter()
        center = np.empty((0, 2))
        if detected:
            path_pixels = cv2.countNonZero(result.path_mask)
            bridge_pixels = (0 if result.dash_mask is None else
                             cv2.countNonZero(cv2.bitwise_and(result.dash_mask, result.path_mask)))
            metadata.update(color_support=float(result.support),
                            dash_fraction=bridge_pixels / max(1, path_pixels))
            center = line_samples(self._ground_points(result))
        estimate = self.estimator.step(float(timestamp), [], [], motion, motion is not None, center=center)
        metadata['geometry_seconds'] = time.perf_counter() - geometry_at
        if not estimate.valid:
            reason = result.status if not detected else estimate.reason
            return LineObservation(timestamp, frame=display, reason=reason, **metadata)
        return LineObservation(
            timestamp, tuple(float(x) for x in estimate.errors), display, estimate.reason,
            path_dkappa=float(estimate.state_vector[3]), speed_factor=estimate.speed_factor,
            speed_limit=LINE_RECOVERY_SPEED if estimate.slow else math.inf, **metadata)

    def with_preview(self, observation):
        """Desenha sob demanda, sem refazer deteccao nem alterar o horario."""
        if (observation is None or observation.frame is not None
                or observation.source_frame is None or observation.path_result is None):
            return observation
        return replace(observation, frame=self._preview(observation.source_frame, observation.path_result))

    def roi_row(self):
        """Primeira linha usada pelo detector da linha (roi_top), na grade de processamento."""
        return int(self.processing_size[1] * self.detect_settings.roi_top / 100)

    def preview_bgr(self, observation):
        """Deteccao desenhada na grade de processamento, em BGR (transmissao)."""
        if observation is None or observation.source_frame is None:
            return None
        if observation.path_result is None:
            if self.view is not None:
                return self.view.warp(observation.source_frame)
            return cv2.resize(observation.source_frame, self.processing_size, interpolation=cv2.INTER_AREA)
        return self._render_small(observation.source_frame, observation.path_result)

    def _render_small(self, frame, result):
        size = result.path_mask.shape[1::-1]
        if self.view is not None:
            source = self.view.warp(frame)  # transmissao mostra a vista de cima
        else:
            source = frame if frame.shape[1::-1] == size else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return render_path(source, result, self.detect_settings, self.anchor)

    def _preview(self, frame, result):
        display = self._render_small(frame, result)
        if display.shape[:2] != frame.shape[:2]:
            display = cv2.resize(display, frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


class StanleyController:
    def __init__(self, gain=1.5, softening=0.3, max_steer=math.radians(20),
                 max_rate=math.radians(60), wheelbase=.36, curvature_gain=0., yaw_damping=0.):
        if not all(math.isfinite(x) and x > 0
                   for x in (gain, softening, max_steer, max_rate, wheelbase)):
            raise ValueError('Parametros Stanley devem ser finitos e positivos.')
        if max_steer >= math.pi / 2:
            raise ValueError('max_steer deve ser menor que pi/2 radianos.')
        if not math.isfinite(curvature_gain) or curvature_gain < 0:
            raise ValueError('O ganho de curvatura deve ser finito e nao negativo.')
        if not math.isfinite(yaw_damping) or yaw_damping < 0:
            raise ValueError('O amortecimento de guinada deve ser finito e nao negativo.')
        # Amortecimento pela guinada MEDIDA (giroscopio), em s: esterco -=
        # k*(guinada medida - v*curvatura do caminho). A imagem chega ~220 ms
        # atrasada; o giroscopio ve o carro comecando a oscilar na hora
        # (Stanley para carros RC de corrida, arXiv 2408.15152). 0 desliga;
        # sem giroscopio valido o termo e zero.
        self.yaw_damping = yaw_damping
        self.gain, self.softening = gain, softening
        self.max_steer, self.max_rate = max_steer, max_rate
        self.wheelbase, self.curvature_gain = wheelbase, curvature_gain
        self.gain_scale = self.rate_scale = 1.  # reforco temporario (curva por uma cor)
        self.trim = 0.  # correcao integral de centralizacao (rad), dada pelo seguidor
        self.reset()

    def reset(self):
        self.steer = 0.
        self.heading_term = 0.
        self.cross_track_term = 0.
        self.feedforward_term = 0.
        self.damping_term = 0.
        self.search_term = 0.
        self.raw_steer = 0.
        self.target_steer = 0.

    def update(self, cross_track, heading_error, speed, dt, curvature=0., search=0.,
               yaw_rate=math.nan, steering_factor=1.):
        """cross_track: linha a esquerda do eixo dianteiro (+), em metros.

        heading_error = direcao da linha - direcao do carro, em radianos.
        curvature: da linha, positiva para a esquerda, em 1/m.
        search: esterco extra (rad) para reencontrar uma faixa perdida.
        yaw_rate: guinada medida pelo giroscopio (rad/s, esquerda +); nan sem IMU.
        """
        if (not all(math.isfinite(x) for x in (cross_track, heading_error, speed, dt, curvature, search))
                or dt <= 0):
            raise ValueError('Stanley requer sinais finitos e dt positivo.')
        self.heading_term = math.atan2(math.sin(heading_error), math.cos(heading_error))
        self.cross_track_term = math.atan2(self.gain * cross_track,
                                          abs(speed) + self.softening)
        # Esterco de regime da curva (bicicleta): sem ele o Stanley so vira
        # depois de acumular erro, tarde demais em velocidade alta.
        self.feedforward_term = math.atan(self.curvature_gain * self.wheelbase * curvature)
        self.search_term = search
        self.damping_term = 0.
        if self.yaw_damping > 0 and math.isfinite(yaw_rate):
            # Guinada que o caminho pede no eixo: v*curvatura.
            self.damping_term = float(np.clip(-self.yaw_damping*(yaw_rate - speed*curvature),
                                              -self.max_steer/2, self.max_steer/2))
        return self.steer_to(self.heading_term + self.cross_track_term + self.feedforward_term
                             + self.damping_term + self.search_term, dt, self.gain_scale,
                             self.rate_scale, steering_factor)

    def steer_to(self, raw_steer, dt, gain=1., rate=1., steering_factor=1.):
        """Satura no batente e limita a taxa (comum ao Stanley e ao pure pursuit).

        gain multiplica o esterco pedido; rate, a taxa maxima do volante; trim
        (correcao de centralizacao, rad) e somado depois.
        """
        self.raw_steer = float(raw_steer)*gain + self.trim
        self.target_steer = steering_factor*float(np.clip(self.raw_steer, -self.max_steer, self.max_steer))
        limit = self.max_rate*rate*dt
        self.steer += float(np.clip(self.target_steer - self.steer, -limit, limit))
        return self.steer



def fit_line_errors(points, front, forward, max_distance=4., with_curvature=False,
                    limit_curve_extrapolation=False, max_near_distance=1.5):
    """Erros na projecao normal do eixo dianteiro sobre a curva local."""
    forward = np.asarray(forward, dtype=float)[:2]
    forward = forward / np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    relative = np.asarray(points)[:, :2] - np.asarray(front)[:2]
    x, y = relative @ forward, relative @ left
    valid = (x >= -.35) & (x <= max_distance) & (np.abs(y) <= 2.)
    x, y = x[valid], y[valid]
    if len(x) < 6 or np.ptp(x) < .3 or np.min(x) > max_near_distance:
        return None
    # Aproximacao local; nao extrapolar uma curva distante ate o veiculo.
    nearby = x <= min(max_distance, max(1.2, float(np.min(x)) + 1.))
    x, y = x[nearby], y[nearby]
    if len(x) < 6 or np.ptp(x) < .3:
        return None
    # Reta secante em uma curva inventava deslocamento lateral em x=0.
    # Parabola separa deslocamento, tangente e curvatura da linha.
    degree = 1 if limit_curve_extrapolation and np.min(x) > np.ptp(x) else 2
    coefficients = np.polyfit(x, y, degree)
    residual = y - np.polyval(coefficients, x)
    keep = np.abs(residual) <= max(.04, 3 * float(np.median(np.abs(residual))))
    if np.count_nonzero(keep) < 6 or np.ptp(x[keep]) < .3:
        return None
    if limit_curve_extrapolation and np.min(x[keep]) > np.ptp(x[keep]):
        degree = 1
    coefficients = np.polyfit(x[keep], y[keep], degree)
    if np.sqrt(np.mean((y[keep] - np.polyval(coefficients, x[keep]))**2)) > .12:
        return None
    if degree == 1:
        slope, intercept = coefficients
        errors = (float(intercept / math.hypot(1., slope)), math.atan(slope))
        return (*errors, 0.) if with_curvature else errors
    a, slope, intercept = coefficients
    # Stanley usa a tangente no ponto mais proximo, que em geral nao e x=0.
    # Minimo de x^2 + y(x)^2: x + y(x)*y'(x) = 0 (polinomio cubico).
    roots = np.roots([2*a*a, 3*a*slope, 1+slope*slope+2*a*intercept,
                      slope*intercept])
    real_roots = [float(root.real) for root in roots if abs(root.imag) < 1e-8]
    closest_x = min(real_roots, key=lambda x: x*x + np.polyval(coefficients, x)**2)
    closest_y = float(np.polyval(coefficients, closest_x))
    closest_slope = float(2*a*closest_x + slope)
    normal_scale = math.sqrt(1+closest_slope*closest_slope)
    errors = ((closest_y-closest_x*closest_slope)/normal_scale,
              math.atan(closest_slope))
    closest_curvature = float(2*a / normal_scale**3)
    # Em curvas fechadas y(x) extrapola mal ate o eixo dianteiro.
    # Compara com um arco: mesma quantidade de parametros, erro em metros.
    xx, yy = x[keep], y[keep]
    design = np.column_stack((2*xx, 2*yy, np.ones(len(xx))))
    center_x, center_y, constant = np.linalg.lstsq(design, xx*xx+yy*yy, rcond=None)[0]
    radius_squared = constant + center_x**2 + center_y**2
    if radius_squared > 0:
        radius = math.sqrt(radius_squared)
        center_distance = math.hypot(center_x, center_y)
        circle_rmse = float(np.sqrt(np.mean(
            (np.hypot(xx-center_x, yy-center_y)-radius)**2)))
        poly_rmse = float(np.sqrt(np.mean(
            ((yy-np.polyval(coefficients, xx))/np.sqrt(1+(2*a*xx+slope)**2))**2)))
        if (1/radius > .15 and .6 < radius < 8
                and center_distance > .1 and circle_rmse < .9*poly_rmse):
            # Tangente orientada no sentido de avanco, normal positiva a esquerda.
            tangent = np.array([center_y, -center_x]) / center_distance
            if tangent[0] < 0:
                tangent = -tangent
            left_normal = np.array([-tangent[1], tangent[0]])
            turn_sign = math.copysign(1., np.dot([center_x, center_y], left_normal))
            errors = (turn_sign*(center_distance-radius),
                      math.atan2(tangent[1], tangent[0]))
            closest_curvature = turn_sign/radius
    return (*errors, closest_curvature) if with_curvature else errors


def curve_speed(reference, curvature, lateral, min_speed=.5,
                lateral_accel=.6):
    """Limita v por v^2*|kappa| e reduz avanco durante recuperacao de erro.

    Sem termo de orientacao: o erro angular extrapolado ate o eixo dianteiro
    oscilava quadro a quadro (+-0,9 rad) e fazia a vref variar 0,19-0,47 m/s.
    """
    if reference <= 0 or min_speed <= 0 or lateral_accel <= 0:
        raise ValueError('Velocidades e aceleracao lateral devem ser positivas.')
    bend_limit = math.sqrt(lateral_accel / max(abs(curvature), 1e-6))
    recovery_limit = reference / (1 + 2*abs(lateral))
    return min(reference, max(min_speed, min(bend_limit, recovery_limit)))


def path_points(errors, dkappa, arcs, max_curvature=math.inf, with_heading=False):
    """Pontos (Nx2, eixo dianteiro) do caminho descrito por (e, psi, kappa) e dkappa.

    Clotoide a partir do ponto mais proximo do eixo, como no estimador de
    cones: integra rumo e posicao com passo de 5 cm. with_heading: devolve
    tambem (rumo, curvatura) em cada ponto.
    """
    lateral, heading, curvature = errors
    dkappa = dkappa if math.isfinite(dkappa) else 0.
    arcs = np.asarray(arcs, float)
    end = max(float(np.max(arcs)), 0.)
    s = np.linspace(0., end, max(2, int(math.ceil(end/.05)) + 1))
    kappa = np.clip(curvature + dkappa*s, -max_curvature, max_curvature)
    step = s[1] - s[0]
    theta = heading + np.concatenate(([0.], np.cumsum((kappa[1:] + kappa[:-1])*step/2)))
    x = -lateral*math.sin(heading) + np.concatenate(([0.], np.cumsum((np.cos(theta[1:]) + np.cos(theta[:-1]))*step/2)))
    y = lateral*math.cos(heading) + np.concatenate(([0.], np.cumsum((np.sin(theta[1:]) + np.sin(theta[:-1]))*step/2)))
    points = np.c_[np.interp(arcs, s, x), np.interp(arcs, s, y)]
    if with_heading:
        return points, np.interp(arcs, s, theta), np.interp(arcs, s, kappa)
    return points


def pure_pursuit_steer(errors, dkappa, lookahead, wheelbase, max_curvature=math.inf):
    """Esterco (rad, esquerda +) que leva o eixo traseiro ao ponto do caminho
    `lookahead` m a frente do eixo dianteiro; devolve (esterco, ponto x, y).

    O caminho e o do eixo DIANTEIRO (centro do corredor); o pure pursuit guia
    o traseiro, que numa curva de raio R anda sqrt(R^2 - L^2), por dentro. Sem
    deslocar o alvo para esse caminho o esterco de regime saia 7-23% curto e o
    carro abria a curva.
    """
    points, heading, curvature = path_points(errors, dkappa, [lookahead], max_curvature, with_heading=True)
    target = points[0]
    kappa = float(curvature[0])
    if abs(kappa) > 1e-6:
        radius = 1./abs(kappa)
        inward = radius - math.sqrt(max(radius*radius - wheelbase*wheelbase, 0.))
        normal = np.array([-math.sin(heading[0]), math.cos(heading[0])])
        target = target + math.copysign(inward, kappa)*normal
    x, y = target[0] + wheelbase, target[1]  # referencial do eixo traseiro
    distance2 = x*x + y*y
    if distance2 < 1e-6:
        return 0., float(target[0]), float(target[1])
    return math.atan(2.*wheelbase*y/distance2), float(target[0]), float(target[1])


def preview_speed(errors, dkappa, reference, distance, lateral_accel=.6, deceleration=.5,
                  reaction=.3, speed=0., min_speed=.15, max_curvature=math.inf, sight_curvature=0.):
    """Maior velocidade agora que ainda permite fazer as curvas dos proximos `distance` m.

    O carro nao freia: so perde velocidade rolando (~deceleration m/s^2). Uma
    curva a s metros, com limite v_c = sqrt(a_lat/|kappa|), aceita hoje
    sqrt(v_c^2 + 2*desacel*(s - v*reacao)). Devolve (velocidade, curvatura
    maxima a frente, distancia dela).

    sight_curvature (1/m): o que vem DEPOIS de `distance` nao e visto; supoe
    ali uma curva com essa curvatura (a mais fechada da pista). E o limite de
    visibilidade: na reta o carro corre o maximo que ainda deixa rolar ate a
    velocidade dessa curva dentro do trecho visto.
    """
    lateral, heading, curvature = errors
    dkappa = dkappa if math.isfinite(dkappa) else 0.
    s = np.linspace(0., max(distance, 0.), 26)
    kappa = np.abs(np.clip(curvature + dkappa*s, -max_curvature, max_curvature))
    if sight_curvature > 0:
        kappa[-1] = max(kappa[-1], min(sight_curvature, max_curvature))
    bend = np.sqrt(lateral_accel/np.maximum(kappa, 1e-6))
    margin = np.maximum(s - abs(speed)*reaction, 0.)
    allowed = np.sqrt(bend*bend + 2.*deceleration*margin)
    worst = int(np.argmax(kappa))
    limit = float(min(reference, max(min_speed, float(np.min(allowed)))))
    return limit, float(kappa[worst]), float(s[worst])


MAX_LATENCY_COMPENSATION = .25  # s; idade tipica da imagem: 0,12 a 0,19 s
# Com o caminho do estimador (clotoide) e a odometria, a compensacao e exata e
# cobre a idade real da imagem: nos ensaios de 10/11 ela chegou a 0,42 s.
MAX_EXACT_COMPENSATION = .5
# Entre quadros (~60 ms) o carro gira < 0,2 rad mesmo a 1,5 m/s no batente.
# Saltos maiores sao deteccao errada (reflexo, outra faixa) e sao descartados,
# mas no maximo MAX_REJECTED_FRAMES seguidos: uma mudanca real e aceita depois.
MAX_HEADING_JUMP = .3  # rad
MAX_LATERAL_JUMP = .08  # m
MAX_REJECTED_FRAMES = 3
# Marcha a re medida so bloqueia apos este numero de passos seguidos com v < -0,05:
# um pico isolado do encoder (visto em ensaio: -0,43 m/s parado) nao desliga o Stanley.
REVERSE_CONFIRM_STEPS = 3
# Decisao de busca ao perder a faixa: em curva (|kappa| acima disto) vira para
# o lado da curva; em reta, para o lado em que a faixa estaria a frente.
SEARCH_MIN_CURVATURE = .25  # 1/m, raio de 4 m
SEARCH_LOOKAHEAD = .4  # m
SEARCH_MIN_OFFSET = .03  # m; faixa centrada na reta: nao ha lado para buscar


class RealStanleyFollower:
    """Comandos somente no laco principal; visao nunca atua nos motores."""
    def __init__(self, settings):
        self.settings = dict(settings)
        self.status = 'Aguardando primeira imagem'
        def positive(name, default):
            value = self.settings.get(name, default)
            if not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} deve ser finito e positivo.')
            return float(value)
        def nonnegative(name, default):
            value = self.settings.get(name, default)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} deve ser finito e nao negativo (0 desativa).')
            return float(value)
        self.timeout = positive('vision_timeout', .3)
        # Perdendo a faixa, segue a ultima trajetoria por ate line_loss_frames
        # quadros seguidos sem deteccao. line_loss_tolerance e o teto em
        # segundos para o caso de a visao parar de entregar quadros.
        self.loss_frames = self.settings.get('line_loss_frames', 8)
        if (isinstance(self.loss_frames, (bool, np.bool_))
                or not isinstance(self.loss_frames, (int, np.integer)) or self.loss_frames < 0):
            raise ValueError('line_loss_frames deve ser inteiro nao negativo (0 desativa).')
        self.loss_frames = int(self.loss_frames)
        self.loss_tolerance = nonnegative('line_loss_tolerance', 1.)
        # Durante a tolerancia, reduz o alvo de velocidade ate curve_min_speed
        # em vez de manter a vref cheia as cegas. Esgotada, neutraliza como antes.
        self.loss_speed_ramp = self.settings.get('loss_speed_ramp', False)
        # A rampa so comeca depois disto (s) e de mais de 1 quadro perdido: um
        # quadro descartado derrubava a vref por ~0,16 s, 10 vezes nos 7 ensaios
        # de 26/11 (o carro "engasgava" a cada salto descartado).
        self.loss_ramp_grace = nonnegative('loss_ramp_grace_s', .15)
        # Teto de velocidade da visao (uma borda 0,9; recuperacao 0,35) so vale
        # se durar isto (s): o maior teto da janela. Em 26/11 esses tetos
        # piscavam por 0,2-0,5 s e a vref despencava e voltava em rampa; mas na
        # pista virtual (5 pistas x 5 sementes, atraso real da direcao) 0,3 s
        # deu 96% das voltas e 0,15 s 92%, contra 100% sem: desligado (0).
        self.limit_debounce = nonnegative('speed_limit_debounce_s', 0.)
        self.limit_history = deque(maxlen=64)  # (horario da foto, teto)
        if not isinstance(self.loss_speed_ramp, (bool, np.bool_)):
            raise ValueError('loss_speed_ramp deve ser True ou False.')
        # Saltos maiores que estes, entre quadros proximos, sao descartados.
        self.max_lateral_jump = positive('max_lateral_jump_m', MAX_LATERAL_JUMP)
        self.max_heading_jump = positive('max_heading_jump_rad', MAX_HEADING_JUMP)
        # Memoria do tracado (s): constante de tempo com que um quadro novo
        # substitui o caminho ja identificado (_remember_path). 0 desliga.
        self.path_memory_s = nonnegative('path_memory_s', 0.)
        self.hold_speed_factor = 1.
        self.last_state = {}  # Resumo do ultimo apply, para o debug por quadro.
        self.reverse_steps = 0  # Fora de reset(): a re real continua bloqueada.
        self.reference = positive('velocity_reference', .5)
        self.min_speed = positive('curve_min_speed', .3)
        self.lateral_accel = positive('curve_lateral_accel', .4)
        self.use_vref = self.settings.get('stanley_use_vref', True)
        if not isinstance(self.use_vref, (bool, np.bool_)):
            raise ValueError('stanley_use_vref deve ser True ou False.')
        # Velocidade do termo atan(k*e/(ks + v)): com use_vref False, a MEDIDA
        # (encoder) passada por um passa-baixa de 1a ordem (constante de tempo
        # stanley_speed_filter_s): a do encoder oscila e faria o ganho efetivo
        # oscilar junto; a vref nao e a velocidade real (na partida e nas curvas,
        # o carro esta mais lento que a vref e o termo ficava fraco). 0: sem filtro.
        self.speed_filter_s = float(self.settings.get('stanley_speed_filter_s', 0.))
        if not math.isfinite(self.speed_filter_s) or self.speed_filter_s < 0:
            raise ValueError('stanley_speed_filter_s deve ser >= 0.')
        self.speed_filtered = None
        self.steering_sign = self.settings.get('steering_sign', 1.)
        if self.steering_sign not in (-1., 1.):
            raise ValueError('steering_sign deve ser -1 ou +1; calibre o sinal do carro real.')
        self.controller = StanleyController(
            gain=positive('stanley_gain', 1.5),
            softening=positive('stanley_softening', .3),
            max_steer=math.radians(min(20., positive('stanley_max_steer_deg', 20.))),
            max_rate=math.radians(positive('stanley_max_rate_deg_s', 60.)),
            wheelbase=positive('wheelbase', .36),
            curvature_gain=nonnegative('stanley_curvature_ff', 1.),
            yaw_damping=nonnegative('stanley_yaw_damping', 0.))
        # Constantes de tempo dos filtros; 0 desliga. O alvo de velocidade
        # oscilava com o erro angular e a curvatura medidos em ~0,5 m de faixa;
        # o carro, que nao freia, so dava trancos no acelerador. O esterco usa
        # a curvatura sem filtro: no mesmo ajuste, erros de orientacao e de
        # curvatura saem opostos e se cancelam na soma do Stanley.
        self.curvature_filter = nonnegative('curvature_filter_s', .2)
        self.speed_filter = nonnegative('speed_filter_s', .3)
        # Curva mais fechada que o batente da direcao nao e seguivel: e ruido.
        self.max_curvature = math.tan(self.controller.max_steer) / self.controller.wheelbase
        # Esterco extra acumulado por segundo de faixa perdida; 0 desliga.
        self.search_rate = math.radians(nonnegative('line_search_rate_deg_s', 40.))
        self.compensate_latency = self.settings.get('vision_latency_compensation', True)
        if not isinstance(self.compensate_latency, (bool, np.bool_)):
            raise ValueError('vision_latency_compensation deve ser True ou False.')
        # Lei de direcao: 'stanley' (erros no eixo) ou 'pure_pursuit' (ponto do
        # caminho a frente; so com path_dkappa, i.e. estimador de cones).
        self.steering_law = self.settings.get('steering_law', 'stanley')
        if self.steering_law not in ('stanley', 'pure_pursuit'):
            raise ValueError("steering_law deve ser 'stanley' ou 'pure_pursuit'.")
        # Distancia do alvo (do eixo dianteiro, no caminho): minimo + tempo*vref, com teto.
        # Pista virtual (tools/pista_virtual.py): 0,4 m + 0,2 s deu metade do erro
        # do Stanley; alvos de 1-2 m cortavam as curvas (o caminho do estimador
        # ja preenche o trecho cego junto ao carro).
        self.pursuit_min = positive('pursuit_min_lookahead_m', .4)
        self.pursuit_time = nonnegative('pursuit_lookahead_s', .2)
        self.pursuit_max = max(self.pursuit_min, positive('pursuit_max_lookahead_m', 1.5))
        # Previa de curva (s m a frente, com o caminho do estimador): o carro nao
        # freia, entao reduz a vref antes da curva. 0 desliga.
        self.preview_distance = nonnegative('speed_preview_m', 0.)
        self.coast_deceleration = positive('coast_deceleration', .5)
        self.preview_reaction = nonnegative('speed_preview_reaction_s', .3)
        # Curva mais fechada que pode aparecer logo depois do trecho visto (1/m);
        # 0 desliga o limite de visibilidade (so curvas ja vistas reduzem).
        self.sight_curvature = nonnegative('speed_sight_curvature', 0.)
        # Velocidade pelo atraso da imagem: o carro anda as cegas v*idade entre a
        # foto e o comando; limita a no maximo este trecho (m). 0 desliga.
        self.latency_budget = nonnegative('latency_speed_budget_m', 0.)
        # Teto de seguranca (m/s): a maior velocidade que ainda para antes de um
        # obstaculo visto no alcance do ultrassom (main.py ultrasonic_speed_cap).
        self.safe_speed = self.settings.get('safe_speed_m_s', math.inf)
        if (isinstance(self.safe_speed, (bool, np.bool_)) or not isinstance(self.safe_speed, (int, float, np.number))
                or math.isnan(self.safe_speed) or self.safe_speed <= 0):
            raise ValueError('safe_speed_m_s deve ser positivo (inf: sem teto).')
        self.safe_speed = float(self.safe_speed)
        # Teto dado de fora a cada passo (placa de PARE se aproximando, aruco.py).
        self.external_speed_limit = math.inf
        self.age_average = None
        self.odometry = None  # PoseHistory do laco de controle (compensacao exata)
        self.exact_compensation = self.settings.get('exact_latency_compensation', True)
        # Atraso da DIRECAO (s): o esterco comandado agora so vira guinada depois
        # disto (filtro do Car + servo + pneu; 26/11: ~0,16-0,20 s, guinada do
        # giroscopio contra o comando). O Stanley e calculado no estado em que o
        # carro estara quando o comando de agora chegar a roda, integrando os
        # comandos ja enviados e ainda "no caminho" (preditor de Smith). Sem
        # isso, so o historico de poses tira a oscilacao mas vira tarde em curva
        # fechada; a guinada de agora x idade da foto antecipava por acaso e
        # realimentava a oscilacao (esterco de lado a lado). 0 desliga.
        self.steer_delay = nonnegative('steer_delay_s', 0.)
        self.steer_response = positive('steering_response', 1.)
        self.commands = deque(maxlen=128)  # (instante, esterco geometrico comandado)
        self.steer_preview = (0., 0., 0.)  # (frente m, esquerda m, giro rad) previstos no atraso
        # Curva vista por uma cor so: multiplica o esterco pedido e a taxa do
        # volante (1 = sem reforco). A borda interna saiu da imagem e o carro
        # tendia a virar tarde/pouco.
        self.one_border_steer_gain = positive('one_border_curve_steer_gain', 1.)
        self.one_border_rate_gain = positive('one_border_curve_rate_gain', 1.)
        # Trajetoria de curva (fora-dentro): com uma curva chegando, a referencia
        # e deslocada para o lado de FORA (o carro abre antes da curva, ve mais
        # cones dela e entra com folga do lado de dentro); dentro da curva, para o
        # lado de dentro (fecha). Deslocamento do carro em relacao ao centro, em m:
        #     racing_line_m * (1,5*sat(kappa_agora) - sat(kappa_a_frente)),  sat = kappa/racing_curvature
        # limitado a meia largura - meia largura do carro - racing_margin_m.
        # kappa_a_frente: a maior curvatura entre racing_ahead_m (de 0,5 a 1,5 m)
        # pela clotoide do estimador. 0 desliga.
        self.racing_line = nonnegative('racing_line_m', 0.)
        self.racing_curvature = positive('racing_curvature', .5)
        self.racing_margin = nonnegative('racing_margin_m', .2)
        self.racing_filter = nonnegative('racing_filter_s', .3)
        self.racing_inside = nonnegative('racing_inside', 1.5)  # peso de "fechar" dentro da curva
        # So em pista estreita: inteiro ate racing_full_width_m de largura, zero a
        # partir de racing_zero_width_m (pista larga ja tem folga; abrir nela tirava
        # a borda de dentro da imagem na pista virtual).
        self.racing_full_width = positive('racing_full_width_m', 1.3)
        self.racing_zero_width = positive('racing_zero_width_m', 1.7)
        # Tracado de corrida (24/11, racing_mode "apice"; "antigo" = formula acima):
        # chega por FORA (racing_line_m), converge para o APICE dentro da curva
        # (racing_apex_m para dentro, limitado pela folga aos cones de dentro) e
        # compensa o arrasto da traseira (L^2*kappa/2: a roda de tras passa por
        # dentro da da frente; o Stanley controla so o eixo dianteiro).
        self.racing_mode = self.settings.get('racing_mode', 'antigo')
        if self.racing_mode not in ('antigo', 'apice'):
            raise ValueError('racing_mode deve ser "apice" ou "antigo".')
        self.racing_apex = nonnegative('racing_apex_m', 0.)
        self.racing_half_car = nonnegative('racing_half_car_m', .13)
        self.racing_offtrack = self.settings.get('racing_offtracking', True)
        self.racing_offset = 0.
        # Centralizacao: integral do erro lateral vira um esterco de correcao
        # (trim), igual para Stanley e pure pursuit. Corrige vies fixo (centro
        # do servo, assimetria esquerda/direita): sem ela o carro ficava 2-12 cm
        # a direita do centro nos ensaios de 10/11. Integra so com referencia
        # confiavel (duas bordas, sem perda), andando, e com limite.
        self.centering_gain = nonnegative('centering_gain', 0.)  # rad por (m*s)
        # Trim pelo esterco medio na reta: seguindo uma reta, a roda real fica em 0,
        # entao o comando medio vale -(vies + trim). Nao depende do erro estimado,
        # que fica enviesado junto com a odometria (sem IMU, a guinada vem do
        # esterco comandado): na pista virtual, 2 graus de vies davam 11 cm de erro
        # real com o estimador vendo -2 cm.
        self.centering_steer_gain = nonnegative('centering_steer_gain', 0.)  # 1/s
        self.centering_limit = math.radians(nonnegative('centering_limit_deg', 4.))
        self.centering_min_speed = nonnegative('centering_min_speed', .2)
        # So aprende em reta (|kappa| no eixo e 1,5 m a frente abaixo disto): em
        # curva o erro natural do seguimento nao e vies do carro.
        self.centering_max_curvature = positive('centering_max_curvature', .15)
        self.centering = 0.  # fora de reset(): o vies do carro nao muda ao perder a faixa
        self.centering_time = 0.  # s de reta em que aprendeu (para salvar o trim)
        # Trim fixo do carro (graus): ponto de partida do aprendido.
        self.trim_base = math.radians(float(self.settings.get('steering_trim_deg', 0.) or 0.))
        self.trim_sent = None
        # Rampa assimetrica da vref: sobe no maximo speed_rise_m_s2 (m/s^2) e
        # desce na hora. 0 mantem o passa-baixas simetrico speed_filter_s.
        self.speed_rise = nonnegative('speed_rise_m_s2', 0.)
        self.steering_start_distance = nonnegative('steering_start_distance_m', 0.)
        self.reset()

    def reset(self, restart_steering=True):
        """Neutraliza o Stanley e esquece a ultima faixa vista."""
        self.controller.reset()
        self.line_memory = None
        self.line_seen_at = None
        self.line_speed_limit = math.inf
        self.line_speed_factor = 1.
        self.path_dkappa = math.nan
        self.one_border_curve = False
        self.pursuit_target = None
        self.preview = None
        self.curvature = None
        self.curvature_at = None
        self.speed_target = None
        self.frame_at = None
        self.frame_rejected = False
        self.rejected_frames = 0
        self.blend_frame_at = None   # memoria do tracado (_remember_path)
        self.blend_output = None
        self.blend_at = None
        self.blend_residual = np.zeros(2)
        self.counted_at = None
        self.lost_frames = 0
        self.missed_frames = 0  # quadros sem deteccao alguma (sem contar descartes)
        self.search_direction = None
        if restart_steering:
            self.distance_since_start = 0.

    def _remember_path(self, errors, observation, now, car):
        """Memoria do tracado: um quadro novo nao troca o caminho de uma vez.

        Entre quadros, o tracado ja identificado e levado pela odometria (e pelo
        giroscopio). Quando chega um quadro novo, a diferenca entre o que ele diz
        e o que o carro ja sabia (deslocamento e rumo no eixo, agora) vira um
        residuo que some com constante de tempo path_memory_s. Se os quadros
        seguintes confirmam o caminho novo, ele vale em ~3 constantes de tempo; se
        era um quadro ruim (uma borda so, 2-4 cones), o caminho lembrado segura.
        Nos ensaios de 24/11 cada quadro discordava do anterior em ~0,15 rad de
        rumo (abaixo do salto maximo de 0,35) e o Stanley obedecia cada um: o
        esterco trocava de lado 2 vezes por segundo ("pedalando")."""
        if self.path_memory_s <= 0:
            return errors
        stamp = getattr(observation, 'timestamp', None)
        if stamp != self.blend_frame_at:
            self.blend_frame_at = stamp
            residual = np.zeros(2)
            # So com o caminho lembrado do passo anterior: depois de um quadro
            # descartado ou de uma perda, ele ja esta velho e o novo entra direto.
            if (self.blend_output is not None and self.blend_at is not None
                    and 0. <= now - self.blend_at <= 3.*float(car.dt)):
                expected = self._predict_line(self.blend_output, car, now - self.blend_at)
                residual = np.array([expected[0] - errors[0],
                                     math.atan2(math.sin(expected[1] - errors[1]), math.cos(expected[1] - errors[1]))])
                # So suaviza o que e plausivel como ruido; salto maior ja e tratado
                # (descartado ou aceito) pelo teste de salto acima.
                residual = np.clip(residual, [-self.max_lateral_jump, -self.max_heading_jump],
                                   [self.max_lateral_jump, self.max_heading_jump])
            self.blend_residual = residual
        elif self.blend_at is not None:
            self.blend_residual = self.blend_residual*math.exp(-max(0., now - self.blend_at)/self.path_memory_s)
        output = (errors[0] + float(self.blend_residual[0]), errors[1] + float(self.blend_residual[1]), errors[2])
        self.blend_output, self.blend_at = output, now
        return output

    def _filter_curvature(self, curvature, timestamp):
        """Passa-baixa por quadro novo (o mesmo quadro serve varios passos)."""
        if self.curvature is None or self.curvature_filter <= 0:
            self.curvature, self.curvature_at = curvature, timestamp
        elif timestamp > self.curvature_at:
            weight = 1. - math.exp(-(timestamp - self.curvature_at) / self.curvature_filter)
            self.curvature += weight * (curvature - self.curvature)
            self.curvature_at = timestamp
        return self.curvature

    def _search_side(self, errors):
        """+1 esquerda, -1 direita, 0 sem lado: decidido uma vez por perda."""
        lateral, heading, curvature = errors
        if self.curvature is not None:
            curvature = self.curvature  # filtrada: um quadro ruidoso nao inverte o lado
        if abs(curvature) >= SEARCH_MIN_CURVATURE:
            return math.copysign(1., curvature)
        ahead = lateral + SEARCH_LOOKAHEAD * math.sin(heading)
        return math.copysign(1., ahead) if abs(ahead) > SEARCH_MIN_OFFSET else 0.

    @property
    def steer_trim(self):
        """Centro da direcao em uso (rad, geometrico): fixo + aprendido."""
        return self.trim_base + self.centering

    def _debounced_limit(self, limit, stamp):
        """Maior teto das fotos dos ultimos limit_debounce s: um teto mais baixo
        so vale depois de se manter esse tempo (o mais alto vale na hora)."""
        try:
            limit = float(limit)
        except (TypeError, ValueError):
            limit = math.inf
        if self.limit_debounce <= 0 or not isinstance(stamp, (int, float, np.number)) or not math.isfinite(stamp):
            return limit
        stamp = float(stamp)
        if self.limit_history and stamp < self.limit_history[-1][0]:
            self.limit_history.clear()  # relogio voltou (nova missao, testes)
        if not self.limit_history or stamp > self.limit_history[-1][0]:
            self.limit_history.append((stamp, limit))
        while len(self.limit_history) > 1 and self.limit_history[1][0] <= stamp - self.limit_debounce:
            self.limit_history.popleft()
        return max(value for _, value in self.limit_history)

    def _record_command(self, now, steer):
        """Esterco (convencao geometrica) enviado em `now`, para _preview_actuator."""
        if isinstance(now, (int, float, np.number)) and math.isfinite(now) and math.isfinite(steer):
            if self.commands and now <= self.commands[-1][0]:
                self.commands.pop()  # mesmo instante: vale o ultimo
            self.commands.append((float(now), float(steer)))

    def _send_trim(self, car):
        """Trim no SERVO (Car.set_steer_trim), fora do esterco usado na odometria."""
        setter = getattr(car, 'set_steer_trim', None)
        trim = self.steering_sign*self.steer_trim
        if callable(setter) and (self.trim_sent is None or abs(trim - self.trim_sent) > 1e-4):
            setter(trim)
            self.trim_sent = trim

    def _racing_line_offset(self, curvature, observation, dt):
        """Deslocamento desejado do carro em relacao ao centro (m, esquerda +)."""
        if (not math.isfinite(self.path_dkappa)
                or self.racing_line <= 0 and (self.racing_mode == 'antigo' or not self.racing_offtrack)
                and self.racing_apex <= 0):
            self.racing_offset = 0.
            return 0.
        width = getattr(observation, 'corridor_width_m', math.nan)
        known = isinstance(width, (int, float, np.number)) and math.isfinite(width)
        if self.racing_mode == 'apice':
            return self._apex_line_offset(curvature, width if known else math.nan, dt)
        arcs = np.linspace(.5, 1.5, 6)
        ahead = curvature + self.path_dkappa*arcs
        ahead = float(ahead[np.argmax(np.abs(ahead))])
        sat = lambda value: float(np.clip(value/self.racing_curvature, -1., 1.))
        target = self.racing_line*(self.racing_inside*sat(curvature) - sat(ahead))
        room = max(0., width/2 - .13 - self.racing_margin) if known else 0.
        if known and self.racing_zero_width > self.racing_full_width:
            target *= float(np.clip((self.racing_zero_width - width)/(self.racing_zero_width - self.racing_full_width),
                                    0., 1.))
        target = float(np.clip(target, -room, room))
        if self.racing_filter > 0:
            self.racing_offset += (1. - math.exp(-float(dt)/self.racing_filter))*(target - self.racing_offset)
        else:
            self.racing_offset = target
        return self.racing_offset

    def _apex_line_offset(self, curvature, width, dt):
        """Fora -> apice -> centro, pela clotoide do estimador (m, carro a esquerda +).

        kappa(s) = kappa + dkappa*s nos proximos 2 m. Com a curva ainda a frente
        (|kappa| no eixo bem menor que o pico), o peso de ENTRADA leva o carro para
        fora; dentro dela, o peso de CURVA leva ao apice. Os dois limites vem da
        largura: para fora ate a borda de fora menos o carro e a margem; para
        dentro ate a borda de dentro menos o carro, a margem e o arrasto da
        traseira."""
        arcs = np.linspace(0., 2., 11)
        profile = np.clip(curvature + self.path_dkappa*arcs, -self.max_curvature, self.max_curvature)
        peak = float(profile[np.argmax(np.abs(profile))])
        here = float(np.clip(curvature, -self.max_curvature, self.max_curvature))
        if abs(peak) < 1e-3:
            target = 0.
        else:
            sign = math.copysign(1., peak)
            sat = lambda value: float(np.clip(abs(value)/self.racing_curvature, 0., 1.))
            entry = sat(peak)*float(np.clip((abs(peak) - abs(here))/abs(peak), 0., 1.))
            bend = sat(here)*(1. - entry)
            offtrack = self.controller.wheelbase**2*abs(here)/2 if self.racing_offtrack else 0.
            target = sign*(-self.racing_line*entry + self.racing_apex*bend) - sign*offtrack
            if math.isfinite(width):
                outside = max(0., width/2 - self.racing_half_car - self.racing_margin)
                inside = max(0., width/2 - self.racing_half_car - self.racing_margin - offtrack)
                if self.racing_zero_width > self.racing_full_width:
                    fade = float(np.clip((self.racing_zero_width - width)
                                         / (self.racing_zero_width - self.racing_full_width), 0., 1.))
                    target = sign*(-(self.racing_line*entry*fade) + self.racing_apex*bend*fade) - sign*offtrack
                # sign*target > 0: para dentro; < 0: para fora.
                target = sign*float(np.clip(sign*target, -outside, inside))
            else:
                target = -sign*offtrack  # sem largura conhecida: so o arrasto
        if self.racing_filter > 0:
            self.racing_offset += (1. - math.exp(-float(dt)/self.racing_filter))*(target - self.racing_offset)
        else:
            self.racing_offset = target
        return self.racing_offset

    def _filter_measured_speed(self, speed, dt):
        """|v| do encoder pelo passa-baixa (nao zera no reset: o carro nao para)."""
        speed = abs(float(speed)) if isinstance(speed, (int, float, np.number)) and math.isfinite(speed) else math.nan
        if not math.isfinite(speed):
            return self.speed_filtered if self.speed_filtered is not None else math.nan
        if self.speed_filtered is None or self.speed_filter_s <= 0:
            self.speed_filtered = speed
        else:
            self.speed_filtered += (1. - math.exp(-float(dt)/self.speed_filter_s))*(speed - self.speed_filtered)
        return self.speed_filtered

    def _filter_speed(self, target, dt, current=0.):
        """current: velocidade medida; a rampa parte dela (na partida e depois de
        uma parada a vref nao salta direto para o alvo)."""
        if self.speed_target is None and self.speed_rise > 0:
            current = float(current) if isinstance(current, (int, float, np.number)) and math.isfinite(current) else 0.
            self.speed_target = min(target, max(0., abs(current)))
        if self.speed_target is None or self.speed_filter <= 0 and self.speed_rise <= 0:
            self.speed_target = target
        elif self.speed_rise > 0:
            # Desce na hora (curva a frente, menos certeza); sobe em rampa: a
            # vref nao oscila com o ruido e o acelerador nao da trancos.
            self.speed_target = min(target, self.speed_target + self.speed_rise*dt)
        else:
            self.speed_target += (1. - math.exp(-dt / self.speed_filter)) * (target - self.speed_target)
        return self.speed_target

    def _predict_line(self, errors, car, dt=None):
        """Avanca a faixa por dt segundos de odometria (padrao: um passo).

        Supoe a linha mantendo a curvatura medida; o eixo dianteiro anda na
        direcao das rodas (car.st) e a carroceria gira com car.w. O sinal do
        atuador volta ambos para a convencao geometrica (esquerda positiva).
        """
        lateral, heading, curvature = errors
        steer = self.steering_sign * float(getattr(car, 'st', 0.))
        yaw_rate = self.steering_sign * float(getattr(car, 'w', 0.))
        steer = steer if math.isfinite(steer) and abs(steer) < math.pi / 2 else 0.
        yaw_rate = yaw_rate if math.isfinite(yaw_rate) else 0.
        dt = car.dt if dt is None else dt
        front_speed = car.v / math.cos(steer)
        relative = heading - steer
        return (lateral + front_speed * math.sin(relative) * dt,
                heading + (curvature * front_speed * math.cos(relative) - yaw_rate) * dt,
                curvature)

    def _propagate(self, errors, observation, now, car):
        """(erros no instante `now`, idade compensada) levando a clotoide da foto
        pelo movimento do carro; None sem o caminho do estimador."""
        dkappa = getattr(observation, 'path_dkappa', math.nan)
        if not isinstance(dkappa, (int, float, np.number)) or not math.isfinite(dkappa):
            return None
        try:
            from .corridor import move_states
        except ImportError:  # modulo carregado fora do pacote (alguns testes)
            from corridor import move_states
        start = float(observation.timestamp)
        age = min(now - start, MAX_EXACT_COMPENSATION)
        if not age > 0:
            return None
        motion = None
        if self.odometry is not None:
            try:
                motion = self.odometry.delta(start, start + age)
            except Exception:  # noqa: BLE001 - historico incompleto: usa o modelo
                motion = None
        if motion is None or not np.all(np.isfinite(motion)):
            # Sem historico: integra a velocidade e a guinada atuais (eixo traseiro).
            speed = float(car.v) if math.isfinite(float(car.v)) else 0.
            yaw_rate = self.steering_sign*float(getattr(car, 'w', 0.) or 0.)
            yaw_rate = yaw_rate if math.isfinite(yaw_rate) else 0.
            dth = yaw_rate*age
            dx, dy = speed*age*math.cos(dth/2), speed*age*math.sin(dth/2)
        else:
            dx, dy, dth = (float(v) for v in motion)
        wheelbase = self.controller.wheelbase  # pose no eixo traseiro -> eixo dianteiro
        dx += wheelbase*(math.cos(dth) - 1)
        dy += wheelbase*math.sin(dth)
        moved = move_states([*errors, float(dkappa), 1.], dx, dy, dth, self.max_curvature)[0]
        return (float(moved[0]), float(moved[1]), float(moved[2])), age

    def _preview_actuator(self, errors, car, now):
        """Erros no instante em que o esterco comandado agora chega a roda.

        O carro ainda vai andar steer_delay s com os comandos dos ultimos
        steer_delay s (ja enviados, ainda no filtro/servo): integra a
        bicicleta com eles (segurando cada comando ate o seguinte) e leva a
        clotoide da foto ate la, como _propagate faz com a odometria."""
        self.steer_preview = (0., 0., 0.)
        speed = float(getattr(car, 'v', 0.))
        if self.steer_delay <= 0 or not self.commands or not math.isfinite(speed) or speed <= .05:
            return errors
        step = .02
        count = max(1, int(round(self.steer_delay/step)))
        times = np.fromiter((c[0] for c in self.commands), float, len(self.commands))
        values = np.fromiter((c[1] for c in self.commands), float, len(self.commands))
        when = now - self.steer_delay + (np.arange(count) + .5)*self.steer_delay/count
        steer = values[np.clip(np.searchsorted(times, when, side='right') - 1, 0, len(values) - 1)]
        wheelbase = self.controller.wheelbase
        rate = speed/wheelbase*np.tan(steer)*self.steer_response
        dt = self.steer_delay/count
        heading = np.concatenate(([0.], np.cumsum(rate*dt)))
        middle = (heading[1:] + heading[:-1])/2
        dx, dy, dth = (float(np.sum(speed*dt*np.cos(middle))), float(np.sum(speed*dt*np.sin(middle))),
                       float(heading[-1]))
        self.steer_preview = (dx, dy, dth)
        dx += wheelbase*(math.cos(dth) - 1)  # eixo traseiro -> eixo dianteiro
        dy += wheelbase*math.sin(dth)
        try:
            from .corridor import move_states
        except ImportError:  # modulo carregado fora do pacote (alguns testes)
            from corridor import move_states
        dkappa = self.path_dkappa if math.isfinite(self.path_dkappa) else 0.
        moved = move_states([*errors, dkappa, 1.], dx, dy, dth, self.max_curvature)[0]
        return float(moved[0]), float(moved[1]), float(moved[2])

    def apply(self, car, observation, now, obstacle_detected=False):
        """Retorna True quando uma observacao atual, ou a faixa vista ha no maximo
        line_loss_frames quadros (e line_loss_tolerance segundos), autoriza seguir."""
        age = float('nan')
        errors = None
        stanley_speed = float('nan')
        steering_factor = float('nan')
        speed_raw = float('nan')
        reference_valid = True
        if (isinstance(now, (int, float, np.number)) and math.isfinite(now)
                and observation is not None):
            try:
                age = float(now - observation.timestamp)
                measured = np.asarray(observation.errors, dtype=float)
                if (math.isfinite(age) and 0 <= age <= self.timeout
                        and measured.shape == (3,) and np.all(np.isfinite(measured))):
                    errors = tuple(float(x) for x in measured)
            except (TypeError, ValueError, AttributeError):
                pass
        finite_sample = (isinstance(car.v, (int, float, np.number))
                         and isinstance(car.dt, (int, float, np.number))
                         and math.isfinite(car.v) and math.isfinite(car.dt) and car.dt > 0)
        gear = getattr(car, 'gear', None)
        reverse_gear = getattr(gear, 'value', gear) in ('reverse', -1)
        reverse_requested = (getattr(car, 'sel_tracao', 1) == 0
                             and getattr(car, 'rc_acelerador', 0.) < 0)
        line_seen = errors is not None
        speed_limit = self._debounced_limit(getattr(observation, 'speed_limit', math.inf),
                                            getattr(observation, 'timestamp', None))
        holding = False
        lost_for = float('nan')
        compensated = 0.
        curvature_raw = errors[2] if line_seen else float('nan')
        if line_seen:
            # Curva mais fechada que o batente da direcao nao e seguivel: e ruido.
            errors = (*errors[:2], float(np.clip(errors[2], -self.max_curvature, self.max_curvature)))
        if line_seen and finite_sample and observation.timestamp != self.frame_at:
            # Julga cada quadro uma vez, embora ele sirva a varios passos.
            self.frame_at = observation.timestamp
            elapsed = float(now) - self.line_seen_at if self.line_memory is not None else math.nan
            # Reencontrando a faixa apos quadros sem deteccao (busca), a previsao
            # ja nao e referencia confiavel: aceita a faixa que voltar.
            jump = (self.missed_frames == 0 and 0. <= elapsed <= self.loss_tolerance
                    and speed_limit <= self.line_speed_limit)
            if jump:
                expected = self._predict_line(self.line_memory, car, elapsed)
                measured = errors[:2]
                if self.compensate_latency and age > 0:
                    measured = self._predict_line(errors, car, min(age, MAX_LATENCY_COMPENSATION))[:2]
                # Estimador ainda convergindo (eixo incerto): a correcao legitima
                # passa do salto maximo; o proprio estimador ja descarta cones ruins.
                sigma = getattr(observation, 'lateral_sigma_m', math.nan)
                allowed = self.max_lateral_jump + (3.*sigma if math.isfinite(sigma) else 0.)
                jump = (abs(measured[0] - expected[0]) > allowed
                        or abs(measured[1] - expected[1]) > self.max_heading_jump)
            self.rejected_frames = self.rejected_frames + 1 if jump else 0
            self.frame_rejected = jump and self.rejected_frames <= MAX_REJECTED_FRAMES
        outlier = line_seen and self.frame_rejected
        if outlier:
            line_seen, errors = False, None
        # Conta cada quadro novo uma vez, embora ele sirva a varios passos.
        frame_time = getattr(observation, 'timestamp', None)
        if line_seen:
            self.lost_frames = self.missed_frames = 0
        elif (isinstance(frame_time, (int, float, np.number)) and math.isfinite(frame_time)
                and frame_time != self.counted_at):
            self.lost_frames += 1
            self.missed_frames += not outlier
        if observation is not None:
            self.counted_at = frame_time
        lost_frames = self.lost_frames  # reset() abaixo zera o contador
        if line_seen:
            self._filter_curvature(errors[2], float(observation.timestamp))
        if line_seen and self.compensate_latency and finite_sample and age > 0:
            # Os erros sao do instante da captura; a 1 m/s o carro anda ~30 cm
            # ate o comando. Com o caminho do estimador, leva a clotoide ao
            # instante atual pelo movimento real (odometria); sem ele, o modelo
            # antigo de curvatura constante, com teto.
            exact = self._propagate(errors, observation, float(now), car) if self.exact_compensation else None
            if exact is not None:
                errors, compensated = exact
            else:
                compensated = min(age, MAX_LATENCY_COMPENSATION)
                errors = self._predict_line(errors, car, compensated)
        if line_seen and finite_sample:
            errors = self._remember_path(errors, observation, float(now), car)
        if line_seen:
            self.line_memory, self.line_seen_at = errors, float(now)
            self.line_speed_limit = speed_limit
            factor = getattr(observation, 'speed_factor', 1.)
            self.line_speed_factor = (float(np.clip(factor, 0., 1.))
                                      if isinstance(factor, (int, float, np.number)) and math.isfinite(factor) else 1.)
            self.one_border_curve = bool(getattr(observation, 'one_border_curve', False))
            dkappa = getattr(observation, 'path_dkappa', math.nan)
            self.path_dkappa = (float(dkappa) if isinstance(dkappa, (int, float, np.number))
                                and math.isfinite(dkappa) else math.nan)
            self.search_direction = None
        elif (self.line_memory is not None and finite_sample
                and isinstance(now, (int, float, np.number)) and math.isfinite(now)):
            lost_for = float(now) - self.line_seen_at
            if 0 <= lost_for <= self.loss_tolerance and lost_frames <= self.loss_frames:
                # Reflexo ou sujeira raramente apagam a faixa de verdade: segue a
                # ultima trajetoria vista, por odometria, e so para se a perda durar.
                if self.search_direction is None:
                    self.search_direction = self._search_side(self.line_memory)
                errors = self.line_memory = self._predict_line(self.line_memory, car)
                holding = True
        self.reverse_steps = self.reverse_steps + 1 if finite_sample and car.v < -.05 else 0
        moving_backward = self.reverse_steps >= REVERSE_CONFIRM_STEPS
        active = (not car.emergencia and not obstacle_detected and finite_sample
                  and getattr(car, 'encoder_valid', True)
                  and not reverse_gear and not reverse_requested
                  and not moving_backward and errors is not None)
        search = 0.
        if active:
            errors = self._preview_actuator(errors, car, float(now))
            lateral, heading, curvature = errors
            offset = self._racing_line_offset(curvature, observation, car.dt)
            if offset:
                # Referencia = centro deslocado `offset` (carro a esquerda do centro: +).
                lateral = lateral + offset
                errors = (lateral, heading, curvature)
            if holding and self.search_direction:
                # Perdeu a faixa numa curva para a esquerda: vira cada vez mais
                # para a esquerda, onde ela provavelmente saiu da imagem.
                search = self.search_direction * min(self.controller.max_steer,
                                                     self.search_rate * lost_for)
            # Primeiro resolve a referencia deste passo (inclui curvas e RC).
            # Usar car.vref antes de set_vel leria a referencia anterior,
            # frequentemente zero no inicio ou apos uma perda de faixa.
            # Curvatura filtrada so no limite de velocidade (antes do 1o quadro: a atual).
            speed_curvature = curvature if self.curvature is None else self.curvature
            if search:
                # Buscando com mais esterco, limita a velocidade pela curva feita.
                searched = math.tan(abs(self.controller.steer) + abs(search)) / self.controller.wheelbase
                speed_curvature = math.copysign(max(abs(speed_curvature), min(searched, self.max_curvature)),
                                                search)
            speed_raw = curve_speed(self.reference, speed_curvature, lateral,
                                    min_speed=self.min_speed, lateral_accel=self.lateral_accel)
            self.preview = None
            if self.preview_distance > 0 and math.isfinite(self.path_dkappa):
                # Curva vista a frente: reduz agora, enquanto ainda da para rolar ate ela.
                self.preview = preview_speed(
                    errors, self.path_dkappa, self.reference, self.preview_distance,
                    lateral_accel=self.lateral_accel, deceleration=self.coast_deceleration,
                    reaction=self.preview_reaction, speed=car.v, min_speed=self.min_speed,
                    max_curvature=self.max_curvature, sight_curvature=self.sight_curvature)
                speed_raw = min(speed_raw, self.preview[0])
            self.hold_speed_factor = 1.
            if holding and self.loss_speed_ramp:
                # Fracao consumida da tolerancia (quadros ou tempo, o que vier antes).
                grace = min(self.loss_ramp_grace, .9*self.loss_tolerance)
                progress = max((lost_frames - 1)/max(self.loss_frames - 1, 1),
                               (lost_for - grace)/(self.loss_tolerance - grace) if self.loss_tolerance > 0 else 1.)
                progress = float(np.clip(progress, 0., 1.))
                ramped = max(min(self.min_speed, speed_raw), (1.-progress)*speed_raw + progress*self.min_speed)
                self.hold_speed_factor = ramped/speed_raw if speed_raw > 0 else 1.
                speed_raw = ramped
            if self.line_speed_factor < 1.:
                # Uma borda so ou memoria: menos certeza, menos velocidade (sem
                # descer abaixo do minimo operacional).
                speed_raw = max(min(self.min_speed, speed_raw), speed_raw*self.line_speed_factor)
            if self.latency_budget > 0 and math.isfinite(age) and age > 0:
                # Idade media da imagem (o mesmo quadro serve varios passos).
                self.age_average = (age if self.age_average is None
                                    else self.age_average + .05*(age - self.age_average))
                speed_raw = min(speed_raw, max(self.min_speed, self.latency_budget/max(self.age_average, .05)))
            limit = min(speed_limit, self.line_speed_limit, self.safe_speed, self.external_speed_limit)
            speed_raw = min(speed_raw, limit)
            car.set_vel(min(self._filter_speed(speed_raw, car.dt, getattr(car, 'v', 0.)), limit))
            active = bool(getattr(car, 'encoder_valid', True))
            stanley_speed = car.vref if self.use_vref else self._filter_measured_speed(car.v, car.dt)
            reference_valid = (isinstance(stanley_speed, (int, float, np.number))
                               and math.isfinite(stanley_speed))
            active = active and reference_valid
            dkappa = self.path_dkappa if math.isfinite(self.path_dkappa) else 0.
            straight = max(abs(curvature), abs(curvature + 1.5*dkappa)) < self.centering_max_curvature
            if ((self.centering_gain > 0 or self.centering_steer_gain > 0) and line_seen and not holding and not search and straight
                    and self.line_speed_factor >= .999 and abs(lateral) < .4
                    and abs(float(car.v)) >= self.centering_min_speed):
                learn = self.centering_gain*lateral + self.centering_steer_gain*self.controller.steer
                self.centering = float(np.clip(self.centering + learn*car.dt,
                                               -self.centering_limit, self.centering_limit))
                self.centering_time += float(car.dt)
            self._send_trim(car)
            boost = self.one_border_curve and not search
            gain = self.one_border_steer_gain if boost else 1.
            rate = self.one_border_rate_gain if boost else 1.
            self.controller.gain_scale, self.controller.rate_scale = gain, rate
            # Com v quase zero, o Stanley amplifica o erro lateral e chega ao
            # batente antes de sair. Libera o angulo pelo deslocamento medido.
            steering_factor = 1.
            if self.steering_start_distance > 0:
                if car.v > .08:
                    self.distance_since_start = min(self.steering_start_distance,
                                                    self.distance_since_start + car.v*car.dt)
                steering_factor = self.distance_since_start / self.steering_start_distance
            pursuit = (active and self.steering_law == 'pure_pursuit'
                       and math.isfinite(self.path_dkappa) and not search)
            self.pursuit_target = None
            if pursuit:
                lookahead = float(np.clip(max(self.pursuit_min + self.pursuit_time*abs(stanley_speed),
                                              getattr(observation, 'min_lookahead_m', 0.) or 0.),
                                          self.pursuit_min, self.pursuit_max))
                raw, x, y = pure_pursuit_steer(errors, self.path_dkappa, lookahead,
                                               self.controller.wheelbase, self.max_curvature)
                steer = self.controller.steer_to(raw, car.dt, gain, rate, steering_factor)
                self.pursuit_target = (lookahead, x, y)
                car.set_steer(self.steering_sign * steer)
                self._record_command(now, steer)
                active = bool(getattr(car, 'encoder_valid', True))
            elif active:
                gyro = getattr(car, 'w_imu', math.nan)
                gyro = (self.steering_sign*float(gyro) if isinstance(gyro, (int, float, np.number))
                        and math.isfinite(gyro) else math.nan)
                steer = self.controller.update(lateral, heading, stanley_speed, car.dt, curvature, search,
                                               yaw_rate=gyro, steering_factor=steering_factor)
                car.set_steer(self.steering_sign * steer)
                self._record_command(now, steer)
                # O setter revalida a serial antes de enviar a direcao.
                active = bool(getattr(car, 'encoder_valid', True))
        if not active:
            car._set_ref(0.)
            car.vel_error_integral = 0.
            car.set_u(0.)
            car.set_steer(0.)
            self._record_command(now, 0.)
            self.reset(restart_steering=not finite_sample or car.v <= .08)
            steer = 0.
        if active and holding and outlier:
            self.status = 'Quadro com salto impossivel descartado; seguindo ultima trajetoria'
        elif active and holding:
            side = {1.: '; procurando a esquerda', -1.: '; procurando a direita'}.get(
                math.copysign(1., search) if search else 0., '')
            ramp = f'; vref x{self.hold_speed_factor:.2f}' if self.loss_speed_ramp else ''
            self.status = (f'Faixa perdida ha {lost_frames} quadro(s) ({lost_for:.2f} s);'
                           f' seguindo ultima trajetoria{side}{ramp} (limite {self.loss_frames} quadros)')
        elif active and math.isfinite(self.line_speed_limit):
            self.status = observation.reason
        elif active:
            self.status = 'Faixa valida; controle liberado'
        elif obstacle_detected:
            self.status = 'Parada por obstaculo travada ate o fim da missao'
        elif car.emergencia:
            self.status = 'Emergencia ativa'
        elif not getattr(car, 'encoder_valid', True):
            self.status = 'Encoder sem leitura valida'
        elif not finite_sample:
            self.status = 'Velocidade ou periodo de controle invalido'
        elif not reference_valid:
            self.status = 'Velocidade selecionada para o Stanley invalida'
        elif reverse_gear or reverse_requested or moving_backward:
            self.status = 'Marcha a re: Stanley dianteiro bloqueado'
        elif observation is None:
            self.status = 'Aguardando primeira imagem'
        elif not math.isfinite(age) or age < 0:
            self.status = 'Horario de imagem invalido'
        elif age > self.timeout:
            self.status = f'Imagem atrasada: {age:.3f} s; limite {self.timeout:.3f} s'
        else:
            self.status = getattr(observation, 'reason', '') or 'Faixa sem erros metricos validos'
            if math.isfinite(lost_for) and lost_frames > self.loss_frames:
                self.status = f'Faixa perdida por mais de {self.loss_frames} quadros: {self.status}'
            elif lost_for > self.loss_tolerance:
                self.status = f'Faixa perdida por mais de {self.loss_tolerance:.2f} s: {self.status}'
        invalid = float('nan')
        lateral, heading, curvature = errors if errors is not None else (invalid, invalid, invalid)
        # line_valid continua sendo somente a faixa vista; durante a tolerancia
        # cross_track/heading registram a trajetoria prevista usada no controle.
        car.log(line_valid=int(line_seen), lateral_active=int(active),
                line_hold=int(holding and active), line_lost_s=lost_for,
                line_outlier=int(outlier), line_lost_frames=lost_frames,
                vision_frame_id=getattr(observation, 'sequence', 0),
                line_color_support=getattr(observation, 'color_support', invalid) if line_seen else invalid,
                line_dash_fraction=getattr(observation, 'dash_fraction', invalid) if line_seen else invalid,
                vision_age_s=age, cross_track_m=lateral, heading_error_rad=heading,
                vision_capture_s=getattr(observation, 'capture_seconds', invalid),
                vision_processing_s=getattr(observation, 'processing_seconds', invalid),
                vision_detection_s=getattr(observation, 'detection_seconds', invalid),
                vision_geometry_s=getattr(observation, 'geometry_seconds', invalid),
                curvature_m_inv=curvature, steer_geometric_rad=steer,
                steer_cmd_rad=float(car.st), steering_sign=float(self.steering_sign),
                steer_filtered_rad=getattr(car, 'st_filtered', invalid),
                stanley_heading_rad=self.controller.heading_term if active else invalid,
                stanley_cross_track_rad=self.controller.cross_track_term if active else invalid,
                stanley_feedforward_rad=self.controller.feedforward_term if active else invalid,
                stanley_damping_rad=self.controller.damping_term if active else invalid,
                line_search_rad=search,
                hold_speed_factor=self.hold_speed_factor if active else invalid,
                vision_compensated_s=compensated,
                curvature_raw_m_inv=curvature_raw, curve_speed_raw_m_s=speed_raw,
                curvature_filtered_m_inv=invalid if self.curvature is None else self.curvature,
                stanley_speed_m_s=float(stanley_speed) if active else invalid,
                stanley_use_vref=int(self.use_vref),
                stanley_raw_rad=self.controller.raw_steer if active else invalid,
                stanley_target_rad=self.controller.target_steer if active else invalid,
                steer_start_factor=steering_factor if active else invalid,
                pursuit_lookahead_m=self.pursuit_target[0] if active and self.pursuit_target else invalid,
                racing_offset_m=self.racing_offset if active else invalid,
                pursuit_target_y_m=self.pursuit_target[2] if active and self.pursuit_target else invalid,
                path_dkappa=self.path_dkappa,
                one_border_curve=int(self.one_border_curve),
                steer_trim_rad=self.steer_trim,
                preview_speed_m_s=self.preview[0] if active and self.preview else invalid,
                preview_curvature_m_inv=self.preview[1] if active and self.preview else invalid,
                vref=float(car.vref), u=float(car.u))
        self.last_state = dict(
            active=bool(active), hold=bool(holding and active), outlier=bool(outlier),
            lost_frames=lost_frames, lost_s=lost_for, line_seen=bool(line_seen),
            hold_speed_factor=self.hold_speed_factor if active else invalid,
            reference_speed_factor=self.line_speed_factor if active else invalid,
            speed_target=speed_raw, vref=float(car.vref), steer_rad=steer,
            steer_start_factor=steering_factor if active else invalid,
            steering_law='pure_pursuit' if active and self.pursuit_target else 'stanley',
            steer_trim_deg=round(math.degrees(self.steer_trim), 2),
            pursuit=None if not (active and self.pursuit_target) else [round(v, 3) for v in self.pursuit_target],
            preview=None if not (active and self.preview) else [round(v, 3) for v in self.preview],
            cross_track_m=lateral, heading_rad=heading, curvature_m_inv=curvature,
            compensated_s=compensated, status=self.status)
        return bool(active)
