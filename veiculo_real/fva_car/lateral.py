"""Stanley metrico para o carro real, sem importar sensores ou atuadores.

A homografia leva pixels da camera fixa ao plano do piso: x para frente e y
para esquerda, em metros, com origem no centro do eixo dianteiro. A selecao
da faixa e exatamente a implementacao compartilhada de tools.yellow_path.
"""
from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from tools.yellow_path import PathResult, PathSettings, YellowPathDetector, render_path


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

    @property
    def valid(self):
        try:
            errors = np.asarray(self.errors, dtype=float)
            return errors.shape == (3,) and bool(np.all(np.isfinite(errors)))
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True)
class GroundCalibration:
    image_size: tuple
    homography: np.ndarray

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
            matrix, _ = cv2.findHomography(pixels, ground, method=0)
        except cv2.error as error:
            raise ValueError('Nao foi possivel calcular a homografia da camera.') from error
        if matrix is None:
            raise ValueError('Correspondencias nao definem uma homografia valida.')
        result = cls(data['image_size'], matrix)
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


class RealLineDetector:
    def __init__(self, calibration, path_settings=None, anchor=.5, render_preview=True,
                 processing_width=0):
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
        self.detector = YellowPathDetector()
        self.detector.reset(anchor=self.anchor)

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
        processing_frame = frame
        if self.processing_size != self.calibration.image_size:
            processing_frame = cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        result = self.detector.process(processing_frame, self.settings, timestamp=timestamp)
        metadata = {'source_frame': frame, 'path_result': result,
                    'processing_size': self.processing_size,
                    'detection_seconds': time.perf_counter() - detection_at}
        display = None
        if self.render_preview:
            display = self._preview(frame, result)
        # A validacao de evidencia pertence ao detector compartilhado. O modo
        # pontilhado pode aceitar suporte < 50% apos validar as duas tangentes.
        if (not result.valid or not math.isfinite(result.support)
                or not 0 <= result.support <= 1):
            return LineObservation(timestamp, frame=display, reason=result.status, **metadata)
        geometry_at = time.perf_counter()
        path_pixels = cv2.countNonZero(result.path_mask)
        bridge_pixels = (0 if result.dash_mask is None else
                         cv2.countNonZero(cv2.bitwise_and(result.dash_mask, result.path_mask)))
        metadata.update(color_support=float(result.support),
                        dash_fraction=bridge_pixels / max(1, path_pixels))
        pixels = result.center
        if self.processing_size != self.calibration.image_size:
            # cv2.resize alinha centros de pixel. Retorna a grade original
            # ANTES da homografia, sem reutilizar pixels pequenos como grandes.
            scale = np.asarray(self.calibration.image_size) / self.processing_size
            pixels = (pixels.astype(float) + .5) * scale - .5
        points = self.calibration.project_pixels(pixels)
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

    def with_preview(self, observation):
        """Desenha sob demanda, sem refazer deteccao nem alterar o horario."""
        if (observation is None or observation.frame is not None
                or observation.source_frame is None or observation.path_result is None):
            return observation
        return replace(observation, frame=self._preview(observation.source_frame, observation.path_result))

    def preview_bgr(self, observation):
        """Deteccao desenhada na grade de processamento, em BGR (transmissao)."""
        if observation is None or observation.source_frame is None:
            return None
        if observation.path_result is None:
            return cv2.resize(observation.source_frame, self.processing_size, interpolation=cv2.INTER_AREA)
        return self._render_small(observation.source_frame, observation.path_result)

    def _render_small(self, frame, result):
        size = result.path_mask.shape[1::-1]
        source = frame if frame.shape[1::-1] == size else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return render_path(source, result, self.settings, self.anchor)

    def _preview(self, frame, result):
        display = self._render_small(frame, result)
        if display.shape[:2] != frame.shape[:2]:
            display = cv2.resize(display, frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


class StanleyController:
    def __init__(self, gain=1.5, softening=0.3, max_steer=math.radians(20),
                 max_rate=math.radians(60), wheelbase=.36, curvature_gain=0.):
        if not all(math.isfinite(x) and x > 0
                   for x in (gain, softening, max_steer, max_rate, wheelbase)):
            raise ValueError('Parametros Stanley devem ser finitos e positivos.')
        if max_steer >= math.pi / 2:
            raise ValueError('max_steer deve ser menor que pi/2 radianos.')
        if not math.isfinite(curvature_gain) or curvature_gain < 0:
            raise ValueError('O ganho de curvatura deve ser finito e nao negativo.')
        self.gain, self.softening = gain, softening
        self.max_steer, self.max_rate = max_steer, max_rate
        self.wheelbase, self.curvature_gain = wheelbase, curvature_gain
        self.reset()

    def reset(self):
        self.steer = 0.
        self.heading_term = 0.
        self.cross_track_term = 0.
        self.feedforward_term = 0.
        self.raw_steer = 0.
        self.target_steer = 0.

    def update(self, cross_track, heading_error, speed, dt, curvature=0.):
        """cross_track: linha a esquerda do eixo dianteiro (+), em metros.

        heading_error = direcao da linha - direcao do carro, em radianos.
        curvature: da linha, positiva para a esquerda, em 1/m.
        """
        if (not all(math.isfinite(x) for x in (cross_track, heading_error, speed, dt, curvature))
                or dt <= 0):
            raise ValueError('Stanley requer sinais finitos e dt positivo.')
        self.heading_term = math.atan2(math.sin(heading_error), math.cos(heading_error))
        self.cross_track_term = math.atan2(self.gain * cross_track,
                                          abs(speed) + self.softening)
        # Esterco de regime da curva (bicicleta): sem ele o Stanley so vira
        # depois de acumular erro, tarde demais em velocidade alta.
        self.feedforward_term = math.atan(self.curvature_gain * self.wheelbase * curvature)
        self.raw_steer = self.heading_term + self.cross_track_term + self.feedforward_term
        self.target_steer = float(np.clip(self.raw_steer, -self.max_steer, self.max_steer))
        self.steer += float(np.clip(self.target_steer - self.steer,
                                    -self.max_rate * dt, self.max_rate * dt))
        return self.steer



def fit_line_errors(points, front, forward, max_distance=4., with_curvature=False):
    """Erros na projecao normal do eixo dianteiro sobre a curva local."""
    forward = np.asarray(forward, dtype=float)[:2]
    forward = forward / np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    relative = np.asarray(points)[:, :2] - np.asarray(front)[:2]
    x, y = relative @ forward, relative @ left
    valid = (x >= -.35) & (x <= max_distance) & (np.abs(y) <= 2.)
    x, y = x[valid], y[valid]
    if len(x) < 6 or np.ptp(x) < .3 or np.min(x) > 1.5:
        return None
    # Aproximacao local; nao extrapolar uma curva distante ate o veiculo.
    nearby = x <= min(max_distance, max(1.2, float(np.min(x)) + 1.))
    x, y = x[nearby], y[nearby]
    if len(x) < 6 or np.ptp(x) < .3:
        return None
    # Reta secante em uma curva inventava deslocamento lateral em x=0.
    # Parabola separa deslocamento, tangente e curvatura da linha.
    coefficients = np.polyfit(x, y, 2)
    residual = y - np.polyval(coefficients, x)
    keep = np.abs(residual) <= max(.04, 3 * float(np.median(np.abs(residual))))
    if np.count_nonzero(keep) < 6 or np.ptp(x[keep]) < .3:
        return None
    coefficients = np.polyfit(x[keep], y[keep], 2)
    if np.sqrt(np.mean((y[keep] - np.polyval(coefficients, x[keep]))**2)) > .12:
        return None
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


def curve_speed(reference, curvature, lateral, heading, min_speed=.5,
                lateral_accel=.6):
    """Limita v por v^2*|kappa| e reduz avanco durante recuperacao de erro."""
    if reference <= 0 or min_speed <= 0 or lateral_accel <= 0:
        raise ValueError('Velocidades e aceleracao lateral devem ser positivas.')
    bend_limit = math.sqrt(lateral_accel / max(abs(curvature), 1e-6))
    recovery_limit = reference / (1 + 2*abs(lateral) + abs(heading))
    return min(reference, max(min_speed, min(bend_limit, recovery_limit)))


MAX_LATENCY_COMPENSATION = .25  # s; idade tipica da imagem: 0,12 a 0,19 s


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
        self.loss_tolerance = nonnegative('line_loss_tolerance', .5)
        self.reference = positive('velocity_reference', .5)
        self.min_speed = positive('curve_min_speed', .3)
        self.lateral_accel = positive('curve_lateral_accel', .4)
        self.use_vref = self.settings.get('stanley_use_vref', True)
        if not isinstance(self.use_vref, (bool, np.bool_)):
            raise ValueError('stanley_use_vref deve ser True ou False.')
        self.steering_sign = self.settings.get('steering_sign', 1.)
        if self.steering_sign not in (-1., 1.):
            raise ValueError('steering_sign deve ser -1 ou +1; calibre o sinal do carro real.')
        self.controller = StanleyController(
            gain=positive('stanley_gain', 1.5),
            softening=positive('stanley_softening', .3),
            max_steer=math.radians(min(20., positive('stanley_max_steer_deg', 20.))),
            max_rate=math.radians(positive('stanley_max_rate_deg_s', 60.)),
            wheelbase=positive('wheelbase', .36),
            curvature_gain=nonnegative('stanley_curvature_ff', 1.))
        self.compensate_latency = self.settings.get('vision_latency_compensation', True)
        if not isinstance(self.compensate_latency, (bool, np.bool_)):
            raise ValueError('vision_latency_compensation deve ser True ou False.')
        self.reset()

    def reset(self):
        """Neutraliza o Stanley e esquece a ultima faixa vista."""
        self.controller.reset()
        self.line_memory = None
        self.line_seen_at = None

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

    def apply(self, car, observation, now, obstacle_detected=False):
        """Retorna True quando uma observacao atual, ou a faixa vista ha no maximo
        line_loss_tolerance segundos, autoriza seguir."""
        age = float('nan')
        errors = None
        stanley_speed = float('nan')
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
        holding = False
        lost_for = float('nan')
        compensated = 0.
        if line_seen and self.compensate_latency and finite_sample and age > 0:
            # Os erros sao do instante da captura; a 1,5 m/s o carro anda
            # ~20 cm ate o comando. Leva a faixa ao instante atual, com teto
            # para nao extrapolar a curvatura alem do trecho realmente visto.
            compensated = min(age, MAX_LATENCY_COMPENSATION)
            errors = self._predict_line(errors, car, compensated)
        if line_seen:
            self.line_memory, self.line_seen_at = errors, float(now)
        elif (self.line_memory is not None and finite_sample
                and isinstance(now, (int, float, np.number)) and math.isfinite(now)):
            lost_for = float(now) - self.line_seen_at
            if 0 <= lost_for <= self.loss_tolerance:
                # Reflexo ou sujeira raramente apagam a faixa de verdade: segue a
                # ultima trajetoria vista, por odometria, e so para se a perda durar.
                errors = self.line_memory = self._predict_line(self.line_memory, car)
                holding = True
        active = (not car.emergencia and not obstacle_detected and finite_sample
                  and getattr(car, 'encoder_valid', True)
                  and not reverse_gear and not reverse_requested
                  and car.v >= -.05 and errors is not None)
        if active:
            lateral, heading, curvature = errors
            # Primeiro resolve a referencia deste passo (inclui curvas e RC).
            # Usar car.vref antes de set_vel leria a referencia anterior,
            # frequentemente zero no inicio ou apos uma perda de faixa.
            car.set_vel(curve_speed(self.reference, curvature, lateral, heading,
                                    min_speed=self.min_speed, lateral_accel=self.lateral_accel))
            active = bool(getattr(car, 'encoder_valid', True))
            stanley_speed = car.vref if self.use_vref else car.v
            reference_valid = (isinstance(stanley_speed, (int, float, np.number))
                               and math.isfinite(stanley_speed))
            active = active and reference_valid
            if active:
                steer = self.controller.update(lateral, heading, stanley_speed, car.dt, curvature)
                car.set_steer(self.steering_sign * steer)
                # O setter revalida a serial antes de enviar a direcao.
                active = bool(getattr(car, 'encoder_valid', True))
        if not active:
            car._set_ref(0.)
            car.vel_error_integral = 0.
            car.set_u(0.)
            car.set_steer(0.)
            self.reset()
            steer = 0.
        if active and holding:
            self.status = (f'Faixa perdida ha {lost_for:.2f} s; seguindo ultima trajetoria'
                           f' (limite {self.loss_tolerance:.2f} s)')
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
        elif reverse_gear or reverse_requested or car.v < -.05:
            self.status = 'Marcha a re: Stanley dianteiro bloqueado'
        elif observation is None:
            self.status = 'Aguardando primeira imagem'
        elif not math.isfinite(age) or age < 0:
            self.status = 'Horario de imagem invalido'
        elif age > self.timeout:
            self.status = f'Imagem atrasada: {age:.3f} s; limite {self.timeout:.3f} s'
        else:
            self.status = getattr(observation, 'reason', '') or 'Faixa sem erros metricos validos'
            if lost_for > self.loss_tolerance:
                self.status = f'Faixa perdida por mais de {self.loss_tolerance:.2f} s: {self.status}'
        invalid = float('nan')
        lateral, heading, curvature = errors if errors is not None else (invalid, invalid, invalid)
        # line_valid continua sendo somente a faixa vista; durante a tolerancia
        # cross_track/heading registram a trajetoria prevista usada no controle.
        car.log(line_valid=int(line_seen), lateral_active=int(active),
                line_hold=int(holding and active), line_lost_s=lost_for,
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
                vision_compensated_s=compensated,
                stanley_speed_m_s=float(stanley_speed) if active else invalid,
                stanley_use_vref=int(self.use_vref),
                stanley_raw_rad=self.controller.raw_steer if active else invalid,
                stanley_target_rad=self.controller.target_steer if active else invalid,
                vref=float(car.vref), u=float(car.u))
        return bool(active)
