"""Modelo bicicleta e Stanley: coordenadas locais (frente, esquerda), SI."""
from dataclasses import dataclass
import math
from collections import deque
from pathlib import Path

import numpy as np
from matplotlib.colors import rgb_to_hsv


# H em graus (0 a 360), S e V entre 0 e 1.
YELLOW_HSV_LOWER = (40., .4, .2)
YELLOW_HSV_UPPER = (75., 1., 1.)


@dataclass
class BicycleModel:
    wheelbase: float = 0.302

    def yaw_rate(self, speed, steer):
        return speed * math.tan(steer) / self.wheelbase

    def derivative(self, state, speed, steer):
        """Estado (x, y, psi) no centro do eixo traseiro."""
        _, _, psi = state
        return np.array([speed * math.cos(psi), speed * math.sin(psi),
                         self.yaw_rate(speed, steer)])

    def linearize(self, speed):
        """Estado (e_y, e_psi), e_y positivo quando carro esta a esquerda."""
        return (np.array([[0., speed], [0., 0.]]),
                np.array([[0.], [speed / self.wheelbase]]))


class StanleyController:
    def __init__(self, gain=1.5, softening=0.3, max_steer=math.radians(20),
                 max_rate=math.radians(60)):
        if not all(math.isfinite(x) and x > 0
                   for x in (gain, softening, max_steer, max_rate)):
            raise ValueError('Parametros Stanley devem ser finitos e positivos.')
        if max_steer >= math.pi / 2:
            raise ValueError('max_steer deve ser menor que pi/2 radianos.')
        self.gain, self.softening = gain, softening
        self.max_steer, self.max_rate = max_steer, max_rate
        self.steer = 0.
        self.heading_term = 0.
        self.cross_track_term = 0.
        self.raw_steer = 0.
        self.target_steer = 0.

    def update(self, cross_track, heading_error, speed, dt):
        """cross_track: linha a esquerda do eixo dianteiro (+), em metros.

        heading_error = direcao da linha - direcao do carro, em radianos.
        """
        if not all(math.isfinite(x) for x in (cross_track, heading_error, speed, dt)) or dt <= 0:
            raise ValueError('Stanley requer sinais finitos e dt positivo.')
        self.heading_term = math.atan2(math.sin(heading_error), math.cos(heading_error))
        self.cross_track_term = math.atan2(self.gain * cross_track,
                                          abs(speed) + self.softening)
        self.raw_steer = self.heading_term + self.cross_track_term
        self.target_steer = float(np.clip(self.raw_steer, -self.max_steer, self.max_steer))
        self.steer += float(np.clip(self.target_steer - self.steer,
                                    -self.max_rate * dt, self.max_rate * dt))
        return self.steer


def yellow_centers(image, hsv_lower=YELLOW_HSV_LOWER, hsv_upper=YELLOW_HSV_UPPER):
    """Segmenta amarelo em HSV e retorna centros por linha da imagem.

    Entrada RGB uint8 da camera; H em graus (0..360), S e V em 0..1.
    Os limites HSV sao inclusivos; nao usam a escala H=0..179 do OpenCV.
    """
    lower, upper = np.asarray(hsv_lower, dtype=float), np.asarray(hsv_upper, dtype=float)
    scale = np.array([360., 1., 1.])
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower < 0) or np.any(upper > scale) or np.any(lower > upper)):
        raise ValueError('Limites HSV: H em 0..360 graus, S/V em 0..1 e minimo <= maximo.')
    hsv = rgb_to_hsv(np.asarray(image, dtype=float) / 255.)
    hsv[..., 0] *= 360.
    mask = np.all((hsv >= lower) & (hsv <= upper), axis=-1)
    centers = []
    previous = image.shape[1] / 2
    # Caminha de baixo para cima para manter continuidade e evitar juntar ramos.
    for row in range(image.shape[0] - 2, int(image.shape[0] * .25), -3):
        columns = np.flatnonzero(mask[row])
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        groups = [group for group in groups if len(group) >= 2]
        if not groups:
            continue
        center = min((float(np.mean(group)) for group in groups), key=lambda x: abs(x-previous))
        if centers and abs(center-previous) > image.shape[1] * .2:
            continue
        centers.append((center, row))
        previous = center
    return np.asarray(centers, dtype=float).reshape(-1, 2)


def project_ground(pixels, shape, camera_matrix, perspective_angle, ground_z):
    """Imagem ja flipud de Car.get_image; camera Coppelia olha em +Z.

    O eixo X da camera aponta para a esquerda da imagem e Y para cima.
    Intersecao dos raios com plano horizontal conhecido (pista plana).
    """
    height, width = shape[:2]
    focal = max(width, height) / (2 * math.tan(perspective_angle / 2))
    matrix = np.asarray(camera_matrix).reshape(3, 4)
    rays = np.column_stack(((width/2 - pixels[:, 0] - .5) / focal,
                            (height/2 - pixels[:, 1] - .5) / focal,
                            np.ones(len(pixels)))) @ matrix[:, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = (ground_z - matrix[2, 3]) / rays[:, 2]
    valid = np.isfinite(distance) & (distance > 0) & (distance < 15)
    return matrix[:, 3] + rays[valid] * distance[valid, None]


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



class LocalLineMemory:
    """Pontos medidos no mundo, com validade limitada e densidade uniforme."""
    def __init__(self, horizon=3.):
        self.horizon = horizon
        self.frames = deque()
        self.last_seen = -math.inf

    def update(self, time, points, front):
        while self.frames and time-self.frames[0][0] > self.horizon:
            self.frames.popleft()
        points = np.asarray(points).reshape(-1, 3)
        if len(points) >= 6:
            # Guarda somente o caminho proximo, evitando acumular outros ramos.
            close = np.linalg.norm(points[:, :2]-np.asarray(front)[:2], axis=1) <= 3.
            if np.count_nonzero(close) >= 6:
                self.frames.append((time, points[close].copy()))
                self.last_seen = time
        # Memoria nao autoriza seguir indefinidamente sem deteccao atual.
        if not self.frames or time-self.last_seen > .3:
            return np.empty((0, 3))
        cloud = np.concatenate([frame[1] for frame in self.frames])
        # Uma amostra por celula de 4 cm impede imagens repetidas de dominar.
        _, indices = np.unique(np.round(cloud[:, :2]/.04), axis=0, return_index=True)
        return cloud[indices]


class YellowLineFollower:
    """Percepcao metrica e comando; uma unica aquisicao de imagem por passo."""
    def __init__(self, car):
        self.car = car
        max_steer_deg = car.parameters.get('stanley_max_steer_deg', 20.)
        if not math.isfinite(max_steer_deg):
            raise ValueError('stanley_max_steer_deg deve ser finito.')
        self.controller = StanleyController(
            gain=car.parameters.get('stanley_gain', 1.5),
            softening=car.parameters.get('stanley_softening', .3),
            # O atuador Car aceita no maximo 20 graus; respeita limites menores.
            max_steer=math.radians(min(max_steer_deg, 20.)),
            max_rate=math.radians(car.parameters.get('stanley_max_rate_deg_s', 60.)))
        # Na cena Car, juntas positivas viram para a direita. Stanley usa
        # angulo geometrico positivo para a esquerda (confirmado pelo log).
        self.steering_sign = car.parameters.get('steering_sign', -1.)
        if self.steering_sign not in (-1., 1.):
            raise ValueError('steering_sign deve ser -1 ou +1.')
        sim = car.sim
        if not sim.getObjectInt32Param(car.cam, sim.visionintparam_perspective_operation):
            raise ValueError('Seguidor requer camera em perspectiva.')
        self.angle = sim.getObjectFloatParam(car.cam, sim.visionfloatparam_perspective_angle)
        self.ground_z = car.parameters.get('line_ground_z', .005)
        self.yellow_hsv_lower = car.parameters.get('yellow_hsv_lower', YELLOW_HSV_LOWER)
        self.yellow_hsv_upper = car.parameters.get('yellow_hsv_upper', YELLOW_HSV_UPPER)
        # Centro entre juntas de direcao: evita assumir a origem de /Car no eixo traseiro.
        left = np.asarray(sim.getObjectPosition(car.steerL, car.robot))
        right = np.asarray(sim.getObjectPosition(car.steerR, car.robot))
        self.front_local = (left + right) / 2
        self.last_valid_time = None
        self.line_memory = LocalLineMemory()
        self.diagnostic_frames = deque(maxlen=20)
        self.last_diagnostic_time = -math.inf
        self.was_valid = False

    def __call__(self, car):
        car.line_image = car.get_image(gray=False)
        pixels = yellow_centers(car.line_image, self.yellow_hsv_lower, self.yellow_hsv_upper)
        sim = car.sim
        matrix = np.asarray(sim.getObjectMatrix(car.robot, -1)).reshape(3, 4)
        front = matrix[:, :3] @ self.front_local + matrix[:, 3]
        camera_matrix = sim.getObjectMatrix(car.cam, -1)
        points = project_ground(pixels, car.line_image.shape,
                                camera_matrix, self.angle, self.ground_z)
        path_points = self.line_memory.update(car.t, points, front)
        errors = fit_line_errors(path_points, front, matrix[:, 2], with_curvature=True)
        if car.t-self.last_diagnostic_time >= .15:
            self.diagnostic_frames.append(dict(
                time=car.t, image=car.line_image.copy(), pixels=pixels.copy(),
                points=points.copy(), path=path_points.copy(), front=front.copy(),
                forward=matrix[:, 2].copy(), camera=np.asarray(camera_matrix)))
            self.last_diagnostic_time = car.t
        if errors is None and self.was_valid and isinstance(getattr(car, 'logfile', None), str):
            # Arquivo numerico + imagens originais; permite reproduzir a percepcao.
            payload = {f'{key}_{i}': value for i, frame in enumerate(self.diagnostic_frames)
                       for key, value in frame.items()}
            np.savez_compressed(Path(car.logfile)/f'line_loss_{car.t:.2f}.npz', **payload)
        self.was_valid = errors is not None
        if errors is None:
            # Parada ativa pelo PI longitudinal; nao segue avancando com erro antigo.
            car.set_vel(0.)
            # Nao desfaz a curva enquanto o carro ainda freia. Memoria limitada,
            # sem continuar avancando a partir de uma linha antiga.
            holding = (self.last_valid_time is not None
                       and car.t-self.last_valid_time <= .4 and abs(car.v) > .05)
            steer = self.controller.steer if holding else 0.
            car.set_steer(self.steering_sign * steer)
            if not holding:
                self.controller.steer = 0.
            lateral, heading, curvature = float('nan'), float('nan'), float('nan')
        else:
            self.last_valid_time = car.t
            lateral, heading, curvature = errors
            steer = self.controller.update(lateral, heading, car.v, car.dt)
            car.set_steer(self.steering_sign * steer)
            car.set_vel(curve_speed(
                car.parameters.get('velocity_reference', 1.5), curvature, lateral, heading,
                min_speed=car.parameters.get('curve_min_speed', .5),
                lateral_accel=car.parameters.get('curve_lateral_accel', .6)))
        car.log(line_valid=int(errors is not None), cross_track_m=lateral,
                curvature_m_inv=curvature,
                stanley_heading_rad=self.controller.heading_term if errors is not None else float('nan'),
                stanley_cross_track_rad=self.controller.cross_track_term if errors is not None else float('nan'),
                stanley_raw_rad=self.controller.raw_steer if errors is not None else float('nan'),
                stanley_target_rad=self.controller.target_steer if errors is not None else float('nan'),
                steer_geometric_rad=steer, steering_sign=self.steering_sign,
                line_pixels=len(pixels), line_ground_points=len(points),
                line_memory_points=len(path_points),
                heading_error_rad=heading, steer_cmd_rad=float(car.st),
                steer_left_rad=sim.getJointPosition(car.steerL),
                steer_right_rad=sim.getJointPosition(car.steerR),
                vref=car.vref, u=car.u, motor_pwm_percent=car.motor_pwm_percent)
