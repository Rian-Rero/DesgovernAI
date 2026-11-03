"""Hierarquia visual: corredor de cones, faixa amarela, parada ativa.

As duas referencias produzem os mesmos erros metricos para um unico Stanley.
Nenhuma posicao de cone da cena e consultada pelo seguidor.
"""
import math

import cv2
import numpy as np

from .cones import ConePathDetector
from .lateral import YellowLineFollower, curve_speed, fit_line_errors, project_ground, yellow_centers

PATH_SOURCE_CODES = {'none': 0, 'yellow': 1, 'cones': 2}


def _finite_errors(errors):
    return errors is not None and len(errors) == 3 and all(math.isfinite(x) for x in errors)


def _image_points(points, shape, camera_matrix, angle):
    """Projeta o caminho medido no mundo de volta na imagem RGB da camera."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    transform = np.asarray(camera_matrix, dtype=float).reshape(3, 4)
    camera = (points - transform[:, 3]) @ transform[:, :3]
    camera = camera[np.all(np.isfinite(camera), axis=1) & (camera[:, 2] > .02)]
    height, width = shape[:2]
    focal = max(width, height) / (2 * math.tan(angle / 2))
    pixels = np.column_stack((width/2 - .5 - focal*camera[:, 0]/camera[:, 2],
                              height/2 - .5 - focal*camera[:, 1]/camera[:, 2]))
    pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)]
    return np.rint(pixels).astype(np.int32)


class HierarchicalPathFollower(YellowLineFollower):
    """Uma captura, escolha de referencia, Stanley e controle de velocidade."""
    def __init__(self, car):
        # Compartilha calibracao e convencoes do seguidor amarelo existente.
        # __call__ e independente: o pai nunca captura ou envia comandos aqui.
        super().__init__(car)
        self.cones = ConePathDetector(car.parameters)
        self.cone_ground_z = car.parameters.get('cone_ground_z', self.ground_z)
        self.source = 'none'
        self.observation = None

    def __call__(self, car):
        image = car.get_image(gray=False)
        sim = car.sim
        matrix = np.asarray(sim.getObjectMatrix(car.robot, -1)).reshape(3, 4)
        front = matrix[:, :3] @ self.front_local + matrix[:, 3]
        forward = matrix[:, 2]
        camera_matrix = sim.getObjectMatrix(car.cam, -1)
        image_valid = (isinstance(image, np.ndarray) and image.ndim == 3
                       and image.shape[2] == 3 and image.size > 0)
        observation = None
        errors = None
        source = 'none'
        pixels = np.empty((0, 2))
        points = path = np.empty((0, 3))
        yellow_evaluated = False
        if image_valid:
            observation = self.cones.observe(image, camera_matrix, self.angle,
                                             self.cone_ground_z, front, forward, car.t)
            if _finite_errors(observation.errors):
                errors, path, source = observation.errors, observation.path, 'cones'
            else:
                # A faixa nao compete com um corredor de cones valido.
                yellow_evaluated = True
                pixels = yellow_centers(image, self.yellow_hsv_lower, self.yellow_hsv_upper)
                points = project_ground(pixels, image.shape, camera_matrix,
                                        self.angle, self.ground_z)
                path = self.line_memory.update(car.t, points, front)
                candidate = fit_line_errors(path, front, forward, with_curvature=True)
                if _finite_errors(candidate):
                    errors, source = candidate, 'yellow'
        else:
            # Expira a memoria mesmo enquanto a camera nao entrega imagens.
            self.line_memory.update(car.t, points, front)
        self.observation = observation
        self.source = car.path_source = source
        cone_reason = observation.reason if observation is not None else 'Camera sem imagem'
        car.path_reason = ('Faixa amarela valida; ' + cone_reason
                           if source == 'yellow' else cone_reason)
        valid = errors is not None
        if valid:
            self.last_valid_time = car.t
            lateral, heading, curvature = errors
            steer = self.controller.update(lateral, heading, car.v, car.dt)
            car.set_steer(self.steering_sign * steer)
            car.set_vel(curve_speed(
                car.parameters.get('velocity_reference', 1.5), curvature, lateral, heading,
                min_speed=car.parameters.get('curve_min_speed', .5),
                lateral_accel=car.parameters.get('curve_lateral_accel', .6)))
        else:
            car.set_vel(0.)
            holding = (self.last_valid_time is not None
                       and car.t-self.last_valid_time <= .4 and abs(car.v) > .05)
            steer = self.controller.steer if holding else 0.
            car.set_steer(self.steering_sign * steer)
            if not holding:
                self.controller.steer = 0.
            lateral = heading = curvature = float('nan')
        self.was_valid = valid
        if image_valid:
            car.line_image = self._overlay(image, observation, path if valid else np.empty((0, 3)),
                                           camera_matrix, source, lateral)
        else:
            car.line_image = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(car.line_image, 'CAMERA SEM IMAGEM - PARANDO', (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 80, 80), 1, cv2.LINE_AA)
        car.log(
            path_valid=int(valid), path_source=PATH_SOURCE_CODES[source],
            cones_valid=int(observation is not None and _finite_errors(observation.errors)),
            cone_inferred_side={'red': 1, 'green': 2}.get(getattr(observation, 'inferred_color', None), 0),
            line_valid=int(source == 'yellow'), yellow_evaluated=int(yellow_evaluated),
            red_cones=len(observation.red_pixels) if observation is not None else 0,
            green_cones=len(observation.green_pixels) if observation is not None else 0,
            red_lateral_m=observation.red_lateral_m if observation is not None else float('nan'),
            green_lateral_m=observation.green_lateral_m if observation is not None else float('nan'),
            cross_track_m=lateral, heading_error_rad=heading, curvature_m_inv=curvature,
            stanley_heading_rad=self.controller.heading_term if valid else float('nan'),
            stanley_cross_track_rad=self.controller.cross_track_term if valid else float('nan'),
            stanley_raw_rad=self.controller.raw_steer if valid else float('nan'),
            stanley_target_rad=self.controller.target_steer if valid else float('nan'),
            steer_geometric_rad=steer, steering_sign=self.steering_sign,
            line_pixels=len(pixels), line_ground_points=len(points),
            line_memory_points=len(path) if source == 'yellow' else 0,
            steer_cmd_rad=float(car.st),
            steer_left_rad=sim.getJointPosition(car.steerL),
            steer_right_rad=sim.getJointPosition(car.steerR),
            vref=car.vref, u=car.u, motor_pwm_percent=car.motor_pwm_percent)

    def _overlay(self, image, observation, path, camera_matrix, source, lateral):
        image = image.copy()
        if observation is not None:
            for pixels, color in ((observation.red_pixels, (255, 70, 70)),
                                  (observation.green_pixels, (70, 255, 70))):
                for pixel in pixels:
                    if np.all(np.isfinite(pixel)):
                        cv2.circle(image, tuple(np.rint(pixel).astype(int)), 4, color, 1, cv2.LINE_AA)
        pixels = _image_points(path, image.shape, camera_matrix, self.angle)
        if len(pixels) >= 2:
            cv2.polylines(image, [pixels], False, (70, 220, 255), 2, cv2.LINE_AA)
        labels = {'cones': 'CONES: centro do corredor',
                  'yellow': 'FAIXA AMARELA: alternativa', 'none': 'SEM REFERENCIA: PARANDO'}
        lines = [labels[source]]
        if math.isfinite(lateral):
            lines.append(f'Erro lateral: {lateral:+.3f} m')
        if observation is not None:
            lines.append(f'Vermelhos: {observation.red_lateral_m:+.2f} m | '
                         f'Verdes: {observation.green_lateral_m:+.2f} m')
            if getattr(observation, 'inferred_color', None):
                lines.append('Borda oculta estimada pela largura observada')
        for index, text in enumerate(lines):
            origin = (8, 18 + index*18)
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
        return image
