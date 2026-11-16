# -*- coding: utf-8 -*-
# Disciplina: Fundamentos de Veiculos Autonomos - 2026/2
# Parte 2: controle lateral (on-off e Stanley) com o PI longitudinal da Parte 1.
#
# Referencias:
#   reta | circulo | oito : trajetoria geometrica; erros calculados com a pose
#                           verdadeira do simulador (sem camera).
#   linha                 : linha amarela da pista detectada pela camera do
#                           carro (HSV + projecao no chao). Nenhuma informacao
#                           da cena e usada pelo controlador; a textura da pista
#                           so e lida para AVALIAR o erro verdadeiro e detectar
#                           saida da pista.
# Controladores:
#   onoff   : rele de tres niveis com histerese sobre o erro de orientacao
#             e = e_psi + atan(e_y / L_d)   (fva_car/lateral.py: OnOffSteering)
#   stanley : delta = e_psi + atan(k e_y / (v + v_s))  (fva_car/lateral.py)
#
# Uso (CoppeliaSim aberto com simulador_cones.ttt e simulacao parada, na raiz):
#   venv/bin/python simulador/controle_lateral.py --referencia oito --controlador onoff --vref 1.0
#   venv/bin/python simulador/controle_lateral.py --referencia linha --controlador stanley --vref 1.5
# Saida: <saida>/<data>/car.csv e resumo.json (metricas).
# A cena nunca e salva; a pose original do carro e restaurada ao final.
########################################
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fva_car import Car
from fva_car.car import CAR
from fva_car.lateral import (
    LocalLineMemory, OnOffSteering, StanleyController, curve_speed,
    fit_line_errors, orientation_reference, project_ground, yellow_centers,
)
from fva_car.trajetorias import circle, figure_eight, straight
from main import parameters as MAIN_PARAMETERS

STEERING_SIGN = -1.0  # na cena, set_steer positivo vira para a direita
TILT_LIMIT = math.radians(45.)  # acima disso o carro capotou
OFF_TRACK_LIMIT = 1.0  # m da linha amarela verdadeira: saiu da pista
LOST_LIMIT = 1.0  # s sem detectar a linha: perdido
# Poses iniciais (x, y, yaw em graus) na regiao sem cones da pista.
GEOMETRIC_POSES = {"reta": (-30., 0., 0.), "circulo": (-14., 0., 0.), "oito": (-14., 0., 0.)}


########################################
# pose do carro (somente com a simulacao parada)
def place_car(sim, robot, pose):
    original = sim.getObjectMatrix(robot, -1)
    matrix = np.asarray(original).reshape(3, 4).copy()
    heading = math.atan2(matrix[1, 2], matrix[0, 2])  # eixo Z local = frente
    angle = math.radians(pose[2]) - heading
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    matrix[:, :3] = rotation @ matrix[:, :3]
    matrix[0, 3], matrix[1, 3] = pose[0], pose[1]
    sim.setObjectMatrix(robot, matrix.ravel().tolist(), -1)
    return original


########################################
# referencia verdadeira (somente avaliacao): esqueleto da linha na textura.
# Adaptado de simulador/avaliar_stanley.py (branch latero-Rian).
class TrackReference:
    def __init__(self, sim):
        import cv2
        pista = sim.getObject('/Pista')
        data, object_matrix = sim.getShapeViz(pista, 0), sim.getObjectMatrix(pista, -1)
        texture = data['texture']
        width, height = texture['resolution']
        rgb = np.frombuffer(texture['texture'], dtype=np.uint8).reshape(height, width, 4)[::-1, :, :3]
        matrix = np.asarray(object_matrix, dtype=float).reshape(3, 4)
        world = (np.asarray(data['vertices']).reshape(-1, 3)[data['indices']]
                 @ matrix[:, :3].T + matrix[:, 3])
        uv = np.asarray(texture['coordinates']).reshape(-1, 2)
        design = np.column_stack((uv, np.ones(len(uv))))
        mapping = np.linalg.lstsq(design, world[:, :2], rcond=None)[0].T
        pixel_to_uv = np.array([[1 / width, 0, .5 / width],
                                [0, -1 / height, 1 - .5 / height], [0, 0, 1]])
        pixel_to_world = np.vstack((mapping, [0, 0, 1])) @ pixel_to_uv
        self.scale = float(np.linalg.norm(pixel_to_world[:2, 0]))
        self.world_to_pixel = np.linalg.inv(pixel_to_world)
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([17, 90, 65], np.uint8),
                           np.array([38, 255, 255], np.uint8))
        skeleton = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        remaining = mask.copy()
        while cv2.countNonZero(remaining):
            opened = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, element)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
            remaining = cv2.erode(remaining, element)
        self.skeleton = skeleton
        self.distance = cv2.distanceTransform(255 - skeleton, cv2.DIST_L2,
                                              cv2.DIST_MASK_PRECISE) * self.scale
        self.rgb = rgb

    def to_pixel(self, xy):
        p = self.world_to_pixel @ np.array([xy[0], xy[1], 1.])
        return p[:2] / p[2]

    def error(self, xy):
        u, v = self.to_pixel(xy)
        h, w = self.distance.shape
        if not (0 <= u < w - 1 and 0 <= v < h - 1):
            return math.nan
        i, j = int(v), int(u)
        a, b = v - i, u - j
        d = self.distance
        return float((1 - a) * ((1 - b) * d[i, j] + b * d[i, j + 1])
                     + a * ((1 - b) * d[i + 1, j] + b * d[i + 1, j + 1]))


########################################
# percepcao da linha amarela pela camera (mesma cadeia de YellowLineFollower)
class LinePerception:
    def __init__(self, car):
        sim = car.sim
        self.angle = sim.getObjectFloatParam(car.cam, sim.visionfloatparam_perspective_angle)
        self.ground_z = MAIN_PARAMETERS.get('line_ground_z', .005)
        self.hsv_lower = MAIN_PARAMETERS['yellow_hsv_lower']
        self.hsv_upper = MAIN_PARAMETERS['yellow_hsv_upper']
        self.memory = LocalLineMemory()

    def __call__(self, car, front, forward):
        image = car.get_image(gray=False)
        pixels = yellow_centers(image, self.hsv_lower, self.hsv_upper)
        points = project_ground(pixels, image.shape, car.sim.getObjectMatrix(car.cam, -1),
                                self.angle, self.ground_z)
        path = self.memory.update(car.t, points, front)
        return fit_line_errors(path, front, forward, with_curvature=True), len(pixels)


########################################
class LateralMission:
    def __init__(self, args):
        self.args = args
        if args.controlador == 'onoff':
            self.controller = OnOffSteering(
                delta_on=math.radians(args.delta_on), h_on=math.radians(args.h_on),
                h_off=math.radians(args.h_off), max_rate=math.radians(args.taxa))
        else:
            self.controller = StanleyController(
                gain=args.ganho, softening=args.suavizacao,
                max_steer=CAR['STEERMAX'], max_rate=math.radians(args.taxa))
        self.path = None
        self.perception = None
        self.track = None
        self.last_valid = None
        self.lost_since = None
        self.failure = None
        self.completed = False
        self.distance = 0.
        self.previous_xy = None
        self.start_xy = None

    def setup(self, car):
        sim = car.sim
        left = np.asarray(sim.getObjectPosition(car.steerL, car.robot))
        right = np.asarray(sim.getObjectPosition(car.steerR, car.robot))
        self.front_local = (left + right) / 2
        ref = self.args.referencia
        if ref == 'linha':
            self.perception = LinePerception(car)
            self.track = TrackReference(sim)
            self.original_pose = None
        else:
            pose = self.args.pose or GEOMETRIC_POSES[ref]
            self.original_pose = place_car(sim, car.robot, pose)
            yaw = math.radians(pose[2])
            front0 = np.array(pose[:2]) + self.front_local[2] * np.array([math.cos(yaw), math.sin(yaw)])
            builder = {'reta': lambda: straight(front0, yaw, offset=self.args.offset),
                       'circulo': lambda: circle(front0, yaw, self.args.raio),
                       'oito': lambda: figure_eight(front0, yaw, self.args.raio)}[ref]
            self.path = builder()

    def restore(self, car):
        if self.original_pose is not None:
            car.sim.setObjectMatrix(car.robot, self.original_pose, -1)

    def steer(self, car, e_y, e_psi):
        if isinstance(self.controller, OnOffSteering):
            error = orientation_reference(e_y, e_psi, self.args.lookahead)
            delta = self.controller.update(error, car.dt)
            return delta, dict(onoff_error_rad=error, onoff_level=self.controller.level)
        delta = self.controller.update(e_y, e_psi, car.v, car.dt)
        return delta, dict(stanley_heading_rad=self.controller.heading_term,
                           stanley_cross_track_rad=self.controller.cross_track_term)

    def __call__(self, car):
        sim = car.sim
        matrix = np.asarray(sim.getObjectMatrix(car.robot, -1)).reshape(3, 4)
        front = matrix[:, :3] @ self.front_local + matrix[:, 3]
        forward = matrix[:, 2]
        heading = math.atan2(forward[1], forward[0])
        # Eixo X local do modelo Car aponta para cima: inclinacao em relacao a vertical.
        tilt = math.acos(float(np.clip(matrix[2, 0], -1, 1)))
        xy = front[:2]
        if self.previous_xy is not None:
            self.distance += float(np.hypot(*(xy - self.previous_xy)))
        else:
            self.start_xy = xy.copy()
        self.previous_xy = xy.copy()
        vref = self.args.vref
        extra = {}
        if self.path is not None:
            e_y, e_psi, s = self.path.errors(front, heading)
            valid, curvature, pixels = True, math.nan, 0
            gt = abs(e_y)
            extra['path_s'] = s
            if s >= self.path.length * (1. if self.path.closed else .98) and car.t > 2:
                self.completed = True
        else:
            result, pixels = self.perception(car, front, forward)
            valid = result is not None
            e_y, e_psi, curvature = result if valid else (math.nan,) * 3
            gt = self.track.error(xy)
            if valid and self.args.limitar_curvas:
                vref = curve_speed(vref, curvature, e_y, e_psi, min_speed=.5,
                                   lateral_accel=self.args.limitar_curvas)
        if valid:
            self.last_valid = car.t
            self.lost_since = None
            delta, info = self.steer(car, e_y, e_psi)
            extra.update(info)
        else:
            # Falha curta: mantem o ultimo esterco; depois para e centraliza.
            self.lost_since = car.t if self.lost_since is None else self.lost_since
            holding = self.last_valid is not None and car.t - self.last_valid <= .4
            delta = self.controller.steer if holding else 0.
            if not holding:
                vref = 0.
        car.set_steer(STEERING_SIGN * delta)
        car.set_vel(vref)
        car.log(x_front=xy[0], y_front=xy[1], heading_rad=heading, cross_track_m=e_y,
                heading_error_rad=e_psi, curvature_m_inv=curvature, line_valid=int(valid),
                line_pixels=pixels, delta_geo_rad=delta, gt_error_m=gt, tilt_rad=tilt,
                distance_m=self.distance, vref_cmd=vref, **extra)
        # Criterios de termino
        if tilt > TILT_LIMIT:
            self.failure = 'capotou'
        elif self.track is not None and math.isfinite(gt) and gt > OFF_TRACK_LIMIT:
            self.failure = 'saiu da pista'
        elif self.lost_since is not None and car.t - self.lost_since > LOST_LIMIT:
            self.failure = 'perdeu a linha'
        elif (self.track is not None and self.distance > 30.
              and np.hypot(*(xy - self.start_xy)) < 1.0):
            self.completed = True
        return self.failure is None and not self.completed


########################################
def summarize(car, mission, args):
    data = car.traj
    t = np.array([d['t'] for d in data])
    get = lambda k: np.array([d.get(k, np.nan) for d in data], dtype=float)
    settle = args.acomodacao
    mask = t >= settle
    gt = get('gt_error_m')[mask]
    gt = gt[np.isfinite(gt)]
    delta = get('delta_geo_rad')
    rate = np.diff(delta) / np.maximum(np.diff(t), 1e-9)
    rate = rate[np.isfinite(rate)]
    ey = get('cross_track_m')[mask]
    ey = ey[np.isfinite(ey)]
    summary = dict(
        referencia=args.referencia, controlador=args.controlador, vref=args.vref,
        limitar_curvas=args.limitar_curvas, concluido=bool(mission.completed),
        falha=mission.failure, tempo_s=float(t[-1]), distancia_m=mission.distance,
        velocidade_media_m_s=float(mission.distance / max(t[-1], 1e-9)),
        erro_rms_m=float(np.sqrt(np.mean(gt ** 2))) if len(gt) else None,
        erro_p95_m=float(np.percentile(gt, 95)) if len(gt) else None,
        erro_max_m=float(np.max(gt)) if len(gt) else None,
        erro_visual_rms_m=float(np.sqrt(np.mean(ey ** 2))) if len(ey) else None,
        taxa_estercamento_rms_deg_s=float(np.degrees(np.sqrt(np.mean(rate ** 2)))),
        fracao_saturado=float(np.nanmean(np.abs(delta) >= CAR['STEERMAX'] - 1e-3)),
        fracao_linha_valida=float(np.nanmean(get('line_valid'))),
        inclinacao_max_deg=float(np.degrees(np.nanmax(get('tilt_rad')))),
        acomodacao_s=settle,
    )
    if isinstance(mission.controller, OnOffSteering):
        summary.update(comutacoes=mission.controller.switches,
                       comutacoes_por_s=mission.controller.switches / max(t[-1], 1e-9),
                       delta_on_deg=args.delta_on, lookahead_m=args.lookahead)
    else:
        summary.update(ganho=args.ganho, suavizacao=args.suavizacao)
    return summary


def run(args):
    config = dict(MAIN_PARAMETERS, ts=args.ts, save=True, logfile=args.saida,
                  beep=False, real_time=False, port=args.port,
                  explicit_vision=True, velocity_reference=args.vref)
    car = Car(config)
    mission = LateralMission(args)
    summary = None
    try:
        mission.setup(car)
        car.start_mission()
        while car.t < args.ts:
            if not car.step():
                break
            if not mission(car):
                break
        if mission.path is not None and not mission.completed and mission.failure is None:
            mission.failure = 'tempo esgotado'
        summary = summarize(car, mission, args)
        car.save()
        Path(car.logfile, 'resumo.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        car.close()
        mission.restore(car)
    return car.logfile, summary


def parse(argv=None):
    p = argparse.ArgumentParser(description='Controle lateral on-off / Stanley no CoppeliaSim.')
    p.add_argument('--referencia', choices=('reta', 'circulo', 'oito', 'linha'), default='linha')
    p.add_argument('--controlador', choices=('onoff', 'stanley'), default='stanley')
    p.add_argument('--vref', type=float, default=1.0, help='referencia de velocidade [m/s]')
    p.add_argument('--ts', type=float, default=400., help='duracao maxima [s]')
    p.add_argument('--ganho', type=float, default=1.5, help='Stanley: k [1/s]')
    p.add_argument('--suavizacao', type=float, default=.3, help='Stanley: v_s [m/s]')
    p.add_argument('--delta-on', type=float, default=15., help='on-off: amplitude [graus]')
    p.add_argument('--h-on', type=float, default=3., help='on-off: liga acima de [graus]')
    p.add_argument('--h-off', type=float, default=1., help='on-off: desliga abaixo de [graus]')
    p.add_argument('--lookahead', type=float, default=.8, help='on-off: L_d [m]')
    p.add_argument('--taxa', type=float, default=60., help='taxa maxima do esterco [graus/s]')
    p.add_argument('--raio', type=float, default=3., help='raio do circulo/oito [m]')
    p.add_argument('--offset', type=float, default=.5, help='reta: deslocamento inicial [m]')
    p.add_argument('--pose', type=float, nargs=3, metavar=('X', 'Y', 'YAW'),
                   help='pose inicial para referencias geometricas [m m graus]')
    p.add_argument('--limitar-curvas', type=float, default=0.,
                   help='linha: aceleracao lateral maxima para reduzir v nas curvas (0 desliga)')
    p.add_argument('--acomodacao', type=float, default=5., help='metricas apos [s]')
    p.add_argument('--saida', default='logs/lateral/')
    p.add_argument('--port', type=int, default=23000)
    return p.parse_args(argv)


if __name__ == '__main__':
    run(parse())
