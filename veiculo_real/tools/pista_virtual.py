#!/usr/bin/env python3
"""Pista virtual de cones: testa estimador + seguidor em malha fechada, sem o carro.

Usa o MESMO codigo do carro (CorridorEstimator, RealStanleyFollower, PoseHistory,
follower_config do main.py) com um carro cinematico e uma camera simulada pela
homografia real (calibracao/camera_lateral.json):

    carro       bicicleta no eixo traseiro, direcao com media de 3 amostras e
                servo de 1a ordem, acelera ate 1 m/s^2 e NAO freia (rola a
                --coast m/s^2 quando a vref cai)
    odometria   encoder com erro de escala e esterco com vies (sem IMU), como o Car
    camera      ~15 quadros/s, latencia 0,12-0,18 s; cone visivel se a base cai
                dentro da imagem 640x480 e a ate 3 m; ruido de 1,5 px por cone e
                de arfagem (3 px, comum ao quadro); cones perdidos e falsos
    pista       retas e arcos; cones a cada --spacing m com erro de 3 cm; largura
                pode variar ao longo da pista

Exemplos:
    python3 tools/pista_virtual.py                         # todas as pistas, leis padrao
    python3 tools/pista_virtual.py --pista grampo --law stanley pure_pursuit
    python3 tools/pista_virtual.py --set speed_preview_m=0 --set velocity_reference=1.0
    python3 tools/pista_virtual.py --pista oval --plot oval.png

Metricas: tempo, velocidade media, erro lateral do eixo dianteiro ao centro
real (RMS e maximo), folga minima ate os cones (meia largura - |erro| - meia
largura do carro), quadros validos do estimador, paradas, aceleracao lateral
maxima e tempo acima de curve_lateral_accel + 0,2 m/s^2 (chegou rapido demais
na curva: o carro nao freia).
"""
import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fva_car.cones import CONE_RECOVERY_SPEED, ConeObservation  # noqa: E402
from fva_car.corridor import CorridorEstimator, CorridorSettings, STATES  # noqa: E402
from fva_car.lateral import GroundCalibration, LineObservation, RealStanleyFollower  # noqa: E402
from fva_car.odometry import PoseHistory  # noqa: E402

CALIBRATION = ROOT/'calibracao'/'camera_lateral.json'
WHEELBASE = .36
STEER_MAX = math.radians(20.)
CAR_HALF_WIDTH = .13
CONE_RADIUS = .05  # meia base do cone (10,5 cm)
# Carroceria: do eixo traseiro ate a frente do pneu dianteiro (7 cm adiante do eixo).
BODY_REAR, BODY_FRONT = -.05, WHEELBASE + .07


def body_clearance(x, y, th, cones):
    """(folga minima carroceria-cone em m, lado do cone: +1 esquerda / -1 direita).

    O carro e um segmento do eixo traseiro a frente, engrossado pela meia largura;
    numa curva fechada a traseira passa por DENTRO da dianteira (~L^2/2R), o que
    a folga so no eixo dianteiro nao via."""
    c, s = math.cos(th), math.sin(th)
    d = cones - (x, y)
    along = c*d[:, 0] + s*d[:, 1]
    side = -s*d[:, 0] + c*d[:, 1]
    near = (along > BODY_REAR - .5) & (along < BODY_FRONT + .5) & (np.abs(side) < 1.)
    if not np.any(near):
        return math.inf, 0
    along, side = along[near], side[near]
    gap_along = np.maximum(0., np.maximum(BODY_REAR - along, along - BODY_FRONT))
    gap_side = np.maximum(0., np.abs(side) - CAR_HALF_WIDTH)
    distance = np.hypot(gap_along, gap_side) - CONE_RADIUS
    k = int(np.argmin(distance))
    return float(distance[k]), (1 if side[k] > 0 else -1)

# (nome, segmentos): ('reta', comprimento) ou ('curva', raio com sinal (esquerda +), angulo em graus)
# Atraso puro da direcao do carro simulado (s), alem do filtro de 3 amostras
# (~0,02) e do servo (servo_tau 0,06): ~0,18 s do comando a guinada, como no
# carro de 26/11 (giroscopio). Antes 0: o simulador nao reproduzia o esterco
# "pedalando" do carro, e ajustes bons aqui oscilavam la.
STEER_DELAY_S = .10

TRACKS = {
    # Termina antes de cruzar o inicio (cones sobrepostos com largura variavel).
    'oval': [('reta', 3.), ('curva', 2.5, 180), ('reta', 5.), ('curva', 2.5, 180), ('reta', 1.)],
    'grampo': [('reta', 3.), ('curva', -1.8, 180), ('reta', 5.)],  # como o ensaio 20261023_080437
    's': [('reta', 3.), ('curva', 2.2, 90), ('curva', -2.2, 180), ('curva', 2.2, 90), ('reta', 4.)],
    'chicane': [('reta', 3.), ('curva', 3., 45), ('curva', -3., 90), ('curva', 3., 45), ('reta', 2.),
                ('curva', -2., 120), ('reta', 4.)],
    # Retas longas terminando em curvas fechadas: onde se ganha (ou se perde) tempo.
    'retas': [('reta', 10.), ('curva', -1.6, 180), ('reta', 10.), ('curva', 2., 90), ('reta', 6.)],
    # Pista dos ensaios de 10/11: curvas de raio ~1,2 m (kappa 0,8), largura ~1,3 m.
    'fechada': [('reta', 6.), ('curva', 1.25, 180), ('reta', 4.), ('curva', -1.25, 90), ('reta', 5.)],
}
# Rotatorias: chega reto, vira 60 graus a direita para entrar no anel, da a volta inteira
# na ilha (sentido anti-horario: ilha a esquerda) e sai virando a direita, como
# numa rotatoria de rua. Os cones que cairiam dentro da faixa de outro trecho
# (entrada e saida cruzando o anel) sao tirados, como as aberturas de uma
# rotatoria real (build_track clear_crossings). Fora de TRACKS para nao mudar as
# comparacoes antigas.
ROUNDABOUTS = {
    'rotatoria': [('reta', 4.), ('curva', -2.0, 60), ('curva', 2.0, 360), ('curva', -2.0, 60), ('reta', 4.)],
    'rotatoria_horaria': [('reta', 4.), ('curva', 2.0, 60), ('curva', -2.0, 360), ('curva', 2.0, 60), ('reta', 4.)],
    'rotatoria_fechada': [('reta', 4.), ('curva', -1.6, 60), ('curva', 1.5, 360), ('curva', -1.6, 60), ('reta', 4.)],
}


def ring_arc(segments):
    """(inicio, fim) em m de caminho do anel (o trecho de curva de 300 graus ou mais)."""
    start = 0.
    for segment in segments:
        length = segment[1] if segment[0] == 'reta' else abs(segment[1])*math.radians(segment[2])
        if segment[0] == 'curva' and abs(segment[2]) >= 300:
            return start, start + length
        start += length
    return None


ALL_TRACKS = {**TRACKS, **ROUNDABOUTS}


@dataclass
class Track:
    center: np.ndarray   # Nx2, a cada 2 cm
    heading: np.ndarray
    s: np.ndarray
    width: np.ndarray
    left: np.ndarray     # cones (Mx2)
    right: np.ndarray
    # Objetos fixos fora da pista que o detector toma por cone (mochila, roupa):
    # (x, y) e a cor (0: entra com os da esquerda, 1: com os da direita).
    decoys: np.ndarray = None
    decoy_color: np.ndarray = None

    @property
    def length(self):
        return float(self.s[-1])


def build_track(segments, width=1.8, width_end=None, spacing=.5, cone_noise=.03, rng=None, swap=None,
                clear_crossings=None):
    """Pista pelos trechos; Track.left/right sao os cones VERDES/VERMELHOS.

    swap=(s0, s1): nesse trecho (m de caminho) as cores trocam de lado (borda
    esquerda vermelha), como uma rotatoria com a ilha de outra cor.
    clear_crossings: tira cones que caem dentro da faixa de outro trecho da
    pista (padrao: so nas rotatorias, onde a pista cruza a si mesma)."""
    rng = rng or np.random.default_rng(0)
    ds = .02
    x = y = th = 0.
    points, headings = [(x, y)], [th]
    for segment in segments:
        if segment[0] == 'reta':
            n, curvature = int(round(segment[1]/ds)), 0.
        else:
            radius, angle = segment[1], math.radians(segment[2])
            n, curvature = int(round(abs(radius)*angle/ds)), 1./radius
        for _ in range(n):
            th_mid = th + curvature*ds/2
            x, y = x + ds*math.cos(th_mid), y + ds*math.sin(th_mid)
            th += curvature*ds
            points.append((x, y))
            headings.append(th)
    center, heading = np.asarray(points), np.asarray(headings)
    s = np.arange(len(center))*ds
    widths = np.full(len(s), width) if width_end is None else np.interp(s, [0., s[-1]], [width, width_end])
    normal = np.c_[-np.sin(heading), np.cos(heading)]
    index = np.searchsorted(s, np.arange(0., s[-1], spacing))
    cones = {}
    for side, sign in (('left', 1.), ('right', -1.)):
        base = center[index] + sign*widths[index, None]/2*normal[index]
        cones[side] = base + rng.normal(0., cone_noise, base.shape)
    keep = np.ones((2, len(index)), bool)
    if clear_crossings if clear_crossings is not None else any(abs(seg[2]) > 300 for seg in segments
                                                                if seg[0] == 'curva'):
        for k, side in enumerate(('left', 'right')):
            for j, cone in enumerate(cones[side]):
                d = np.hypot(*(center - cone).T)
                other = np.abs(s - s[index[j]]) > 1.5*width     # outro trecho, nao o vizinho
                keep[k, j] = not np.any(other & (d < widths/2 - .1))
    green, red = cones['left'][keep[0]], cones['right'][keep[1]]
    if swap is not None:
        arcs = s[index]
        swapped = (arcs >= swap[0]) & (arcs <= swap[1])
        green = np.vstack([cones['left'][keep[0] & ~swapped], cones['right'][keep[1] & swapped]])
        red = np.vstack([cones['right'][keep[1] & ~swapped], cones['left'][keep[0] & swapped]])
    return Track(center, heading, s, widths, green, red)


def add_decoys(track, count, rng, gap=(.2, .8)):
    """`count` objetos fixos a gap m FORA da borda, lado e cor ao acaso."""
    if count <= 0:
        return track
    index = np.searchsorted(track.s, rng.uniform(2., max(2.1, track.length - 1.), count))
    index = np.clip(index, 0, len(track.s) - 1)
    sign = rng.choice([-1., 1.], count)
    normal = np.c_[-np.sin(track.heading[index]), np.cos(track.heading[index])]
    offset = sign*(track.width[index]/2 + rng.uniform(*gap, count))
    track.decoys = track.center[index] + offset[:, None]*normal
    track.decoy_color = rng.integers(0, 2, count)
    return track


class SimCar:
    """Mesma interface que o RealStanleyFollower usa do Car real."""
    def __init__(self, pose, rng, coast=.5, scale_error=.02, steer_bias_deg=.5, servo_tau=.06, fixed_bias_deg=0.,
                 steer_response=1., model_response=1., gyro=False, steer_delay=0.):
        """steer_response: fracao da guinada do modelo que o carro faz de verdade
        (0,88-0,90 no carro de 19/11); model_response: a que a odometria supoe
        (steering_response do main.py). gyro: odometria pela guinada medida
        (IMU), com ruido e vies pequenos, em vez do modelo."""
        self.servo_tau = servo_tau
        # Atraso puro da direcao (s), alem do servo: no carro de 26/11 o comando
        # so aparecia na guinada do giroscopio ~0,18 s depois (ajuste atraso
        # puro + 1a ordem em 13 s de oscilacao), contra ~0,08 s deste modelo.
        self.steer_queue = deque([0.]*max(0, int(round(steer_delay/.02))))
        self.steer_response, self.model_response, self.gyro = steer_response, model_response, gyro
        self.gyro_bias = rng.normal(0., .005) if gyro else 0.
        self.w_imu = math.nan
        self.trim = 0.
        self.X, self.Y, self.th = pose          # verdade, eixo traseiro
        self.odo = list(pose)                   # odometria do Car (encoder + esterco)
        self.v_true = 0.
        self.v = 0.
        self.w = 0.
        self.dt = .02
        self.t = 0.
        self.vref = 0.
        self.u = 0.
        self.st = 0.
        self.st_filtered = 0.
        self.steer_actual = 0.
        self.history = deque([0.]*3, maxlen=3)
        self.emergencia = False
        self.encoder_valid = True
        self.sel_tracao = self.sel_direcao = 1
        self.gear = None
        self.vel_error_integral = 0.
        self.rng = rng
        self.coast = coast
        self.scale = 1. + rng.normal(0., scale_error)
        self.steer_bias = math.radians(rng.normal(0., steer_bias_deg) + fixed_bias_deg)
        self.last_log = {}

    def _set_ref(self, vref):
        self.vref = float(np.clip(vref, -2., 2.))

    def set_vel(self, vref):
        self._set_ref(vref)

    def set_u(self, u):
        self.u = u

    def set_steer_trim(self, trim):
        self.trim = float(trim)  # centro do servo: muda a roda, nao a odometria
        return self.trim

    def set_steer(self, steer):
        self.st = float(np.clip(steer, -STEER_MAX, STEER_MAX))
        self.history.append(self.st)
        self.st_filtered = float(np.mean(self.history))

    def log(self, **values):
        self.last_log.update(values)

    def advance(self):
        dt = self.dt
        # Direcao: media de 3 amostras (steering_filter_size) + servo ~60 ms; vies mecanico.
        commanded = self.st_filtered
        if self.steer_queue:
            self.steer_queue.append(commanded)
            commanded = self.steer_queue.popleft()
        self.steer_actual += (commanded - self.steer_actual)*(1. - math.exp(-dt/self.servo_tau))
        steer = self.steer_actual + self.steer_bias + self.trim
        # Velocidade: PI rapido para subir; para descer o carro so rola.
        error = self.vref - self.v_true
        if error > 0:
            self.v_true += min(1., error/.25)*dt
        else:
            self.v_true -= min(self.coast, -error/.25)*dt
        self.v_true = max(self.v_true, 0.)
        self.X += self.v_true*math.cos(self.th)*dt
        self.Y += self.v_true*math.sin(self.th)*dt
        true_rate = self.v_true/WHEELBASE*math.tan(steer)*self.steer_response
        self.th += true_rate*dt
        self.t += dt
        # Sensores do Car: encoder com erro de escala; w pelo giroscopio (IMU) ou
        # pelo modelo com o comando.
        self.v = max(0., self.v_true*self.scale + self.rng.normal(0., .015))
        if self.gyro:
            self.w_imu = true_rate + self.gyro_bias + self.rng.normal(0., .005)
            self.w = self.w_imu
        else:
            self.w = self.v/WHEELBASE*math.tan(self.st)*self.model_response
        self.odo[0] += self.v*math.cos(self.odo[2])*dt
        self.odo[1] += self.v*math.sin(self.odo[2])*dt
        self.odo[2] += self.w*dt

    @property
    def front(self):
        return np.array([self.X + WHEELBASE*math.cos(self.th), self.Y + WHEELBASE*math.sin(self.th)])


class SimCamera:
    def __init__(self, calibration, rng, pixel_sigma=1.5, pitch_sigma=3., miss=.08, far_miss=.25,
                 false_rate=.05, max_range=3., row_bias=0., column_bias=0.):
        """row_bias / column_bias: erro FIXO da calibracao, px (camera real deslocada
        em relacao a homografia: + linha = cones parecem mais perto; + coluna = a direita)."""
        self.bias = np.array([column_bias, row_bias], float)
        self.H = np.asarray(calibration.homography, float)
        self.Hinv = np.linalg.inv(self.H)
        self.size = calibration.image_size
        self.rng = rng
        self.pixel_sigma, self.pitch_sigma = pixel_sigma, pitch_sigma
        self.miss, self.far_miss, self.false_rate = miss, far_miss, false_rate
        self.max_range = max_range

    def _to_pixels(self, ground):
        q = np.c_[ground, np.ones(len(ground))] @ self.Hinv.T
        return q[:, :2]/q[:, 2:3], q[:, 2]

    def _to_ground(self, pixels):
        q = np.c_[pixels, np.ones(len(pixels))] @ self.H.T
        return q[:, :2]/q[:, 2:3]

    def observe_line(self, car, track, spacing=.15, line_sigma=.01, miss=.05):
        """Pontos da linha amarela (linha central da pista) visiveis na imagem,
        espacados como o detector entrega ao estimador (line_samples)."""
        if self.rng.random() < miss:
            return np.empty((0, 2))
        c, s = math.cos(car.th), math.sin(car.th)
        d = track.center[::max(1, int(round(spacing/.02)))] - car.front
        local = np.c_[c*d[:, 0] + s*d[:, 1], -s*d[:, 0] + c*d[:, 1]]
        local = local[(local[:, 0] > .3) & (local[:, 0] < 3.)]
        if not len(local):
            return local
        pixels, _ = self._to_pixels(local)
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < self.size[0]) & (pixels[:, 1] >= 0)
                  & (pixels[:, 1] < self.size[1]))
        local = local[inside]
        return local + self.rng.normal(0., line_sigma, local.shape) if len(local) else local

    def observe(self, car, track):
        """Cones (esquerda, direita) no referencial do eixo dianteiro, como o detector entrega."""
        c, s = math.cos(car.th), math.sin(car.th)
        front = car.front
        pitch = self.rng.normal(0., self.pitch_sigma)
        output = []
        sources = [track.left, track.right]
        if track.decoys is not None:
            sources = [np.vstack([cones, track.decoys[track.decoy_color == k]]) for k, cones in enumerate(sources)]
        for cones in sources:
            d = cones - front
            local = np.c_[c*d[:, 0] + s*d[:, 1], -s*d[:, 0] + c*d[:, 1]]
            local = local[(local[:, 0] > .2) & (local[:, 0] < self.max_range + .3)]
            if not len(local):
                output.append(np.empty((0, 2)))
                continue
            pixels, w = self._to_pixels(local)
            inside = ((w*np.sign(self.H[2, 2] if self.H[2, 2] else 1.) != 0) & (pixels[:, 0] >= 0)
                      & (pixels[:, 0] < self.size[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.size[1]))
            pixels, local = pixels[inside], local[inside]
            missed = self.rng.random(len(local)) < np.where(local[:, 0] > 2.5, self.far_miss, self.miss)
            pixels = pixels[~missed]
            pixels = pixels + self.bias + self.rng.normal(0., self.pixel_sigma, pixels.shape)
            pixels[:, 1] += pitch
            ground = self._to_ground(pixels) if len(pixels) else np.empty((0, 2))
            ground = ground[(ground[:, 0] > 0) & (ground[:, 0] <= self.max_range)]
            output.append(ground)
        if self.rng.random() < self.false_rate:
            # Falso positivo (reflexo, porta) em um dos lados.
            pixel = np.array([[self.rng.uniform(0, self.size[0]), self.rng.uniform(.62, 1.)*self.size[1]]])
            point = self._to_ground(pixel)
            if 0 < point[0, 0] <= self.max_range:
                side = int(self.rng.random() < .5)
                output[side] = np.vstack([output[side], point])
        return output


def track_error(track, point, hint):
    """(erro lateral com sinal, esquerda +; indice; meia largura) do ponto ao centro real."""
    low, high = max(0, hint - 150), min(len(track.s), hint + 400)
    d = track.center[low:high] - point
    index = low + int(np.argmin(np.einsum('ij,ij->i', d, d)))
    normal = np.array([-math.sin(track.heading[index]), math.cos(track.heading[index])])
    return float(np.dot(point - track.center[index], normal)), index, float(track.width[index]/2)


def run(track, config, corridor=None, seed=0, fps=15., latency=(.12, .18), duration=None, coast=.5,
        camera_kwargs=None, trace=False, servo_tau=.06, steer_bias_deg=0., line=False, decoys=0,
        steer_response=None, color_sides=False, steer_delay=None):
    """Uma volta; devolve dicionario de metricas (e a serie, com trace).

    steer_response: None (padrao) = carro ideal (roda faz a curva pedida, odometria
    pelo modelo puro); um numero = a roda faz essa fracao e a odometria usa o
    steering_response da configuracao. config['sim_gyro']: odometria pelo giroscopio.
    color_sides: como no carro, o lado de cada cor e decidido pelos proprios
    cones (SideTracker de fva_car/cones.py), com o estimador recomecando quando
    troca; sem isso, verde = esquerda sempre. steer_delay: atraso puro da
    direcao (s); None = STEER_DELAY_S (o do carro)."""
    import main as vehicle_main

    if steer_delay is None:
        steer_delay = STEER_DELAY_S

    rng = np.random.default_rng(seed)
    add_decoys(track, decoys, np.random.default_rng(1000 + seed))
    calibration = GroundCalibration.from_file(CALIBRATION)
    camera = SimCamera(calibration, rng, **(camera_kwargs or {}))
    if corridor is None:  # como no main.py: a linha amarela tem o estimador proprio
        corridor = (vehicle_main.line_estimator_settings(config) if line
                    else vehicle_main.corridor_settings(config))
    settings = CorridorSettings.from_value(
        corridor, track_width_m=config['cone_settings'].get('track_width_cm', 90)/100, max_range_m=3.)
    estimator = CorridorEstimator(settings, calibration.homography)
    map_path = None
    if config.get('cone_map', False) and not line:
        from fva_car.map_path import MapPath
        map_path = MapPath(wheelbase=WHEELBASE, default_width=settings.track_width_m,
                           replace=config.get('cone_map') == 'substituir')
    sides = None
    if color_sides:
        from fva_car.cones import SideTracker, handle_side_flip
        sides = SideTracker()
    flips = 0
    follower = RealStanleyFollower(vehicle_main.follower_config(config, 'cones', corridor))
    run.last_follower = follower  # trim aprendido (corridas seguidas)
    poses = PoseHistory()
    if config.get('exact_compensation', True):
        follower.odometry = poses  # como no carro (desde 26/11): compensacao exata do atraso
    start = -WHEELBASE  # eixo dianteiro no inicio da pista
    real = steer_response is not None
    car = SimCar((start*math.cos(track.heading[0]), start*math.sin(track.heading[0]), track.heading[0]),
                 rng, coast=coast, servo_tau=servo_tau, fixed_bias_deg=steer_bias_deg,
                 steer_response=float(steer_response) if real else 1.,
                 model_response=float(config.get('steering_response', 1.)) if real else 1.,
                 gyro=bool(config.get('sim_gyro', False)), steer_delay=steer_delay)
    duration = duration or (track.length/.25 + 5.)
    pending, observation = [], None
    next_capture, last_capture = 0., None
    hint = 0
    rows = []
    body = []  # (folga da carroceria, lado do cone, sinal da curva)
    all_cones = np.vstack([track.left, track.right])
    states = {name: 0 for name in STATES}
    frames = valid = 0
    finished = False
    max_index = 0
    while car.t < duration:
        car.advance()
        now = car.t
        poses.append(now, *car.odo)
        if now >= next_capture:
            motion = poses.delta(last_capture, now) if last_capture is not None else None
            if line:  # linha amarela no centro, vista de cima -> estimador como linha central
                estimate = estimator.step(now, [], [], motion, pose_frame=True,
                                          center=camera.observe_line(car, track))
            else:
                green, red = camera.observe(car, track)
                green_left = True
                if sides is not None:  # como RealConeDetector._corridor_sides
                    green_left, info, flipped = sides.update(green, red, estimator, True)
                    if flipped:
                        flips += 1
                        handle_side_flip(estimator, info, map_path)
                left, right = (green, red) if green_left else (red, green)
                estimate = estimator.step(now, left, right, motion, pose_frame=True)
                if map_path is not None:
                    estimate = map_path.step(now, green, red, green_left, motion, True, estimate,
                                             estimator._point_covariance)
            last_capture = now
            frames += 1
            states[estimate.state] += 1
            valid += estimate.valid
            ready = now + rng.uniform(*latency)
            one_border = estimate.valid and estimate.state in ('SO_EXTERNA', 'SO_INTERNA', 'UMA_BORDA_RETA')
            fields = dict(speed_factor=estimate.speed_factor if estimate.valid else 1.,
                          one_border_curve=estimate.valid and estimate.state in ('SO_EXTERNA', 'SO_INTERNA'),
                          path_dkappa=float(estimate.state_vector[3]) if estimate.valid else math.nan,
                          speed_limit=(CONE_RECOVERY_SPEED if estimate.valid and estimate.slow
                                       else settings.one_border_speed if one_border else math.inf))
            pending.append((ready, ConeObservation(now, errors=estimate.errors, reason=estimate.reason,
                                                   corridor_width_m=estimate.width_m if estimate.valid else math.nan,
                                                   **fields)))
            next_capture = now + rng.normal(1./fps, .006)
        while pending and pending[0][0] <= now:
            observation = pending.pop(0)[1]
        follower.apply(car, observation, now)
        error, hint, half = track_error(track, car.front, hint)
        max_index = max(max_index, hint)
        lateral_accel = car.v_true**2*math.tan(car.steer_actual + car.steer_bias + car.trim)/WHEELBASE
        rows.append((now, car.X, car.Y, car.front[0], car.front[1], error, half, car.v_true, car.vref,
                     car.st, follower.last_state.get('active', False), lateral_accel))
        gap, cone_side = body_clearance(car.X, car.Y, car.th, all_cones)
        body.append((gap, cone_side, math.copysign(1., car.st) if abs(car.st) > .05 else 0.))
        if track.s[max_index] >= track.length - 3.2:
            finished = True
            break
        if abs(error) > half + .5:
            break  # saiu da pista
    data = np.asarray(rows, float)
    accel_limit = float(config.get('curve_lateral_accel', .4)) + .2
    moving = data[:, 7] > .05
    errors = data[moving, 5] if np.any(moving) else data[:, 5]
    clearance = data[:, 6] - np.abs(data[:, 5]) - CAR_HALF_WIDTH
    stopped = np.diff(np.r_[0, (data[:, 7] < .03) & (data[:, 0] > 2.), 0])
    result = dict(
        finished=finished, time_s=float(data[-1, 0]), distance_m=float(track.s[max_index]),
        mean_speed=float(track.s[max_index]/max(data[-1, 0], 1e-6)),
        rms_m=float(np.sqrt(np.mean(errors**2))), max_m=float(np.max(np.abs(errors))),
        mean_m=float(np.mean(errors)),
        min_clearance_m=float(np.min(clearance)), hits=int(np.sum(np.diff(np.r_[0, clearance < 0, 0]) > 0)),
        stops=int(np.sum(stopped > 0)), valid=valid/max(frames, 1),
        max_lateral_accel=float(np.max(np.abs(data[:, 11]))),
        over_accel_s=float(np.sum(np.abs(data[:, 11]) > accel_limit)*car.dt),
        states={k: v/max(frames, 1) for k, v in states.items() if v}, side_flips=flips,
        steer_reversals=steer_reversals(data[:, 0], data[:, 9]),
        **_body_metrics(np.asarray(body, float)),
        width_learned=estimator.learned_width)
    if trace:
        result['trace'] = data
    return result


def steer_reversals(t, steer, threshold_deg=8., window_s=.5):
    """'Pedaladas': esterco de mais de threshold de um lado ao outro em ate window_s."""
    steer = np.degrees(np.asarray(steer, float))
    count, i = 0, 0
    while i < len(steer):
        if abs(steer[i]) > threshold_deg:
            j = i + 1
            while j < len(steer) and t[j] - t[i] < window_s:
                if steer[j]*steer[i] < 0 and abs(steer[j]) > threshold_deg:
                    count += 1
                    i = j - 1
                    break
                j += 1
        i += 1
    return count


def _body_metrics(body):
    """Folga minima da carroceria aos cones e toques (do lado de dentro da curva a parte)."""
    if not len(body):
        return dict(body_clearance_m=math.nan, body_hits=0, inside_hits=0, inside_clearance_m=math.nan)
    gap, side, turn = body.T
    touching = gap < 0
    starts = np.diff(np.r_[0, touching.astype(int), 0]) > 0
    inside = (turn != 0) & (side == turn)
    inside_gap = gap[inside]
    return dict(body_clearance_m=float(np.min(gap)), body_hits=int(np.sum(starts)),
                inside_hits=int(np.sum(starts[:len(gap)] & inside)),
                inside_clearance_m=float(np.min(inside_gap)) if len(inside_gap) else math.nan)


def parse_value(text):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return {'true': True, 'false': False}.get(text.lower(), text)


def base_config(changes=None):
    import main as vehicle_main

    config = dict(vehicle_main.parameters)
    # O trim inicial do main.py e do carro REAL (vies da direcao dele); o carro
    # simulado tem o proprio vies (SimCar) e comeca sem trim.
    config['steering_trim_deg'] = 0.0
    config.update(changes or {})
    return config


def plot(track, results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(track.left[:, 0], track.left[:, 1], 'g^', ms=4)
    axis.plot(track.right[:, 0], track.right[:, 1], 'r^', ms=4)
    axis.plot(track.center[:, 0], track.center[:, 1], 'k--', lw=.6)
    for label, result in results.items():
        data = result['trace']
        axis.plot(data[:, 3], data[:, 4], lw=1.4, label=f"{label}: rms {100*result['rms_m']:.0f} cm")
    axis.set_aspect('equal')
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pista', nargs='+', default=list(TRACKS), choices=list(ALL_TRACKS))
    parser.add_argument('--linha', action='store_true',
                        help='segue a linha amarela no centro (estimador como linha central) em vez dos cones')
    parser.add_argument('--law', nargs='+', default=['stanley', 'pure_pursuit'], choices=['stanley', 'pure_pursuit'])
    parser.add_argument('--width', type=float, default=1.8, help='largura real da pista, m')
    parser.add_argument('--width-end', type=float, help='largura no fim da pista (varia linearmente)')
    parser.add_argument('--spacing', type=float, default=.5, help='espaco entre cones, m')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--coast', type=float, default=.5, help='desaceleracao rolando, m/s^2')
    parser.add_argument('--latency', type=float, nargs=2, default=(.12, .18), help='atraso da visao, s (min max)')
    parser.add_argument('--fps', type=float, default=15.)
    parser.add_argument('--servo-tau', type=float, default=.06, help='constante de tempo do servo, s')
    parser.add_argument('--steer-bias', type=float, default=0., help='vies fixo da direcao (graus, + esquerda)')
    parser.add_argument('--row-bias', type=float, default=0., help='erro fixo da calibracao na linha, px')
    parser.add_argument('--column-bias', type=float, default=0., help='erro fixo da calibracao na coluna, px')
    parser.add_argument('--objetos', type=int, default=0,
                        help='objetos fixos tomados por cone, 20-80 cm fora da pista (mochila, roupa)')
    parser.add_argument('--falsos', type=float, default=.05, help='fracao de quadros com um falso ao acaso')
    parser.add_argument('--set', action='append', default=[], metavar='AJUSTE=VALOR',
                        help='sobrepoe main.parameters (ex.: velocity_reference=1.0)')
    parser.add_argument('--plot', type=Path, help='PNG com as trajetorias (1a semente)')
    args = parser.parse_args(argv)
    changes = {}
    for item in args.set:
        key, _, value = item.partition('=')
        changes[key.strip()] = parse_value(value.strip())
    print(f"{'pista':8s} {'lei':13s} {'ok':>3s} {'tempo':>6s} {'v med':>6s} {'rms cm':>7s} {'max cm':>7s}"
          f" {'folga cm':>8s} {'toques':>6s} {'paradas':>7s} {'valid':>6s} {'W apr':>6s} {'alat max':>8s} {'s acima':>7s}"
          f" {'media cm':>8s}")
    for name in args.pista:
        traced = {}
        for law in args.law:
            config = base_config({**changes, 'steering_law': law})
            results = []
            for seed in range(args.seeds):
                track = build_track(ALL_TRACKS[name], args.width, args.width_end, args.spacing,
                                    rng=np.random.default_rng(100 + seed))
                result = run(track, config, seed=seed, coast=args.coast, trace=bool(args.plot) and seed == 0,
                             latency=tuple(args.latency), fps=args.fps, servo_tau=args.servo_tau,
                             steer_bias_deg=args.steer_bias, line=args.linha, decoys=args.objetos,
                             camera_kwargs=dict(row_bias=args.row_bias, column_bias=args.column_bias,
                                                false_rate=args.falsos))
                results.append(result)
                if 'trace' in result:
                    traced[law] = result
            mean = lambda key: float(np.mean([r[key] for r in results]))
            worst = lambda key, f=max: float(f(r[key] for r in results))
            learned = [r['width_learned'] for r in results if r['width_learned']]
            print(f"{name:8s} {law:13s} {sum(r['finished'] for r in results):>2d}/{len(results)}"
                  f" {mean('time_s'):6.1f} {mean('mean_speed'):6.2f} {100*mean('rms_m'):7.1f}"
                  f" {100*worst('max_m'):7.1f} {100*worst('min_clearance_m', min):8.1f}"
                  f" {sum(r['hits'] for r in results):6d} {sum(r['stops'] for r in results):7d}"
                  f" {mean('valid'):6.2f} {np.mean(learned) if learned else float('nan'):6.2f}"
                  f" {worst('max_lateral_accel'):8.2f} {mean('over_accel_s'):7.2f} {100*mean('mean_m'):+8.1f}", flush=True)
        if args.plot and traced:
            path = args.plot if len(args.pista) == 1 else args.plot.with_name(f'{args.plot.stem}_{name}{args.plot.suffix}')
            plot(build_track(ALL_TRACKS[name], args.width, args.width_end, args.spacing, rng=np.random.default_rng(100)),
                 traced, path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
