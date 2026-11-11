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
from fva_car.cones import CONE_RECOVERY_SPEED  # noqa: E402
from fva_car.corridor import CorridorEstimator, CorridorSettings, STATES  # noqa: E402
from fva_car.lateral import GroundCalibration, LineObservation, RealStanleyFollower  # noqa: E402
from fva_car.odometry import PoseHistory  # noqa: E402

CALIBRATION = ROOT/'calibracao'/'camera_lateral.json'
WHEELBASE = .36
STEER_MAX = math.radians(20.)
CAR_HALF_WIDTH = .13

# (nome, segmentos): ('reta', comprimento) ou ('curva', raio com sinal (esquerda +), angulo em graus)
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


@dataclass
class Track:
    center: np.ndarray   # Nx2, a cada 2 cm
    heading: np.ndarray
    s: np.ndarray
    width: np.ndarray
    left: np.ndarray     # cones (Mx2)
    right: np.ndarray

    @property
    def length(self):
        return float(self.s[-1])


def build_track(segments, width=1.8, width_end=None, spacing=.5, cone_noise=.03, rng=None):
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
    return Track(center, heading, s, widths, cones['left'], cones['right'])


class SimCar:
    """Mesma interface que o RealStanleyFollower usa do Car real."""
    def __init__(self, pose, rng, coast=.5, scale_error=.02, steer_bias_deg=.5, servo_tau=.06, fixed_bias_deg=0.):
        self.servo_tau = servo_tau
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
        self.steer_actual += (self.st_filtered - self.steer_actual)*(1. - math.exp(-dt/self.servo_tau))
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
        self.th += self.v_true/WHEELBASE*math.tan(steer)*dt
        self.t += dt
        # Sensores do Car: encoder com erro de escala, w pelo modelo com o comando.
        self.v = max(0., self.v_true*self.scale + self.rng.normal(0., .015))
        self.w = self.v/WHEELBASE*math.tan(self.st)
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
        for cones in (track.left, track.right):
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
        camera_kwargs=None, trace=False, servo_tau=.06, steer_bias_deg=0., line=False):
    """Uma volta; devolve dicionario de metricas (e a serie, com trace)."""
    import main as vehicle_main

    rng = np.random.default_rng(seed)
    calibration = GroundCalibration.from_file(CALIBRATION)
    camera = SimCamera(calibration, rng, **(camera_kwargs or {}))
    corridor = corridor if corridor is not None else vehicle_main.corridor_settings(config)
    settings = CorridorSettings.from_value(
        corridor, track_width_m=config['cone_settings'].get('track_width_cm', 90)/100, max_range_m=3.)
    estimator = CorridorEstimator(settings, calibration.homography)
    follower = RealStanleyFollower(vehicle_main.follower_config(config, 'cones', corridor))
    run.last_follower = follower  # trim aprendido (corridas seguidas)
    poses = PoseHistory()
    start = -WHEELBASE  # eixo dianteiro no inicio da pista
    car = SimCar((start*math.cos(track.heading[0]), start*math.sin(track.heading[0]), track.heading[0]),
                 rng, coast=coast, servo_tau=servo_tau, fixed_bias_deg=steer_bias_deg)
    duration = duration or (track.length/.25 + 5.)
    pending, observation = [], None
    next_capture, last_capture = 0., None
    hint = 0
    rows = []
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
                left, right = camera.observe(car, track)
                estimate = estimator.step(now, left, right, motion, pose_frame=True)
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
            pending.append((ready, LineObservation(now, errors=estimate.errors, reason=estimate.reason,
                                                   reference_source='cones', **fields)))
            next_capture = now + rng.normal(1./fps, .006)
        while pending and pending[0][0] <= now:
            observation = pending.pop(0)[1]
        follower.apply(car, observation, now)
        error, hint, half = track_error(track, car.front, hint)
        max_index = max(max_index, hint)
        lateral_accel = car.v_true**2*math.tan(car.steer_actual + car.steer_bias + car.trim)/WHEELBASE
        rows.append((now, car.X, car.Y, car.front[0], car.front[1], error, half, car.v_true, car.vref,
                     car.st, follower.last_state.get('active', False), lateral_accel))
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
        states={k: v/max(frames, 1) for k, v in states.items() if v},
        width_learned=estimator.learned_width)
    if trace:
        result['trace'] = data
    return result


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
    parser.add_argument('--pista', nargs='+', default=list(TRACKS), choices=list(TRACKS))
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
                track = build_track(TRACKS[name], args.width, args.width_end, args.spacing,
                                    rng=np.random.default_rng(100 + seed))
                result = run(track, config, seed=seed, coast=args.coast, trace=bool(args.plot) and seed == 0,
                             latency=tuple(args.latency), fps=args.fps, servo_tau=args.servo_tau,
                             steer_bias_deg=args.steer_bias, line=args.linha,
                             camera_kwargs=dict(row_bias=args.row_bias, column_bias=args.column_bias))
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
            plot(build_track(TRACKS[name], args.width, args.width_end, args.spacing, rng=np.random.default_rng(100)),
                 traced, path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
