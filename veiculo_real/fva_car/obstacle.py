"""Parada pelo ultrassom em curva: ignora o eco que e cone da borda fora do caminho.

Em curva, com a direcao ja esterçada, o ultrassom (que aponta para frente) ve o
cone da borda de fora muito perto e travava o carro, embora ele fosse passar ao
lado. A visao sabe onde estao as bordas (estimador do corredor): se o eco esta
exatamente onde ha uma borda, dentro do feixe e na distancia medida, e o carro
esta virando para o lado da curva e dentro do corredor com folga, o eco e o cone
da borda e a parada e ignorada. Obstaculo no meio da faixa nao coincide com
borda nenhuma: para como antes. Sem corredor valido: para como antes.

Na RETA tambem (24/11): a ~1 m/s a distancia de parada passa de 1 m, e a 1 m os
cones das bordas de uma pista de 1,3 m ficam a ~33 graus do eixo, ainda no feixe
largo do HC-SR04 (24/11: paradas a 0,72-0,85 m a ~1 m/s em reta). Na reta
a borda continua (modelo) "explicaria" qualquer eco de 0,9 a 2,7 m, inclusive um
obstaculo no meio da faixa: so vale um CONE REAL (detectado no quadro ou no mapa
SLAM) na distancia do eco e ao lado do caminho (o carro passa sem encostar). Em
24/11 um dos ecos era um cone a 13 graus e 20 cm do eixo: o carro bateria nele, e
a parada continua.
"""
import math

import numpy as np

from .corridor import move_states, offsets, sample_centerline

# HC-SR04: feixe nominal de +-15 graus, mas cone perto e de lado ecoa ate ~+-35.
BEAM_HALF_DEG = 35.
ECHO_TOLERANCE_M = .25    # eco e borda a mesma distancia, com esta folga (borda junto ao carro e extrapolada)
MIN_STEER_DEG = 6.        # roda virada para o lado CONTRARIO da curva alem disto: nao confia
MIN_CURVATURE = .15       # 1/m (R <= 6,7 m) em algum ponto do proximo 1,5 m de caminho
CURVE_LOOKAHEAD_M = 1.5
CAR_HALF_WIDTH_M = .13
SIDE_MARGIN_M = .05       # folga lateral minima ate a borda, alem da meia largura do carro
# Na reta, so um CONE REAL (visto ou no mapa) na distancia do eco e que o carro
# PASSA AO LADO: |y| >= meia largura do carro + meia base do cone + folga.
STRAIGHT_MIN_LATERAL_M = CAR_HALF_WIDTH_M + .05 + .07
STRAIGHT_MAX_BEARING_DEG = 45.  # feixe largo do HC-SR04 com cone perto
STRAIGHT_RANGE_TOLERANCE_M = .25
SENSOR_X_M = 0.           # ultrassom (embaixo da camera) na linha do eixo dianteiro (medido em 19/11)
MAX_AGE_S = .5            # estado da visao mais velho que isto nao serve
ONE_OR_TWO_BORDERS = ('AMBAS', 'SO_EXTERNA', 'SO_INTERNA', 'UMA_BORDA_RETA', 'MEMORIA')
RESUME_ECHO_TOLERANCE_M = .12


def border_echo(distance, state_vector, sensor_x=SENSOR_X_M, beam_half_deg=BEAM_HALF_DEG,
                tolerance=ECHO_TOLERANCE_M, min_bearing_deg=0.):
    """'esquerda'/'direita' se ha ponto dessa borda no feixe (entre min_bearing_deg e
    beam_half_deg do eixo), a `distance` m do sensor; senao None."""
    width = float(state_vector[4])
    arcs = np.arange(-.6, 3.5, .02)
    for side, offset in (('esquerda', width/2), ('direita', -width/2)):
        points = sample_centerline(state_vector, arcs, offset) - [sensor_x, 0.]
        ranges = np.hypot(points[:, 0], points[:, 1])
        bearing = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        hit = ((points[:, 0] > 0) & (np.abs(ranges - distance) <= tolerance)
               & (np.abs(bearing) <= beam_half_deg) & (np.abs(bearing) >= min_bearing_deg))
        if np.any(hit):
            return side
    return None


def _visible_cones(observation):
    debug = getattr(observation, 'debug', None) or {}
    detection = debug.get('detection') or {}
    points = [p for color in ('green', 'red') for p in ((detection.get(color) or {}).get('m') or [])]
    return points + list(debug.get('map_cones') or [])


def real_cone_echo(distance, observation, speed, age, min_lateral_m=STRAIGHT_MIN_LATERAL_M,
                   max_bearing_deg=STRAIGHT_MAX_BEARING_DEG, tolerance=STRAIGHT_RANGE_TOLERANCE_M):
    """(x, y) de um cone visto (quadro ou mapa SLAM) na distancia do eco e ao lado do
    caminho reto (|y| >= min_lateral_m), ou None."""
    travelled = max(0., speed)*age
    for x, y in _visible_cones(observation):
        x = float(x) - travelled - SENSOR_X_M   # o carro andou desde a foto
        rng, bearing = math.hypot(x, y), math.degrees(math.atan2(abs(y), x))
        if x > 0 and abs(rng - distance) <= tolerance and abs(y) >= min_lateral_m and bearing <= max_bearing_deg:
            return x, float(y)
    return None


def sign_echo_matches(distance, position):
    if position is None:
        return False
    x, y = position[0] - SENSOR_X_M, position[1]
    return (x > 0 and abs(y) <= x and abs(math.hypot(x, y) - distance) <= RESUME_ECHO_TOLERANCE_M)


def resume_border_echo(distance, observation, steer_rad, now, wheelbase, front_overhang,
                       motion=None, sign_position=None):
    """Na saida do PARE, associa o eco a um cone real e testa a folga do carro."""
    corridor = getattr(observation, 'corridor', None)
    if (corridor is None or not corridor.valid or corridor.state not in ONE_OR_TWO_BORDERS
            or corridor.state_vector is None or not 0. <= now - observation.timestamp <= MAX_AGE_S
            or not math.isfinite(steer_rad)
            or (motion is None and now - observation.timestamp > .12)):
        return False, 'sem geometria confiavel para a retomada'
    vector = corridor.state_vector
    cones = np.asarray(_visible_cones(observation), float).reshape(-1, 2)
    if motion is not None:
        dx, dy, angle = motion  # eixo traseiro na foto -> eixo traseiro agora
        c, s = math.cos(angle), math.sin(angle)
        front = np.array([dx + wheelbase*(c - 1), dy + wheelbase*s])
        vector = move_states(vector, *front, angle)[0]
        cones = (cones - front) @ np.array([[c, -s], [s, c]])
    if sign_position is not None:
        cones = np.vstack((cones, sign_position))

    matches = cones[[sign_echo_matches(distance, cone) for cone in cones]]
    if not len(matches):
        return False, 'eco sem cone identificado na distancia medida'
    cross, _, _, _ = offsets(matches, vector)
    if np.any(np.abs(cross) < max(STRAIGHT_MIN_LATERAL_M, vector[4]/2 - .18)):
        return False, 'eco compativel com ponto dentro do corredor'

    # Amostra o eixo da carroceria (eixo traseiro ate a frente) nas proximas
    # poses dadas pelo esterco atual. A distancia inclui o arrasto da traseira.
    advance = np.linspace(0., distance + .25, 25)
    curvature = math.tan(steer_rad) / wheelbase
    heading = advance * curvature
    rear = np.column_stack((advance*np.sinc(heading/np.pi) - wheelbase,
                            .5*advance*heading*np.sinc(heading/(2*np.pi))**2))
    tangent = np.column_stack((np.cos(heading), np.sin(heading)))
    delta = matches[:, None, :] - rear
    along = np.clip(np.einsum('cij,ij->ci', delta, tangent), 0., wheelbase + front_overhang)
    clearance = np.min(np.linalg.norm(delta - along[:, :, None]*tangent, axis=-1))
    if clearance < STRAIGHT_MIN_LATERAL_M:
        return False, 'cone dentro da trajetoria da carroceria na retomada'
    return True, 'eco associado a cone da borda, fora da trajetoria de retomada'


def curve_echo(distance, observation, steer_rad, speed, now):
    """(ignorar a parada?, motivo). steer_rad: esterco geometrico (esquerda +)."""
    corridor = getattr(observation, 'corridor', None)
    if corridor is None or not corridor.valid or corridor.state not in ONE_OR_TWO_BORDERS \
            or corridor.state_vector is None:
        return False, 'sem corredor da visao'
    try:
        age = float(now - observation.timestamp)
    except (TypeError, ValueError):
        return False, 'horario da visao invalido'
    if not 0. <= age <= MAX_AGE_S:
        return False, f'visao atrasada ({age:.2f} s)'
    vector = np.asarray(corridor.state_vector, float)
    e, kappa, dkappa, width = vector[0], vector[2], vector[3], vector[4]
    ahead = kappa + dkappa*np.linspace(0., CURVE_LOOKAHEAD_M, 16)
    turn = ahead[np.argmax(np.abs(ahead))]
    straight = abs(turn) < MIN_CURVATURE
    # Stanley e pure pursuit seguem a curva: nao precisa a roda ja estar virada,
    # so nao pode estar virada claramente para o lado contrario.
    if not math.isfinite(steer_rad) or (not straight and abs(steer_rad) >= math.radians(MIN_STEER_DEG)
                                        and math.copysign(1., steer_rad) != math.copysign(1., turn)):
        return False, 'direcao virada para fora da curva'
    if abs(e) > width/2 - CAR_HALF_WIDTH_M - SIDE_MARGIN_M:
        return False, 'carro perto da borda'
    if straight:
        cone = real_cone_echo(distance, observation, speed, age)
        if cone is None:
            return False, 'eco no meio da faixa (obstaculo)'
        return True, f'cone da borda {"esquerda" if cone[1] > 0 else "direita"} na reta'
    # O estado e do instante da foto: desde entao o carro andou ~v*idade.
    side = border_echo(distance + max(0., speed)*age, vector)
    if side is None:
        return False, 'eco no meio da faixa (obstaculo)'
    return True, f'cone da borda {side} na curva'
