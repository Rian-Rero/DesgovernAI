"""Parada pelo ultrassom em curva: ignora o eco que e cone da borda fora do caminho.

Em curva, com a direcao ja esterçada, o ultrassom (que aponta para frente) ve o
cone da borda de fora muito perto e travava o carro, embora ele fosse passar ao
lado. A visao sabe onde estao as bordas (estimador do corredor): se o eco esta
exatamente onde ha uma borda, dentro do feixe e na distancia medida, e o carro
esta virando para o lado da curva e dentro do corredor com folga, o eco e o cone
da borda e a parada e ignorada. Obstaculo no meio da faixa nao coincide com
borda nenhuma: para como antes. Reta, sem corredor valido ou sem esterco: para
como antes.
"""
import math

import numpy as np

from .corridor import sample_centerline

# HC-SR04: feixe nominal de +-15 graus, mas cone perto e de lado ecoa ate ~+-35.
BEAM_HALF_DEG = 35.
ECHO_TOLERANCE_M = .25    # eco e borda a mesma distancia, com esta folga (borda junto ao carro e extrapolada)
MIN_STEER_DEG = 6.        # roda virada para o lado CONTRARIO da curva alem disto: nao confia
MIN_CURVATURE = .15       # 1/m (R <= 6,7 m) em algum ponto do proximo 1,5 m de caminho
CURVE_LOOKAHEAD_M = 1.5
CAR_HALF_WIDTH_M = .13
SIDE_MARGIN_M = .05       # folga lateral minima ate a borda, alem da meia largura do carro
SENSOR_X_M = .05          # ultrassom a frente do eixo dianteiro
MAX_AGE_S = .5            # estado da visao mais velho que isto nao serve
ONE_OR_TWO_BORDERS = ('AMBAS', 'SO_EXTERNA', 'SO_INTERNA', 'UMA_BORDA_RETA', 'MEMORIA')


def border_echo(distance, state_vector, sensor_x=SENSOR_X_M, beam_half_deg=BEAM_HALF_DEG,
                tolerance=ECHO_TOLERANCE_M):
    """'esquerda'/'direita' se ha ponto dessa borda no feixe, a `distance` m do sensor; senao None."""
    width = float(state_vector[4])
    arcs = np.arange(-.6, 3.5, .02)
    for side, offset in (('esquerda', width/2), ('direita', -width/2)):
        points = sample_centerline(state_vector, arcs, offset) - [sensor_x, 0.]
        ranges = np.hypot(points[:, 0], points[:, 1])
        bearing = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        hit = ((points[:, 0] > 0) & (np.abs(ranges - distance) <= tolerance)
               & (np.abs(bearing) <= beam_half_deg))
        if np.any(hit):
            return side
    return None


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
    if abs(turn) < MIN_CURVATURE:
        return False, 'reta'
    # Stanley e pure pursuit seguem a curva: nao precisa a roda ja estar virada,
    # so nao pode estar virada claramente para o lado contrario.
    if not math.isfinite(steer_rad) or (abs(steer_rad) >= math.radians(MIN_STEER_DEG)
                                        and math.copysign(1., steer_rad) != math.copysign(1., turn)):
        return False, 'direcao virada para fora da curva'
    if abs(e) > width/2 - CAR_HALF_WIDTH_M - SIDE_MARGIN_M:
        return False, 'carro perto da borda'
    # O estado e do instante da foto: desde entao o carro andou ~v*idade.
    side = border_echo(distance + max(0., speed)*age, vector)
    if side is None:
        return False, 'eco no meio da faixa (obstaculo)'
    return True, f'cone da borda {side} na curva'
