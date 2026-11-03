"""Cones RGB -> contato com o solo -> corredor metrico (frente, esquerda).

Nao consulta objetos da cena. As unicas informacoes geometricas externas sao
calibracao/pose da camera e pose do eixo dianteiro, como no seguidor amarelo.
"""
from dataclasses import dataclass, field
from itertools import combinations
import math

import cv2
import numpy as np

from .lateral import fit_line_errors, project_ground


def _empty(n=3):
    return np.empty((0, n), dtype=float)


def _validate(parameters):
    defaults = {'cone_max_distance': 12., 'cone_max_lateral': 8.,
                'cone_memory_horizon': 6., 'cone_freshness': .5,
                'cone_single_side_timeout': 12.,
                'cone_min_span': 1., 'cone_max_near_distance': 6.,
                'cone_min_width': .75, 'cone_max_width': 7.}
    for name, default in defaults.items():
        value = parameters.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} deve ser finito e positivo.')
    area = parameters.get('cone_min_area', 3)
    if isinstance(area, bool) or not isinstance(area, int) or area < 1:
        raise ValueError('cone_min_area deve ser inteiro positivo.')
    if parameters.get('cone_min_width', .75) >= parameters.get('cone_max_width', 7.):
        raise ValueError('cone_min_width deve ser menor que cone_max_width.')
    if parameters.get('cone_max_near_distance', 6.) > parameters.get('cone_max_distance', 12.):
        raise ValueError('cone_max_near_distance nao pode exceder cone_max_distance.')


@dataclass
class ConeObservation:
    errors: tuple | None = None
    path: np.ndarray = field(default_factory=_empty)
    red_points: np.ndarray = field(default_factory=_empty)
    green_points: np.ndarray = field(default_factory=_empty)
    red_pixels: np.ndarray = field(default_factory=lambda: _empty(2))
    green_pixels: np.ndarray = field(default_factory=lambda: _empty(2))
    red_lateral_m: float = math.nan
    green_lateral_m: float = math.nan
    reason: str = 'Cones insuficientes nas duas bordas'
    inferred_color: str | None = None

    @property
    def valid(self):
        return self.errors is not None and all(math.isfinite(v) for v in self.errors)


def detect_cones(image, min_area=3):
    """Bases de componentes vermelhos/verdes, em pixels (coluna, linha).

    A base, e nao o centro do corpo alto, e projetada no chao. Rejeita blobs
    largos, sem afunilamento e recortes da imagem que escondem o contato.
    Entrada RGB uint8; o vermelho inclui os dois extremos do circulo HSV.
    """
    result = {'red': _empty(2), 'green': _empty(2)}
    if image is None:
        return result
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return result
    hsv = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    colored = (saturation >= 110) & (value >= 45)
    masks = {'red': colored & ((hue <= 10) | (hue >= 170)),
             'green': colored & (hue >= 38) & (hue <= 83)}
    height, width = image.shape[:2]
    for color, mask in masks.items():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        bases = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < min_area or h < 3 or y+h >= height:
                continue
            component = labels[y:y+h, x:x+w] == label
            if w > 1.5*h:
                # A base de um cone vermelho pode tocar a zebra vermelha.
                # Retira apenas a faixa horizontal comprida para separar o
                # corpo vertical; depois recupera seu contato na mascara original.
                stripe = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_OPEN,
                                          np.ones((1, max(7, int(.6*h))), np.uint8))
                bodies = component & ~stripe.astype(bool)
                bodies = cv2.morphologyEx(bodies.astype(np.uint8), cv2.MORPH_CLOSE,
                                           np.ones((5, 1), np.uint8)).astype(bool) & component
                n, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(bodies.astype(np.uint8), 8)
                for body_id in range(1, n):
                    bx, by, bw, bh, ba = body_stats[body_id]
                    if ba < min_area or bh < 5 or bw < 2 or bw > 1.5*bh:
                        continue
                    body = body_labels[by:by+bh, bx:bx+bw] == body_id
                    top = np.count_nonzero(np.any(body[:max(1, bh//3)], axis=0))
                    bottom = np.count_nonzero(np.any(body[-max(1, bh//3):], axis=0))
                    if top >= .95*bottom:
                        continue
                    cx, row = x+bx+(bw-1)/2, y+by+bh-1
                    for candidate in range(row+1, y+h):
                        if mask[candidate, int(round(cx))]:
                            row = candidate
                        else:
                            break
                    if 0 < cx < width-1:
                        bases.append((cx, float(row)))
                continue
            if w < 2 or x == 0 or x+w >= width:
                continue
            if h >= 7:
                top = np.flatnonzero(np.any(component[:max(1, h//3)], axis=0))
                bottom = np.flatnonzero(np.any(component[-max(1, h//3):], axis=0))
                if len(top) >= .95*len(bottom):
                    continue
            rows = np.flatnonzero(np.any(component, axis=1))
            base_row = rows[-1]
            columns = np.flatnonzero(np.any(component[max(0, base_row-1):base_row+1], axis=0))
            bases.append((x+float(np.mean(columns)), y+float(base_row)))
        result[color] = np.asarray(bases, dtype=float).reshape(-1, 2)
    return result


class _ConeMemory:
    def __init__(self, horizon, freshness):
        self.horizon, self.freshness = horizon, freshness
        self.tracks = []
        self.last_seen = -math.inf

    def update(self, now, points):
        self.tracks = [(stamp, point) for stamp, point in self.tracks if now-stamp <= self.horizon]
        if len(points):
            if now-self.last_seen > self.freshness:
                self.tracks.clear()
            matched = set()
            for point in points:
                distances = [float(np.linalg.norm(point[:2]-known[:2])) if index not in matched else math.inf
                             for index, (_, known) in enumerate(self.tracks)]
                index = int(np.argmin(distances)) if distances else -1
                if index >= 0 and distances[index] < .7:
                    # Associa espacialmente o MESMO cone. Quantizar em celulas
                    # criava varios cones falsos quando a suspensao oscilava.
                    self.tracks[index] = (now, .65*self.tracks[index][1]+.35*point)
                else:
                    index = len(self.tracks)
                    self.tracks.append((now, point.copy()))
                matched.add(index)
            self.last_seen = now
        if not self.tracks or now-self.last_seen > self.freshness:
            return _empty()
        return np.asarray([point for _, point in self.tracks])


def _local(points, front, forward):
    points = points[np.all(np.isfinite(points), axis=1)]
    delta = points[:, :2]-np.asarray(front)[:2]
    left = np.array([-forward[1], forward[0]])
    return np.column_stack((delta@forward, delta@left))


def _boundary(points, max_distance, max_lateral):
    """Seleciona a borda proxima, evitando ramos alem do corredor visivel."""
    points = points[(points[:, 0] >= -.8) & (points[:, 0] <= max_distance)
                    & (np.abs(points[:, 1]) <= max_lateral)]
    if len(points) < 2:
        return _empty(2)
    # Em cada faixa de profundidade, preferir o cone da borda mais proxima,
    # nao os cones de outra volta da pista que aparecem atras dela.
    points = points[np.argsort(points[:, 0])]
    bins = np.round(points[:, 0]/.45)
    selected = []
    for bin_index in np.unique(bins):
        group = points[bins == bin_index]
        selected.append(group[np.argmin(np.abs(group[:, 1]))])
    points = np.asarray(selected)
    if len(points) < 2:
        return points
    # Ancora no cone espacialmente mais proximo. Comecar pelo menor x podia
    # selecionar outro trecho da pista quase ao lado do carro, mas a 5 m.
    anchor = int(np.argmin(np.linalg.norm(points, axis=1)))
    keep = [anchor]
    for indices in (range(anchor+1, len(points)), range(anchor-1, -1, -1)):
        previous = points[anchor]
        for index in indices:
            dx, dy = points[index]-previous
            if abs(dy) <= max(.7, 1.3*abs(dx)):
                keep.append(index)
                previous = points[index]
    return points[sorted(keep)]


def _fit_boundary(points):
    degree = 2 if len(points) >= 3 and np.ptp(points[:, 0]) >= 2. else 1
    initial = np.polyfit(points[:, 0], points[:, 1], degree)
    initial_residual = np.abs(points[:, 1]-np.polyval(initial, points[:, 0]))
    initial_slope = float(np.polyval(np.polyder(initial), 0.))
    if (np.max(initial_residual) <= .18 and abs(initial_slope) <= 2.5
            and abs(float(np.polyval(initial, 0.))) <= 8.
            and (degree == 1 or abs(initial[0]) <= .8)):
        return initial, points, float(np.sqrt(np.mean(initial_residual**2)))
    best = None
    # RANSAC deterministico para o pequeno conjunto de cones proximos.
    # Uma mediana de residuos apos ajustar TODOS os pontos dava suporte a
    # cones de outro ramo visivel entre os cones da borda correta.
    for candidate_degree in range(1, degree+1):
        for indices in combinations(range(len(points)), candidate_degree+1):
            support = points[list(indices)]
            if np.ptp(support[:, 0]) < 1. or np.min(np.diff(support[:, 0])) < .35:
                continue
            coefficients = np.polyfit(support[:, 0], support[:, 1], candidate_degree)
            # Avaliar plausibilidade ANTES de escolher a hipotese. Tres
            # pontos de ramos diferentes tambem definem uma parabola exata,
            # mas podem extrapolar 20 m e 80 graus ate o eixo dianteiro.
            slope = float(np.polyval(np.polyder(coefficients), 0.))
            intercept = float(np.polyval(coefficients, 0.))
            if abs(slope) > 2.5 or abs(intercept) > 8.:
                continue
            if candidate_degree == 2 and abs(coefficients[0]) > .8:
                continue
            residual = np.abs(points[:, 1]-np.polyval(coefficients, points[:, 0]))
            good = residual <= .18
            # Igual suporte prefere reta. Distancia desempata levemente;
            # nunca deve fazer mistura de tres cones proximos superar tres
            # cones alinhados da borda correta um pouco mais a frente.
            score = float(np.count_nonzero(good))-.12*(candidate_degree-1)-.001*float(np.sum(points[good, 0]))
            if best is None or score > best[0]:
                best = (score, good, candidate_degree)
    if best is not None:
        _, good, degree = best
        points = points[good]
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    rms = float(np.sqrt(np.mean((points[:, 1]-np.polyval(coefficients, points[:, 0]))**2)))
    return coefficients, points, rms


def _circle(points):
    if len(points) < 3:
        return None
    x, y = points.T
    design = np.column_stack((2*x, 2*y, np.ones(len(x))))
    values, _, rank, _ = np.linalg.lstsq(design, x*x+y*y, rcond=None)
    cx, cy, constant = values
    radius_squared = constant+cx*cx+cy*cy
    if rank < 3 or radius_squared <= 0:
        return None
    radius = math.sqrt(radius_squared)
    residual = float(np.sqrt(np.mean((np.hypot(x-cx, y-cy)-radius)**2)))
    if not .65 < radius < 30 or residual > .15:
        return None
    return np.array([cx, cy]), radius, residual


def reconstruct_cone_path(red, green, front, forward, ground_z=0., parameters=None):
    """Reconstrucao independente da camera; util para validar a geometria.

    As duas bordas sao ajustadas no mesmo referencial. Nao exige cones
    exatamente aos pares: os modelos sao avaliados na mesma profundidade.
    A extrapolacao ate o carro e limitada pelo cone mais proximo e pelo
    comprimento realmente observado em ambas as bordas.
    """
    config = {} if parameters is None else parameters
    _validate(config)
    forward = np.asarray(forward, dtype=float)[:2].copy()
    if (not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-9
            or not np.isfinite(front).all()):
        return ConeObservation(reason='Orientacao do veiculo invalida')
    forward /= np.linalg.norm(forward)
    maximum = config.get('cone_max_distance', 12.)
    lateral_limit = config.get('cone_max_lateral', 8.)
    red = _boundary(_local(np.asarray(red).reshape(-1, 3), front, forward), maximum, lateral_limit)
    green = _boundary(_local(np.asarray(green).reshape(-1, 3), front, forward), maximum, lateral_limit)
    result = ConeObservation()
    if min(len(red), len(green)) < 2:
        return result
    span = config.get('cone_min_span', 1.)
    if min(np.ptp(red[:, 0]), np.ptp(green[:, 0])) < span:
        result.reason = 'Cones sem comprimento suficiente para orientar a trajetoria'
        return result
    nearest = max(float(np.min(red[:, 0])), float(np.min(green[:, 0])))
    if nearest > config.get('cone_max_near_distance', 6.):
        result.reason = 'Corredor de cones distante demais para orientar o carro'
        return result
    # Priorizamos apoio local: cones muito distantes de uma curva futura nao
    # devem alterar a tangente da reta em que o eixo dianteiro ainda esta.
    upper = min(maximum, max(5., nearest+4.))
    red = red[red[:, 0] <= upper]
    green = green[green[:, 0] <= upper]
    if min(len(red), len(green)) < 2:
        return result
    red_fit, red, red_rms = _fit_boundary(red)
    green_fit, green, green_rms = _fit_boundary(green)
    if max(red_rms, green_rms) > .3:
        result.reason = 'Bordas de cones inconsistentes'
        return result
    end = min(float(np.max(red[:, 0])), float(np.max(green[:, 0])))
    start = max(0., nearest)
    if end-start < .5*span:
        result.reason = 'Sem sobreposicao longitudinal entre as duas bordas'
        return result
    xs = np.linspace(0., min(end, 4.), 45)
    red_y, green_y = np.polyval(red_fit, xs), np.polyval(green_fit, xs)
    widths = np.abs(red_y-green_y)
    min_width, max_width = config.get('cone_min_width', .75), config.get('cone_max_width', 7.)
    if (np.min(widths) < min_width or np.max(widths) > max_width
            or np.any((red_y-green_y)*(red_y[0]-green_y[0]) <= 0)
            or max(abs(np.polyval(np.polyder(red_fit), 0.)),
                   abs(np.polyval(np.polyder(green_fit), 0.))) > 2.5):
        result.reason = 'Bordas cruzadas ou largura de corredor implausivel'
        return result
    center = (red_y+green_y)/2
    result.red_lateral_m, result.green_lateral_m = float(red_y[0]), float(green_y[0])
    # Em arcos as bordas sao concentricas. Ajusta-las como y(x) e tomar sua
    # media desloca o centro, especialmente na borda interna de curva fechada.
    red_circle, green_circle = _circle(red), _circle(green)
    use_circle = False
    if red_circle is not None and green_circle is not None:
        rc, rr, re = red_circle
        gc, gr, ge = green_circle
        radius = (rr+gr)/2
        use_circle = (np.linalg.norm(rc-gc) < .6 and min_width < abs(rr-gr) < max_width
                      and radius < 20 and re+ge < max(.025, red_rms+green_rms))
        if use_circle:
            origin = (rc+gc)/2
            distance = float(np.linalg.norm(origin))
            if distance > .1:
                tangent = np.array([origin[1], -origin[0]])/distance
                if tangent[0] < 0:
                    tangent = -tangent
                normal = np.array([-tangent[1], tangent[0]])
                turn = math.copysign(1., float(origin@normal))
                cross_track = turn*(distance-radius)
                heading = math.atan2(tangent[1], tangent[0])
                result.errors = (cross_track, heading, turn/radius)
                result.red_lateral_m = turn*(float(np.linalg.norm(rc))-rr)
                result.green_lateral_m = turn*(float(np.linalg.norm(gc))-gr)
                radial = -origin/distance
                angles = turn*np.linspace(0., min(end, 4.)/radius, 45)
                rotations = np.stack((np.cos(angles)*radial[0]-np.sin(angles)*radial[1],
                                      np.sin(angles)*radial[0]+np.cos(angles)*radial[1]), axis=1)
                local_path = origin+radius*rotations
            else:
                use_circle = False
    if not use_circle:
        local_path = np.column_stack((xs, center))
        # Mesmo erro normal e tangente usados no seguidor amarelo/Stanley.
        # O trecho extrapolado so e criado apos validar apoio de ambas bordas.
        world_xy = (np.asarray(front)[:2]+local_path[:, 0, None]*forward
                    +local_path[:, 1, None]*np.array([-forward[1], forward[0]]))
        path = np.column_stack((world_xy, np.full(len(world_xy), ground_z)))
        result.errors = fit_line_errors(path, front, forward, with_curvature=True)
    world_xy = (np.asarray(front)[:2]+local_path[:, 0, None]*forward
                +local_path[:, 1, None]*np.array([-forward[1], forward[0]]))
    result.path = np.column_stack((world_xy, np.full(len(world_xy), ground_z)))
    result.reason = 'Centro do corredor de cones' if result.valid else 'Centro dos cones sem ajuste local confiavel'
    return result


class ConePathDetector:
    def __init__(self, parameters=None):
        self.parameters = dict(parameters or {})
        _validate(self.parameters)
        horizon = self.parameters.get('cone_memory_horizon', 6.)
        freshness = self.parameters.get('cone_freshness', .5)
        if not all(math.isfinite(v) and v > 0 for v in (horizon, freshness)):
            raise ValueError('Tempos da memoria de cones devem ser finitos e positivos.')
        self.memory = {color: _ConeMemory(horizon, freshness) for color in ('red', 'green')}
        self.width = None
        self.width_time = -math.inf
        self.sides = {}

    def _single_side(self, color, points, front, forward, ground_z):
        """Compensa uma oclusao com a largura medida antes nas DUAS bordas."""
        forward = np.asarray(forward, dtype=float)[:2].copy()
        if not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-9 or not np.isfinite(front).all():
            return None
        forward /= np.linalg.norm(forward)
        local = _boundary(_local(points, front, forward),
                          self.parameters.get('cone_max_distance', 12.),
                          self.parameters.get('cone_max_lateral', 8.))
        if len(local) < 3 or np.ptp(local[:, 0]) < 1.5:
            return None
        if np.min(local[:, 0]) > self.parameters.get('cone_max_near_distance', 6.):
            return None
        local = local[local[:, 0] <= max(5., float(np.min(local[:, 0]))+4.)]
        if len(local) < 3:
            return None
        coefficients, support, rms = _fit_boundary(local)
        if len(support) < 3 or rms > .2:
            return None
        side, half_width = self.sides[color], self.width/2
        arc = _circle(support)
        if arc is not None and arc[1] < 20.:
            origin, radius, _ = arc
            distance = float(np.linalg.norm(origin))
            if distance < .1:
                return None
            tangent = np.array([origin[1], -origin[0]])/distance
            if tangent[0] < 0:
                tangent = -tangent
            turn = math.copysign(1., float(origin@np.array([-tangent[1], tangent[0]])))
            center_radius = radius+side*turn*half_width
            if not .8 < center_radius < 30.:
                return None
            errors = (turn*(distance-center_radius), math.atan2(tangent[1], tangent[0]),
                      turn/center_radius)
            lateral = turn*(distance-radius)
            radial = -origin/distance
            angles = turn*np.linspace(0., min(4., float(np.max(support[:, 0])))/center_radius, 45)
            local_path = origin+center_radius*np.column_stack((
                np.cos(angles)*radial[0]-np.sin(angles)*radial[1],
                np.sin(angles)*radial[0]+np.cos(angles)*radial[1]))
        else:
            xs = np.linspace(0., min(4., float(np.max(support[:, 0]))), 45)
            slopes = np.polyval(np.polyder(coefficients), xs)
            normals = np.column_stack((-slopes, np.ones(len(xs))))/np.sqrt(1+slopes**2)[:, None]
            local_path = np.column_stack((xs, np.polyval(coefficients, xs)))-side*half_width*normals
            lateral = float(np.polyval(coefficients, 0.))/math.sqrt(1+float(np.polyval(np.polyder(coefficients), 0.))**2)
            errors = None
        left = np.array([-forward[1], forward[0]])
        xy = np.asarray(front)[:2]+local_path[:, 0, None]*forward+local_path[:, 1, None]*left
        path = np.column_stack((xy, np.full(len(xy), ground_z)))
        if errors is None:
            errors = fit_line_errors(path, front, forward, with_curvature=True)
        if errors is None or abs(errors[0]) > self.width/2 or abs(errors[1]) > 1.:
            return None
        distances = {color: lateral, 'green' if color == 'red' else 'red': lateral-side*self.width}
        return ConeObservation(errors=errors, path=path, red_lateral_m=distances['red'],
                               green_lateral_m=distances['green'],
                               inferred_color='green' if color == 'red' else 'red',
                               reason='Oclusao: centro pela borda visivel e largura ja medida')

    def observe(self, image, camera_matrix, angle, ground_z, front, forward, time):
        if not math.isfinite(time):
            return ConeObservation(reason='Instante da camera invalido')
        pixels = detect_cones(image, self.parameters.get('cone_min_area', 3))
        clouds = {}
        for color in ('red', 'green'):
            points = (project_ground(pixels[color], image.shape, camera_matrix, angle, ground_z)
                      if len(pixels[color]) else _empty())
            clouds[color] = self.memory[color].update(time, points)
        observation = reconstruct_cone_path(clouds['red'], clouds['green'], front, forward,
                                             ground_z, self.parameters)
        if observation.valid:
            width = abs(observation.red_lateral_m-observation.green_lateral_m)
            if self.width is None or .75*self.width <= width <= 1.25*self.width:
                self.width = width if self.width is None else .9*self.width+.1*width
                self.width_time = time
                self.sides = {'red': math.copysign(1., observation.red_lateral_m-observation.green_lateral_m)}
                self.sides['green'] = -self.sides['red']
        elif (self.width is not None
              and time-self.width_time <= self.parameters.get('cone_single_side_timeout', 12.)):
            for color, other in (('red', 'green'), ('green', 'red')):
                if len(clouds[color]) >= 3 and len(clouds[other]) < 2:
                    inferred = self._single_side(color, clouds[color], front, forward, ground_z)
                    if inferred is not None:
                        observation = inferred
                        break
        observation.red_pixels, observation.green_pixels = pixels['red'], pixels['green']
        observation.red_points, observation.green_points = clouds['red'], clouds['green']
        if not observation.valid and (not len(clouds['red']) or not len(clouds['green'])):
            observation.reason = 'Cones vermelhos e verdes recentes insuficientes'
        return observation
