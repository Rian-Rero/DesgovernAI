"""Trajetoria pelo meio dos "portoes" de cones, no piso (metros).

Substitui como metodo principal a cadeia em pixels + curvas por secoes normais:

1. Bordas: cada cor e encadeada no piso a partir do cone mais proximo,
   com espacamento metrico e limite de curva entre elos; com a borda anterior
   (levada pela odometria) como semente, nao pula para outra fileira.
2. Portoes: cada cone forma par com o ponto mais proximo da borda oposta; o
   ponto medio e um ponto do centro. Cones intercalados e espacamentos
   diferentes por lado nao atrapalham.
3. Ajuste robusto: polinomio em comprimento de arco, descartando o ponto
   medio que mais destoa (falso positivo no meio do corredor).

Convencao: frente +x, esquerda +y, origem no eixo dianteiro.
"""
import math

import numpy as np


def _closest_on_polyline(point, polyline):
    """(ponto mais proximo, distancia, indice do segmento) de point na polilinha."""
    polyline = np.asarray(polyline, float).reshape(-1, 2)
    if len(polyline) == 1:
        return polyline[0], float(np.linalg.norm(point-polyline[0])), 0
    a, b = polyline[:-1], polyline[1:]
    d = b - a
    length = np.maximum(np.sum(d*d, axis=1), 1e-12)
    t = np.clip(np.sum((point-a)*d, axis=1)/length, 0., 1.)
    q = a + t[:, None]*d
    distance = np.linalg.norm(q-point, axis=1)
    index = int(np.argmin(distance))
    return q[index], float(distance[index]), index


def _tangent(polyline, index):
    polyline = np.asarray(polyline, float)
    if len(polyline) < 2:
        return np.array([1., 0.])
    i = int(np.clip(index, 0, len(polyline)-2))
    d = polyline[i+1]-polyline[i]
    n = np.linalg.norm(d)
    return d/n if n > 1e-9 else np.array([1., 0.])


def chain_boundary(cones, prior=None, *, min_gap=.2, max_gap=1.6, max_turn_deg=60., behind=-.4,
                   prior_gate=.35):
    """Ordena os cones de uma borda do mais proximo ao mais distante.

    prior: borda anterior (polilinha no referencial atual). Com ela, comeca no
    cone mais proximo do carro que esta sobre a borda prevista e penaliza
    afastar-se dela, o que impede ligar outra fileira da mesma cor.
    """
    cones = np.asarray(cones, float).reshape(-1, 2)
    cones = cones[cones[:, 0] >= behind]
    if not len(cones):
        return np.empty((0, 2))
    prior = None if prior is None or len(prior) < 2 else np.asarray(prior, float)
    order = np.argsort(np.hypot(cones[:, 0], cones[:, 1]*.5))
    start = order[0]
    if prior is not None:
        near_prior = [i for i in order if _closest_on_polyline(cones[i], prior)[1] <= prior_gate]
        if near_prior:
            start = near_prior[0]
    chain = [int(start)]
    used = {int(start)}
    if prior is not None:
        direction = _tangent(prior, _closest_on_polyline(cones[start], prior)[2])
    else:
        direction = np.array([1., 0.])
    max_turn = math.radians(max_turn_deg)
    while True:
        current = cones[chain[-1]]
        best, best_cost = None, math.inf
        for i, point in enumerate(cones):
            if i in used:
                continue
            step = point - current
            distance = float(np.linalg.norm(step))
            if not min_gap <= distance <= max_gap:
                continue
            along = float(step @ direction)
            if along <= 0:
                continue
            turn = math.acos(min(1., along/distance))
            if turn > max_turn:
                continue
            cost = distance*(1. + 1.5*turn)
            if prior is not None:
                cost += 2.*min(_closest_on_polyline(point, prior)[1], .6)
            if cost < best_cost:
                best, best_cost = i, cost
        if best is None:
            break
        step = cones[best] - current
        direction = step/np.linalg.norm(step)
        chain.append(best)
        used.add(best)
    return cones[chain]


def gate_midpoints(left, right, *, min_width=.4, max_width=1.4):
    """Pontos medios dos portoes e larguras; descarta pares cruzados/fora da largura."""
    left = np.asarray(left, float).reshape(-1, 2)
    right = np.asarray(right, float).reshape(-1, 2)
    mids, widths = [], []
    for cones, other, sign in ((left, right, 1.), (right, left, -1.)):
        if len(other) < 2:
            continue
        for point in cones:
            q, width, index = _closest_on_polyline(point, other)
            if not min_width <= width <= max_width:
                continue
            # O cone esquerdo precisa ficar a esquerda do direito (normal da borda).
            tangent = _tangent(other, index)
            normal = np.array([-tangent[1], tangent[0]])
            separation = point - q
            # Um pareamento preso na ponta da borda oposta pode ser diagonal.
            # Aceita ate 25 cm de desencontro longitudinal entre as fileiras.
            if abs(separation @ tangent) > .25 or sign*(separation @ normal) <= 0:
                continue
            mids.append((point + q)/2)
            widths.append(width)
    if not mids:
        return np.empty((0, 2)), np.empty(0)
    mids, widths = np.asarray(mids), np.asarray(widths)
    # Portoes quase iguais (cone de um lado casado com o do outro e vice-versa).
    keep = []
    for i, mid in enumerate(mids):
        if all(np.linalg.norm(mid - mids[j]) > .08 for j in keep):
            keep.append(i)
    return mids[keep], widths[keep]


def virtual_midpoints(boundary, width, boundary_is_left):
    """Uma borda so: o centro fica a meia largura pela normal (largura aprendida)."""
    boundary = np.asarray(boundary, float).reshape(-1, 2)
    if len(boundary) < 2:
        return np.empty((0, 2))
    tangent = np.gradient(boundary, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    return boundary + (-1. if boundary_is_left else 1.)*width/2*normal


def order_path(points):
    """Encadeia pontos do mais proximo do carro em diante (vizinho mais proximo a frente)."""
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 2:
        return points
    remaining = list(range(len(points)))
    current = int(np.argmin(np.hypot(points[:, 0], points[:, 1]*.5)))
    order = [current]
    remaining.remove(current)
    direction = np.array([1., 0.])
    while remaining:
        steps = points[remaining] - points[current]
        distance = np.linalg.norm(steps, axis=1)
        along = steps @ direction
        cost = np.where(along > -.05, distance, distance + 10.)
        k = int(np.argmin(cost))
        nxt = remaining[k]
        if distance[k] > 1e-9 and along[k] > -.05:
            direction = steps[k]/distance[k]
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return points[order]


def fit_path(points, *, step=.05, residual_max=.12, max_outliers=2):
    """Polinomio em arco pelos pontos medios; devolve (curva, curvatura, info).

    Grau 1 com 2 pontos, 2 ate 5 pontos, 3 a partir de 6 (curva em S).
    Descarta ate max_outliers pontos com residuo > residual_max.
    """
    points = order_path(points)
    if len(points) < 2:
        return None
    keep = np.ones(len(points), bool)
    outliers = 0
    while True:
        used = points[keep]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(used, axis=0), axis=1))]
        if arc[-1] < .15:
            return None
        degree = 1 if len(used) == 2 else 2 if len(used) < 6 else 3
        cx = np.polyfit(arc, used[:, 0], degree)
        cy = np.polyfit(arc, used[:, 1], degree)
        residual = np.hypot(used[:, 0]-np.polyval(cx, arc), used[:, 1]-np.polyval(cy, arc))
        worst = int(np.argmax(residual))
        if residual[worst] <= residual_max or outliers >= max_outliers or len(used) <= 3:
            break
        keep[np.flatnonzero(keep)[worst]] = False
        outliers += 1
    s = np.arange(0., arc[-1] + 1e-9, step)
    if len(s) < 2:
        s = np.array([0., arc[-1]])
    curve = np.c_[np.polyval(cx, s), np.polyval(cy, s)]
    dx, dy = np.polyval(np.polyder(cx), s), np.polyval(np.polyder(cy), s)
    ddx = np.polyval(np.polyder(cx, 2), s) if degree >= 2 else np.zeros_like(s)
    ddy = np.polyval(np.polyder(cy, 2), s) if degree >= 2 else np.zeros_like(s)
    speed = np.maximum(np.hypot(dx, dy), 1e-9)
    curvature = (dx*ddy - dy*ddx)/speed**3
    rms = float(np.sqrt(np.mean(residual**2)))
    info = {'points': int(len(points)), 'used': int(keep.sum()), 'outliers': outliers,
            'degree': degree, 'rms_m': rms, 'length_m': float(arc[-1]),
            'outlier_points': points[~keep]}
    return curve, curvature, info


def blend_paths(new, previous, weight):
    """Funde o caminho novo com o anterior (ja no referencial atual).

    weight: fracao do anterior (0 = so o novo). Pontos do novo sem vizinho
    anterior proximo (trecho recem-visto) ficam como estao.
    Retorna (caminho, desvio medio m entre os dois).
    """
    previous = np.asarray(previous, float).reshape(-1, 2)
    if len(previous) < 2 or weight <= 0:
        return new, None
    matched, deviations = [], []
    output = new.copy()
    for i, point in enumerate(new):
        q, distance, _ = _closest_on_polyline(point, previous)
        if distance <= .35:
            output[i] = (1-weight)*point + weight*q
            deviations.append(distance)
            matched.append(i)
    deviation = float(np.mean(deviations)) if deviations else None
    return output, deviation


def curve_geometry(curve):
    """Rumo e curvatura por diferencas no caminho amostrado."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if len(curve) < 3 or np.any(np.diff(arc) < 1e-6):
        return np.zeros(len(curve))
    tangent = np.gradient(curve, arc, axis=0)
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    return np.gradient(heading, arc)
