"""Tracado pelo SVM: o centro da pista como a fronteira de MARGEM MAXIMA entre as bordas.

Cones da borda esquerda sao uma classe (+1), os da direita outra (-1). Um SVM de
kernel RBF acha a fronteira que fica o mais longe possivel dos cones dos dois
lados ao mesmo tempo: e o caminho com a maior folga, inclusive nas curvas, onde
"meio entre pares de cones" falha (os cones de dentro e de fora nao ficam
alinhados). Usado com o mapa do SLAM (cone_slam.py), que guarda os cones ao lado
e atras do carro, a fronteira perto do eixo vem de cones reais e nao de
extrapolacao.

So numpy (no Raspberry nao ha scikit-learn e o OpenCV 5 nao tem o modulo ml): o
problema dual com o vies absorvido no kernel (K + 1) vira uma caixa 0 <= a <= C,
resolvida por gradiente projetado. Com ate ~40 cones: poucos milissegundos.

Com uma borda so, a outra e criada a meia distancia conhecida (largura da pista),
ao longo da normal da borda vista ("cones virtuais").
"""
import math

import numpy as np


def rbf(a, b, gamma):
    d2 = np.sum(a*a, 1)[:, None] + np.sum(b*b, 1)[None, :] - 2*a @ b.T
    return np.exp(-gamma*np.maximum(d2, 0.))


class KernelSVM:
    """SVM de margem (quase) rigida, kernel RBF, dual por gradiente projetado."""
    def __init__(self, gamma=1., C=10., iterations=300):
        self.gamma, self.C, self.iterations = float(gamma), float(C), int(iterations)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        K = rbf(X, X, self.gamma) + 1.   # +1: vies dentro do kernel (sem a restricao de igualdade)
        Q = (y[:, None]*y[None, :])*K
        step = 1./max(float(np.linalg.eigvalsh(Q)[-1]), 1e-9)
        alpha = np.zeros(len(y))
        for _ in range(self.iterations):
            alpha = np.clip(alpha + step*(1. - Q @ alpha), 0., self.C)
        self.X, self.coef = X, alpha*y
        return self

    def decision(self, Z):
        Z = np.asarray(Z, float).reshape(-1, 2)
        return (rbf(Z, self.X, self.gamma) + 1.) @ self.coef


def _border_normal(points):
    """Normal unitaria (esquerda) da borda: reta pelos cones (um so: eixo do carro)."""
    if len(points) >= 2:
        centered = points - points.mean(0)
        direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        if direction[0] < 0:
            direction = -direction
    else:
        direction = np.array([1., 0.])
    return np.array([-direction[1], direction[0]])


def virtual_border(points, width, side):
    """Cones da borda que falta: os vistos deslocados `width` para o lado `side` (+1 esquerda)."""
    return points + side*width*_border_normal(points)


def densify(points, step=.25, max_gap=2.5):
    """Borda continua: pontos a cada `step` m entre cones vizinhos da mesma borda.

    Com cones a ~1 m e kernel de ~0,4 m, o SVM so com os cones fecha a fronteira
    em volta de cada um (rumo errado); com a borda preenchida, a fronteira segue
    a pista. Os cones sao encadeados do mais proximo do carro ao mais longe (o
    vizinho mais perto ainda nao usado); saltos maiores que max_gap nao sao
    preenchidos (pode ser outro trecho da pista)."""
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 2:
        return points
    remaining = list(range(len(points)))
    current = min(remaining, key=lambda k: points[k, 0])
    order = [current]
    remaining.remove(current)
    while remaining:
        current = min(remaining, key=lambda k: np.hypot(*(points[k] - points[order[-1]])))
        order.append(current)
        remaining.remove(current)
    chain = points[order]
    out = [chain[:1]]
    for a, b in zip(chain[:-1], chain[1:]):
        gap = float(np.hypot(*(b - a)))
        if gap <= max_gap:
            n = max(1, int(math.ceil(gap/step)))
            out.append(a + (b - a)*np.arange(1, n + 1)[:, None]/n)
        else:
            out.append(b[None])
    return np.vstack(out)


def centerline(left, right, width=1.3, x_range=None, step=.1, gamma=None, C=10.):
    """Pontos (Nx2) da fronteira de margem maxima entre a borda esquerda e a direita.

    left/right: cones no referencial do carro. Sem um dos lados, ele e virtual (a
    `width` da borda vista). Devolve (pontos, margem em m) ou (None, nan)."""
    left = np.asarray(left, float).reshape(-1, 2)
    right = np.asarray(right, float).reshape(-1, 2)
    if not len(left) and not len(right):
        return None, math.nan
    if not len(left):
        left = virtual_border(right, width, +1.)
    if not len(right):
        right = virtual_border(left, width, -1.)
    # Limites do tracado pelos cones reais; o SVM usa as bordas preenchidas.
    # Sem folga alem dos cones: nas pontas, com um lado so, a fronteira entorta
    # para o lado que tem menos pontos.
    low = max(left[:, 0].min(), right[:, 0].min())
    high = min(left[:, 0].max(), right[:, 0].max())
    cones = np.vstack([left, right])
    left, right = densify(left), densify(right)
    X = np.vstack([left, right])
    y = np.r_[np.ones(len(left)), -np.ones(len(right))]
    # Largura do kernel: numa curva de R = 2 m, 0,4 m deu 6,6 cm de erro medio ao
    # centro real; 0,6-1,3 m arredondavam demais a curva (rumo errado no eixo).
    scale = max(.4, .3*width)
    svm = KernelSVM(gamma=gamma or 1./(2*scale*scale), C=C).fit(X, y)
    # So onde ha cones dos dois lados (low/high): fora disso a fronteira e extrapolacao.
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step) if x_range else np.arange(max(-.6, low), high + 1e-9, step)
    lateral = np.arange(-2.5, 2.5 + 1e-9, .04)
    points, previous = [], None
    for x in xs:
        values = svm.decision(np.c_[np.full(len(lateral), x), lateral])
        crossings = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
        if not len(crossings):
            continue
        # Interpolacao linear no cruzamento; com varios, o mais perto do anterior.
        ys = [lateral[k] - values[k]*(lateral[k+1] - lateral[k])/(values[k+1] - values[k]) for k in crossings]
        if previous is None:  # primeiro: o cruzamento mais perto do meio das bordas
            target = .5*(left[np.argmin(np.abs(left[:, 0] - x)), 1] + right[np.argmin(np.abs(right[:, 0] - x)), 1])
        else:
            target = previous
        y0 = min(ys, key=lambda value: abs(value - target))
        points.append((x, y0))
        previous = y0
    if len(points) < 3:
        return None, math.nan
    points = np.asarray(points)
    # Margem: menor distancia de um cone a linha central.
    distance = np.min(np.linalg.norm(cones[:, None, :] - points[None, :, :], axis=2), axis=1)
    return points, float(np.min(distance))


def path_errors(points, near_weight=.5, max_start=.3):
    """(e, psi, kappa, dkappa) do caminho no eixo dianteiro, por um polinomio cubico
    ponderado pela proximidade (cones perto mandam): o que o Stanley recebe.

    So se o tracado chega ao eixo (comeca antes de max_start m): extrapolar um
    cubico de 1,5 m a frente ate o eixo da rumos absurdos; nesse caso fica o
    estimador do corredor, cujo modelo (clotoide) e feito para isso."""
    x, y = points[:, 0], points[:, 1]
    if np.ptp(x) < .5 or len(x) < 4 or x.min() > max_start:
        return None
    weights = 1./(near_weight + np.maximum(x, 0.))**2
    degree = 3 if np.ptp(x) > 1.2 and len(x) >= 6 else 2
    coefficients = np.polyfit(x, y, degree, w=np.sqrt(weights))
    c = coefficients[::-1]  # c0 + c1 x + c2 x^2 + c3 x^3
    c = np.r_[c, np.zeros(4 - len(c))]
    slope = c[1]
    scale = math.sqrt(1. + slope*slope)
    e = c[0]/scale
    psi = math.atan(slope)
    kappa = 2*c[2]/scale**3
    dkappa = 6*c[3]/scale**4
    return float(e), float(psi), float(kappa), float(dkappa)
