"""Estimador do corredor de cones: deslocamento, rumo, curvatura e largura filtrados.

A camera ve os dois lados juntos so entre ~1,9 e 3 m (o cone esquerdo entra
na imagem bem depois do direito) e mede a distancia mal (1 px ~ 5 cm a 2,5 m,
e 1 grau de arfagem desloca ~10 linhas). Recalcular o centro do zero a cada
quadro, com 2-3 portoes, da curvatura ruidosa e nenhum centro junto ao eixo.

Aqui o corredor e um ESTADO, levado de quadro em quadro pela odometria e
corrigido pelos cones (filtro de Kalman estendido iterado):

    x = [e, psi, kappa, dkappa, W]
    e       deslocamento do centro em relacao ao eixo dianteiro (esquerda +), m
    psi     rumo do centro no ponto mais proximo do eixo (esquerda +), rad
    kappa   curvatura do centro nesse ponto (esquerda +), 1/m
    dkappa  variacao da curvatura por metro de caminho (clotoide), 1/m^2:
            representa a entrada e a saida das curvas
    W       largura do corredor, m

Mesmos sinais do Stanley: (e, psi, kappa) sao os erros que ele recebe.

Cada cone e uma medida "sua distancia a linha central vale +W/2 (borda
esquerda) ou -W/2 (direita)". As duas cores entram no mesmo ajuste; com uma cor
so, o modelo continua dando rumo e curvatura e a largura vem da memoria. A
incerteza de cada cone vem da homografia: pequena na direcao (coluna), grande
na distancia (linha + arfagem), e o erro do modelo cresce com a distancia. Um
cone fora de 3 sigma da previsao (reflexo, porta, cone de outro trecho) e
descartado.

Estados (PLANO_MELHORIAS_CONES.md, tabela 4.4):
    AMBAS           cones dos dois lados neste quadro
    SO_EXTERNA      uma borda so, a de fora da curva
    SO_INTERNA      uma borda so, a de dentro da curva
    UMA_BORDA_RETA  uma borda so, trecho reto
    MEMORIA         nenhum cone aceito; segue o estado previsto pela odometria
    PERDIDO         sem estado, ou MEMORIA alem de memory_m / memory_s
"""
from dataclasses import dataclass, fields, replace
import math

import numpy as np


# LINHA: pontos da linha central (linha amarela), no fim para nao mudar os indices do car.csv.
# MAPA: tracado pelo mapa (SLAM + SVM, map_path.py) com o estimador sem cones suficientes.
STATES = ('AMBAS', 'SO_EXTERNA', 'SO_INTERNA', 'UMA_BORDA_RETA', 'MEMORIA', 'PERDIDO', 'LINHA', 'MAPA')
SPEED_FACTOR = {'MAPA': .7, 'LINHA': 1., 'AMBAS': 1., 'SO_EXTERNA': .8, 'SO_INTERNA': .8, 'UMA_BORDA_RETA': .85,
                'MEMORIA': 0., 'PERDIDO': 0.}
E, PSI, KAPPA, DKAPPA, WIDTH = range(5)
# Linha central amostrada de -1 a 4,5 m de caminho (a camera ve ate ~3 m).
ARC = np.arange(-1., 4.5 + 1e-9, .05)
ARC_ZERO = int(np.argmin(np.abs(ARC)))


@dataclass(frozen=True)
class CorridorSettings:
    # A largura da pista muda de um dia para outro: track_width_m e so o palpite
    # inicial (cone_settings.track_width_cm). Com as duas bordas, W e medido; a
    # largura aprendida (media dos quadros AMBAS) vale dai em diante, inclusive
    # depois de uma perda, e segura W quando so uma borda aparece.
    track_width_m: float = .9      # palpite inicial, m
    min_width_m: float = .5        # limites fisicos de W
    max_width_m: float = 3.
    init_width_sigma: float = .6   # incerteza do palpite ainda nao confirmado, m (pista de 1 a 2,5 m)
    width_prior_sigma: float = .2  # atracao fraca para a largura aprendida, m (por quadro)
    # m/sqrt(m): a largura muda ao longo da pista (24/11: 1,1 a 1,8 m, ate 30 cm
    # numa mesma volta). Com .05 W demorava a acompanhar e o centro ia junto:
    # tools/avaliar_tracado.py (erro do CENTRO previsto, que nao depende de W),
    # 38 voltas: mediana 4,5 -> 4,2 cm, p90 26 -> 18 cm; numa volta o centro
    # caia na fileira errada (118 cm) e passou a 4 cm.
    width_noise: float = .15
    width_learn_rate: float = .2   # peso de cada quadro AMBAS confiavel na largura aprendida
    # Largura MEDIDA na pista do dia (m; 0 = desconhecida). Com ela o estimador ja
    # comeca sabendo W (nao mede pelo primeiro par de cones, que em 24/11 deu 1,10
    # numa volta e 1,88 na seguinte) e a aprendida fica a +-known_width_tolerance.
    known_width_m: float = 0.
    known_width_tolerance: float = .15
    max_range_m: float = 3.        # cones mais distantes nao entram
    pixel_sigma: float = 2.        # erro da base do cone, px (na resolucao calibrada)
    row_sigma: float = 4.          # px extras na vertical: arfagem do carro (~0,4 grau)
    model_sigma: float = .05       # cones fora da curva ideal (posicionamento), m
    # Hierarquia dos cones: cada um pesa no ajuste pelo inverso da sua variancia,
    # que cresce com a distancia (homografia, quadratica, + este termo linear).
    # Desvio de um cone de borda numa reta (calibracao de 17/11): 0,8 m 13 cm,
    # 1 m 15, 1,5 m 20, 2 m 25, 2,5 m 31, 3 m 36 -> peso relativo ao de 1 m: 1,34 /
    # 1 / 0,56 / 0,36 / 0,25 / 0,18. Era 0,05/m (peso 0,24 a 3 m). Pista virtual
    # (cones a 1 m, 7 q/s, ruido e erro de calibracao nos longe): 0,10/m tirou os
    # toques (2 -> 0) e o pior erro (59 -> 35 cm); 0,2/m ou mais piora: com cones
    # esparsos os longe ainda sao os que mostram a curva chegando.
    # Replay dos 13 ensaios de 24/11 (previsao dos cones vistos ate 1 s depois):
    # model_growth 0,05 e kappa/dkappa_noise 0,2/1,0 erravam 6,8 cm contra 7,7, mas
    # pioravam objetos falsos ao lado (15 cm contra < 12), a partida com cores
    # trocadas e a partida so com cones longe. Ficaram os valores robustos.
    model_growth: float = .10      # m por m de caminho: o modelo vale perto
    gate_sigma: float = 3.         # cone a mais de N sigma da previsao e descartado
    # Faixa de busca: depois do ajuste com todos os cones aceitos, cone a mais de
    # band_m (+ band_growth por m alem de 1 m) da borda AJUSTADA sai e o quadro e
    # refeito sem ele. O portao acima e largo longe (curva ainda desconhecida:
    # 3 sigma passa de 1,5 m a 2,5 m); uma curva de verdade move todos os cones
    # juntos e o ajuste acompanha, um falso isolado (mochila, roupa) fica fora.
    # Pista virtual, 15 objetos fixos 20-80 cm fora da pista com cor ao acaso:
    # sem faixa 11/12 voltas, 11 cm RMS (pior 41), 2 toques; com 0,2 + 0,1/m
    # 12/12, 4,4 cm (pior 6,6), sem toque, igual a pista sem objetos. Nos 11
    # ensaios reais de 10/11 tira 22 de ~11 mil cones, sem perder quadros validos.
    band_m: float = .2             # 0 desliga
    band_growth: float = .1        # m de faixa por m alem de 1 m (0,3 m a 2 m; 0,4 m a 3 m)
    kappa_noise: float = .1        # (1/m)^2 por m andado
    dkappa_noise: float = .5       # (1/m^2)^2 por m andado: curvas comecam de repente
    init_kappa_sigma: float = .4  # prior da curvatura ao (re)comecar, 1/m
    # Prior do deslocamento e do rumo no eixo: .6 m / .6 rad ao recomecar depois
    # de uma perda (o carro pode estar em qualquer lugar). Na PRIMEIRA vez da
    # corrida o carro foi posto a mao no corredor, alinhado: start_sigma_e/psi.
    # Com cones so a 1,2-2,6 m, o eixo e pouco observavel e o prior decide quanto
    # tempo a incerteza leva para cair (partida, README_cones.md). Replay dos 11
    # ensaios de 10/11: 0,25 m / 0,3 rad tirava quase toda a espera, mas na pista
    # virtual com objetos falsos numa curva fechada o erro subia de 5 para 7 cm;
    # 0,5 passa em todos os testes de robustez e encurta a espera nos piores.
    start_sigma_e: float = .5     # m
    start_sigma_psi: float = .5    # rad
    init_dkappa_sigma: float = .4  # prior da variacao da curvatura, 1/m^2
    time_noise: float = 1.         # escala do ruido por segundo (mesmo parado o modelo erra)
    max_curvature: float = 2.      # 1/m
    max_dkappa: float = 3.         # 1/m^2
    memory_m: float = .8           # distancia maxima andada so com a previsao
    # Cones para (re)comecar quando a largura da pista ja foi aprendida nesta
    # corrida (sem ela, sempre 3). Ver _initialize.
    restart_cones: float = 2.
    # Tempo maximo sem cones aceitos. A memoria tambem expira por distancia.
    memory_s: float = 2.5
    # Liberacao do controle: incerteza lateral da linha central a check_distance_m
    # a frente (onde o carro estara em ~1 s), nao no eixo. Com so cones distantes
    # (partida), e e psi no eixo sao incertos mas correlacionados: o caminho que
    # o carro vai percorrer e conhecido; ele parte devagar ate os cones chegarem.
    check_distance_m: float = 1.
    max_sigma_lateral: float = .2  # m, a check_distance_m
    max_sigma_e: float = .5        # teto no eixo, m
    max_sigma_psi: float = .5      # rad (e e psi no eixo sao correlacionados; vale o lateral)
    # Velocidade de recuperacao (CONE_RECOVERY_SPEED) so quando o caminho A FRENTE
    # e incerto: sigma lateral a check_distance_m > slow_sigma_lateral, ou o eixo
    # muito incerto (slow_sigma_e). Antes valia sigma no eixo > 0,1 m: na partida,
    # com os cones a 1,5 m, isso segurava o carro a 0,35 m/s por 1,6-4,1 s nos 11
    # ensaios de 10/11 (motor na zona morta do ESC: o "engasgo"), e em 4 deles
    # tambem em 70-100% do resto da volta, com o caminho a frente bem medido.
    # Com o novo criterio, a partida libera em 0-2,3 s (replay_corredor dos mesmos).
    slow_sigma_lateral: float = .1  # m, a check_distance_m
    slow_sigma_e: float = .3       # m, no eixo
    # Eixo ainda incerto (sigma > near_sigma_e) mas caminho a frente conhecido: o
    # pure pursuit mira pelo menos check_distance_m a frente (alvo bem medido),
    # nao 0,4 m (alvo pela incerteza do eixo, direcao batendo no fim de curso).
    near_sigma_e: float = .1       # m
    straight_curvature: float = .15  # 1/m: abaixo disso uma borda so e "reta"
    pose_offset_m: float = .36     # do ponto da pose (eixo traseiro) ao eixo dianteiro
    # Teto absoluto (m/s) com uma borda so (SO_EXTERNA/SO_INTERNA/UMA_BORDA_RETA):
    # com velocidade alta nas retas, uma fracao da vref ainda seria rapido demais.
    one_border_speed: float = .9

    @classmethod
    def from_value(cls, value, **defaults):
        """True / dict de ajustes / CorridorSettings -> CorridorSettings."""
        if isinstance(value, cls):
            return value
        settings = cls(**defaults)
        if value is True or value is None:
            return settings
        if not isinstance(value, dict):
            raise ValueError('cone_estimator deve ser True, False ou um dicionario de ajustes.')
        known = {f.name for f in fields(cls)}
        if set(value) - known:
            raise ValueError('Ajuste do estimador desconhecido: ' + str(sorted(set(value) - known)))
        for name, item in value.items():
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(item) or item < 0):
                raise ValueError(f'cone_estimator.{name} deve ser numero finito nao negativo.')
        return replace(settings, **{k: float(v) for k, v in value.items()})


@dataclass(frozen=True)
class CorridorEstimate:
    valid: bool
    state: str
    errors: tuple | None = None    # (e, psi, kappa) para o Stanley
    width_m: float = math.nan
    sigma: tuple = (math.nan,)*5   # desvios de (e, psi, kappa, dkappa, W)
    used: tuple = (0, 0)           # cones aceitos (esquerda, direita)
    rejected: int = 0
    blind_m: float = 0.            # andado desde o ultimo cone aceito
    reason: str = ''
    state_vector: tuple | None = None  # (e, psi, kappa, dkappa, W) para desenhar
    slow: bool = False             # incerteza alta: limitar a velocidade
    near_uncertain: bool = False   # eixo incerto (sigma e > near_sigma_e): mirar longe
    learned_width_m: float = math.nan  # largura aprendida nesta corrida (nan: ainda nao medida)
    remembered_state: str | None = None  # origem da trajetoria em MEMORIA

    @property
    def reference_state(self):
        return self.remembered_state or self.state

    @property
    def speed_factor(self):
        return SPEED_FACTOR[self.reference_state]

    def debug(self):
        r = lambda v, d=3: None if v is None or not math.isfinite(v) else round(float(v), d)
        vector = self.state_vector
        return {'state': self.state, 'valid': bool(self.valid),
                'errors': None if self.errors is None else [r(v, 4) for v in self.errors],
                'dkappa': None if vector is None else r(vector[DKAPPA]),
                'width_m': r(self.width_m), 'sigma': [r(v, 4) for v in self.sigma],
                'learned_width_m': r(self.learned_width_m),
                'used': list(self.used), 'rejected': int(self.rejected),
                'blind_m': r(self.blind_m), 'speed_factor': self.speed_factor, 'slow': bool(self.slow),
                'near_uncertain': bool(self.near_uncertain),
                'reason': self.reason}


STEP = ARC[1] - ARC[0]
PROBE = 1e-4  # passo das derivadas numericas
# Estado base e um passo em cada componente da linha (e, psi, kappa, dkappa):
# as 5 linhas centrais saem de uma vez (no Pi, 5 chamadas separadas custavam
# ~5x mais pelo overhead do numpy em vetores pequenos).
PROBES = np.vstack([np.zeros(5), PROBE*np.eye(5)[:4]])


def centerlines(states, max_curvature=2.):
    """Linhas centrais de varios estados (S, 5): (pontos (S, M, 2), rumo (S, M))."""
    states = np.atleast_2d(np.asarray(states, float))
    e, psi, kappa, dkappa = (states[:, k:k+1] for k in range(4))
    curvature = np.clip(kappa + dkappa*ARC, -max_curvature, max_curvature)
    # Integral trapezoidal a partir de s = 0 (ancora: ponto mais proximo do eixo).
    theta = np.concatenate((np.zeros((len(states), 1)),
                            np.cumsum((curvature[:, 1:] + curvature[:, :-1])*STEP/2, axis=1)), axis=1)
    theta = psi + theta - theta[:, ARC_ZERO:ARC_ZERO+1]
    c, s = np.cos(theta), np.sin(theta)
    zero = np.zeros((len(states), 1))
    x = np.concatenate((zero, np.cumsum((c[:, 1:] + c[:, :-1])*STEP/2, axis=1)), axis=1)
    y = np.concatenate((zero, np.cumsum((s[:, 1:] + s[:, :-1])*STEP/2, axis=1)), axis=1)
    x = x - x[:, ARC_ZERO:ARC_ZERO+1] - e*np.sin(psi)
    y = y - y[:, ARC_ZERO:ARC_ZERO+1] + e*np.cos(psi)
    return np.stack((x, y), axis=-1), theta


def centerline(state, max_curvature=2.):
    """(pontos Nx2, rumo N) da linha central amostrada em ARC, referencial do carro."""
    line, theta = centerlines(np.asarray(state, float)[None, :], max_curvature)
    return line[0], theta[0]


def offsets_many(points, states, index, max_curvature=2., lines=None):
    """Distancias (S, N) dos pontos a cada linha central, com a amostra `index` fixa.
    lines: (pontos, rumos) de centerlines(states) ja calculados."""
    line, theta = centerlines(states, max_curvature) if lines is None else lines
    delta = points[None, :, :] - line[:, index]
    angle = theta[:, index]
    return delta[..., 1]*np.cos(angle) - delta[..., 0]*np.sin(angle)


def offsets(points, state, index=None, max_curvature=2., line=None):
    """(distancia a linha central, esquerda +; avanco no caminho; indice; normal) de cada ponto.

    index: amostra mais proxima de cada ponto, para manter fixa nas derivadas
    numericas (trocar de amostra no meio da derivada a deixaria aos saltos).
    line: (pontos, rumo) de centerline(state) ja calculados.
    """
    points = np.asarray(points, float).reshape(-1, 2)
    line, theta = centerline(state, max_curvature) if line is None else line
    if index is None:
        diff = points[:, None, :] - line[None, :, :]
        index = np.argmin(np.einsum('nmk,nmk->nm', diff, diff), axis=1)
    delta = points - line[index]
    # column_stack em vez de np.c_ (lento: ~10 us por chamada, milhares por volta).
    tangent = np.column_stack((np.cos(theta[index]), np.sin(theta[index])))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    d = np.einsum('nk,nk->n', delta, normal)
    along = ARC[index] + np.einsum('nk,nk->n', delta, tangent)
    return d, along, index, normal


def sample_centerline(state, arc_lengths, offset=0., max_curvature=2.):
    """Pontos da linha central deslocada `offset` (esquerda +) nos avancos dados, Nx2."""
    line, theta = centerline(state, max_curvature)
    s = np.asarray(arc_lengths, float)
    x, y, t = (np.interp(s, ARC, v) for v in (line[:, 0], line[:, 1], theta))
    return np.column_stack((x - offset*np.sin(t), y + offset*np.cos(t)))


def move_states(states, dx, dy, dth, max_curvature=2.):
    """Estados (S, 5) vistos do eixo dianteiro deslocado (dx, dy) e girado dth.

    Leva a clotoide inteira: o novo ponto mais proximo do eixo, o rumo ali e a
    curvatura (kappa + dkappa*s). Usado na previsao do estimador e na
    compensacao do atraso da imagem no seguidor.
    """
    states = np.atleast_2d(np.asarray(states, float))
    line, theta = centerlines(states, max_curvature)
    diff = np.array([dx, dy]) - line
    index = np.argmin(np.einsum('smk,smk->sm', diff, diff), axis=1)
    rows = np.arange(len(states))
    delta, angle = diff[rows, index], theta[rows, index]
    c, s_ = np.cos(angle), np.sin(angle)
    d = delta[:, 1]*c - delta[:, 0]*s_
    s = ARC[index] + delta[:, 0]*c + delta[:, 1]*s_
    kappa, dkappa = states[:, KAPPA], states[:, DKAPPA]
    moved = np.array(states, float)
    moved[:, E] = -d
    psi = states[:, PSI] + kappa*s + dkappa*s*s/2 - dth
    moved[:, PSI] = np.arctan2(np.sin(psi), np.cos(psi))
    moved[:, KAPPA] = np.clip(kappa + dkappa*s, -max_curvature, max_curvature)
    return moved


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CorridorEstimator:
    def __init__(self, settings=None, ground_from_pixel=None, pixel_scale=1.):
        """ground_from_pixel: homografia 3x3 pixel -> chao (m) da imagem processada,
        usada para a incerteza de cada cone; pixel_scale: px da imagem processada
        por px da resolucao calibrada. None: incerteza so pelo modelo."""
        self.settings = settings or CorridorSettings()
        self.homography = None if ground_from_pixel is None else np.asarray(ground_from_pixel, float)
        self.pixel_scale = float(pixel_scale)
        self.learned_width = None  # sobrevive a reset(): a pista nao muda ao perder os cones
        self.started = False       # ja inicializou nesta corrida (sobrevive a reset())
        self.reset()
        self.forget_width()

    def forget_width(self):
        """Largura aprendida de volta ao ponto de partida: a medida, se houver."""
        known = self.settings.known_width_m
        self.learned_width = float(known) if known and known > 0 else None

    @property
    def nominal_width(self):
        return self.settings.track_width_m if self.learned_width is None else self.learned_width

    def _learn_width(self, sigma_width):
        """Quadro com as duas bordas e W bem medido: atualiza a largura aprendida."""
        if sigma_width > .1:
            return
        width = float(self.x[WIDTH])
        if self.learned_width is None:
            self.learned_width = width
        else:
            self.learned_width += self.settings.width_learn_rate*(width - self.learned_width)
        known = self.settings.known_width_m
        if known and known > 0:
            margin = self.settings.known_width_tolerance*known
            self.learned_width = float(np.clip(self.learned_width, known - margin, known + margin))

    def reset(self):
        self.x = None
        self.P = None
        self.last_time = None
        self.last_update_time = None
        self.last_visual_estimate = None
        self.blind_m = 0.
        self.bad_frames = 0
        self.used_center = 0
        self.band_rejected = 0  # cones tirados pela faixa no ultimo quadro (ja somados em rejected)

    def relax(self, sigma_e=.3, sigma_psi=.2):
        """Abre a incerteza do deslocamento e do rumo sem perder o corredor.

        Usado quando as cores trocam de lado com o corredor ja estimado (ilha de
        rotatoria de outra cor): a geometria continua a mesma, so a cor de cada
        borda mudou; os proximos cones recentralizam o estado sem o carro parar."""
        if self.P is None:
            return
        for index, sigma in ((E, sigma_e), (PSI, sigma_psi)):
            self.P[index, :] *= 0.
            self.P[:, index] *= 0.
            self.P[index, index] = sigma*sigma

    def set_homography(self, ground_from_pixel, pixel_scale=1.):
        self.homography = np.asarray(ground_from_pixel, float)
        self.pixel_scale = float(pixel_scale)

    # ------------------------------------------------------------------ previsao
    def _move(self, state, dx, dy, dth):
        """Estado visto do eixo dianteiro deslocado (dx, dy) e girado dth."""
        return self._move_many(np.asarray(state, float)[None, :], dx, dy, dth)[0]

    def _move_many(self, states, dx, dy, dth):
        """_move de varios estados (S, 5) de uma vez (estado e passos do jacobiano)."""
        return move_states(states, dx, dy, dth, self.settings.max_curvature)

    def predict(self, motion, pose_frame=True, dt=0.):
        """motion: (frente, esquerda, yaw) desde o quadro anterior, ou None.

        pose_frame: o deslocamento e do ponto da pose (eixo traseiro, como o
        PoseHistory); converte para o eixo dianteiro, origem dos cones.
        """
        if self.x is None:
            return
        cfg = self.settings
        dt = max(0., float(dt))
        timed = cfg.time_noise*dt*np.array([.03**2, .05**2, 0., 0., .01**2])
        if motion is None or not np.all(np.isfinite(motion)):
            # Sem odometria: estado parado, mas a incerteza cresce mais rapido.
            self.P = self.P + 4*timed + np.diag([.02**2, .03**2, .02, .1, 1e-4])
            return
        dx, dy, dth = (float(v) for v in motion)
        if pose_frame and cfg.pose_offset_m:
            dx += cfg.pose_offset_m*(math.cos(dth) - 1)
            dy += cfg.pose_offset_m*math.sin(dth)
        x0 = self.x
        moved = self._move_many(x0 + PROBES, dx, dy, dth)
        x1 = moved[0]
        jacobian = np.eye(5)
        delta = moved[1:] - x1
        delta[:, PSI] = np.arctan2(np.sin(delta[:, PSI]), np.cos(delta[:, PSI]))
        jacobian[:, :4] = (delta/PROBE).T
        ds = math.hypot(dx, dy)
        noise = np.diag([(.005 + .03*ds)**2, (.005 + .15*abs(dth) + .02*ds)**2,
                         cfg.kappa_noise*ds + 1e-5, cfg.dkappa_noise*ds + 1e-4,
                         cfg.width_noise**2*ds + 1e-6]) + np.diag(timed)
        self.x = x1
        self.P = jacobian @ self.P @ jacobian.T + noise
        self.blind_m += ds

    # --------------------------------------------------------------- correcao
    def _point_covariance(self, points):
        """Covariancia 2x2 de cada cone no chao, pela homografia (N, 2, 2)."""
        n = len(points)
        if self.homography is None or not n:
            return np.zeros((n, 2, 2))
        cfg = self.settings
        H = self.homography
        pixels = np.column_stack((points, np.ones(n))) @ np.linalg.inv(H).T
        pixels = pixels[:, :2]/pixels[:, 2:3]
        w = pixels @ H[2, :2] + H[2, 2]
        gx, gy = points[:, 0], points[:, 1]
        J = np.empty((n, 2, 2))
        J[:, 0, 0] = (H[0, 0] - gx*H[2, 0])/w
        J[:, 0, 1] = (H[0, 1] - gx*H[2, 1])/w
        J[:, 1, 0] = (H[1, 0] - gy*H[2, 0])/w
        J[:, 1, 1] = (H[1, 1] - gy*H[2, 1])/w
        su = (cfg.pixel_sigma*self.pixel_scale)**2
        sv = su + (cfg.row_sigma*self.pixel_scale)**2
        return J @ np.diag([su, sv]) @ J.transpose(0, 2, 1)

    def _measure(self, state, points, sides, covariance):
        """Residuos h (medida esperada 0), jacobiano H (N, 5), variancia R e avanco."""
        cfg = self.settings
        # A linha e as 4 sondas das derivadas numa so chamada (antes 2: no Pi o
        # custo e o overhead do numpy por chamada; ~5 medidas por quadro).
        lines, thetas = centerlines(np.asarray(state, float) + PROBES, cfg.max_curvature)
        d, along, index, normal = offsets(points, state, max_curvature=cfg.max_curvature,
                                          line=(lines[0], thetas[0]))
        h = d - sides*state[WIDTH]/2
        H = np.zeros((len(points), 5))
        probes = offsets_many(points, None, index, cfg.max_curvature, lines=(lines[1:], thetas[1:]))
        H[:, :4] = ((probes - d)/PROBE).T
        H[:, WIDTH] = -sides/2
        # Derivada da distancia em relacao ao ponto: a normal da linha central.
        R = np.einsum('ni,nij,nj->n', normal, covariance, normal)
        R = R + (cfg.model_sigma + cfg.model_growth*np.abs(along))**2
        return h, H, R, along

    def _solve(self, x0, P0, points, sides, covariance, iterations=4, first=None):
        """EKF iterado (Gauss-Newton com a previsao como prior).

        first: (h, H, R) de _measure(x0, points) ja calculados (o portao mede no
        mesmo estado: cada linha so depende do seu ponto)."""
        x = x0.copy()
        K = H = R = None
        for iteration in range(iterations):
            if iteration == 0 and first is not None:
                h, H, R = first
            else:
                h, H, R, _ = self._measure(x, points, sides, covariance)
            delta = x0 - x
            delta[PSI] = _wrap(delta[PSI])
            innovation = -h - H @ delta
            S = H @ P0 @ H.T + np.diag(R)
            K = np.linalg.solve(S, H @ P0).T
            step = x0 + K @ innovation - x
            x = x + step
            x[PSI] = _wrap(x[PSI])
            if np.max(np.abs(step[:4])) < 1e-4:
                break
        I_KH = np.eye(5) - K @ H
        P = I_KH @ P0 @ I_KH.T + K @ np.diag(R) @ K.T
        return x, P

    def _band(self, along):
        """Meia largura da faixa de busca em torno da borda ajustada, m."""
        return self.settings.band_m + self.settings.band_growth*np.maximum(np.asarray(along) - 1., 0.)

    def _usable(self, along):
        return (along > -.6) & (along < ARC[-1] - .2)

    def _constrain(self, x, P):
        cfg = self.settings
        if cfg.width_prior_sigma > 0 and self.learned_width is not None:
            # Atracao fraca para a largura aprendida: com uma cor so por muito
            # tempo, W nao deriva. Antes de medir a pista, o palpite nao puxa.
            gain = P[:, WIDTH]/(P[WIDTH, WIDTH] + cfg.width_prior_sigma**2)
            x = x + gain*(self.learned_width - x[WIDTH])
            P = P - np.outer(gain, P[WIDTH, :])
        x[WIDTH] = float(np.clip(x[WIDTH], cfg.min_width_m, max(cfg.min_width_m, cfg.max_width_m)))
        x[KAPPA] = float(np.clip(x[KAPPA], -cfg.max_curvature, cfg.max_curvature))
        x[DKAPPA] = float(np.clip(x[DKAPPA], -cfg.max_dkappa, cfg.max_dkappa))
        return x, (P + P.T)/2

    def _initialize(self, points, sides, covariance):
        cfg = self.settings
        x0 = np.array([0., 0., 0., 0., self.nominal_width])
        left, right = points[sides > 0], points[sides < 0]
        if len(left) and len(right) and not self.started:
            # So na PRIMEIRA partida (carro posto a mao, sem objetos estranhos a
            # frente; num recomeco um falso pode ser o cone mais perto).
            # Ponto de partida pelos cones dos dois lados (o mais perto de cada):
            # centro no meio e, sem largura aprendida, a largura entre eles. Com o
            # palpite fixo (0,9 m) numa pista de 1,9 m e 2 cones, o ajuste girava o
            # corredor 17 graus em vez de alarga-lo (partida de 19/11).
            # O par mais parecido com um portao (mesma distancia a frente), perto.
            cost = np.abs(left[:, None, 0] - right[None, :, 0]) + .2*(left[:, None, 0] + right[None, :, 0])
            i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
            near_left, near_right = left[i], right[j]
            x0[E] = float(np.clip((near_left[1] + near_right[1])/2, -1., 1.))
            if self.learned_width is None:
                x0[WIDTH] = float(np.clip(near_left[1] - near_right[1], cfg.min_width_m, cfg.max_width_m))
        width_sigma = cfg.init_width_sigma if self.learned_width is None else (cfg.width_prior_sigma or .2)
        sigma_e, sigma_psi = (.6, .6) if self.started else (cfg.start_sigma_e, cfg.start_sigma_psi)
        P0 = np.diag([sigma_e**2, sigma_psi**2, cfg.init_kappa_sigma**2, cfg.init_dkappa_sigma**2,
                      width_sigma**2])
        keep = np.ones(len(points), bool)
        x, P = x0, P0
        # Com a largura da pista ja aprendida nesta corrida, 2 cones bastam para
        # recomecar (a largura vem da memoria; o portao de incerteza a frente
        # ainda decide se vale). Antes exigia 3: nos ensaios de 17/11 o carro via
        # 0-2 cones por cor e ficou ate 9 s parado com 2 a vista.
        # Um cone de CADA lado tambem basta: o par ja da o centro e a largura (em
        # 19/11, pista de ~1,9 m, o carro ficou parado vendo 1 verde e 1 vermelho).
        both_sides = bool(np.any(sides > 0) and np.any(sides < 0))
        needed = int(cfg.restart_cones) if (self.learned_width is not None or both_sides) else 3
        for _ in range(3):
            if np.count_nonzero(keep) < needed:
                return None
            x, P = self._solve(x0, P0, points[keep], sides[keep], covariance[keep], iterations=12)
            h, _, R, along = self._measure(x, points, sides, covariance)
            new_keep = (np.abs(h) <= cfg.gate_sigma*np.sqrt(R)) & self._usable(along)
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        left, right = int(np.sum(keep & (sides > 0))), int(np.sum(keep & (sides < 0)))
        center = int(np.sum(keep & (sides == 0)))
        if center:
            if center < 3:
                return None
        else:
            if left + right < needed or (min(left, right) == 0 and max(left, right) < needed):
                return None
        x, P = self._constrain(x, P)
        self.started = True
        return x, P, keep

    def update(self, left, right, center=None):
        """Corrige com os cones deste quadro (Nx2 no chao, referencial do eixo dianteiro).

        center: pontos da propria linha central (linha amarela): distancia 0.
        Devolve (cones aceitos esquerda, direita, descartados); os da linha
        aceitos ficam em self.used_center.
        """
        cfg = self.settings
        left = np.asarray(left, float).reshape(-1, 2)
        right = np.asarray(right, float).reshape(-1, 2)
        center = np.empty((0, 2)) if center is None else np.asarray(center, float).reshape(-1, 2)
        points = np.vstack([left, right, center])
        sides = np.concatenate((np.ones(len(left)), -np.ones(len(right)), np.zeros(len(center))))
        self.used_center = 0
        self.band_rejected = 0
        near = (points[:, 0] > 0) & (points[:, 0] <= cfg.max_range_m) & np.all(np.isfinite(points), axis=1)
        points, sides = points[near], sides[near]
        if not len(points):
            return 0, 0, 0
        covariance = self._point_covariance(points)
        if self.x is None:
            started = self._initialize(points, sides, covariance)
            if started is None:
                return 0, 0, len(points)
            self.x, self.P, keep = started
        else:
            h, H, R, along = self._measure(self.x, points, sides, covariance)
            S = np.einsum('ni,ij,nj->n', H, self.P, H) + R
            keep = (h*h <= cfg.gate_sigma**2*S) & self._usable(along)
            rejected = len(points) - int(np.count_nonzero(keep))
            # Previsao que perdeu o corredor descarta tudo: com rejeicao
            # dominante em dois quadros seguidos, recomeca so pelos cones atuais.
            self.bad_frames = self.bad_frames + 1 if rejected >= 3 and rejected > np.count_nonzero(keep) else 0
            if self.bad_frames >= 2:
                started = self._initialize(points, sides, covariance)
                if started is not None:
                    self.x, self.P, keep = started
                    self.bad_frames = 0
            elif np.any(keep):
                x, P = self._solve(self.x, self.P, points[keep], sides[keep], covariance[keep],
                                   first=(h[keep], H[keep], R[keep]))
                if cfg.band_m > 0 and np.count_nonzero(keep) > 3:
                    h, _, _, along = self._measure(x, points, sides, covariance)
                    outside = keep & (np.abs(h) > self._band(along))
                    if np.any(outside) and np.count_nonzero(keep & ~outside) >= 3:
                        keep = keep & ~outside
                        self.band_rejected += int(np.count_nonzero(outside))
                        x, P = self._solve(self.x, self.P, points[keep], sides[keep], covariance[keep])
                self.x, self.P = self._constrain(x, P)
        left_used = int(np.sum(keep & (sides > 0)))
        right_used = int(np.sum(keep & (sides < 0)))
        self.used_center = int(np.sum(keep & (sides == 0)))
        return left_used, right_used, len(points) - left_used - right_used - self.used_center

    # ------------------------------------------------------------------ passo
    def step(self, timestamp, left, right, motion=None, pose_frame=True, center=None):
        """Previsao pela odometria + correcao pelos cones (ou pela linha central,
        center); devolve CorridorEstimate."""
        cfg = self.settings
        if self.last_time is not None and timestamp <= self.last_time:
            self.reset()
        if self.last_time is not None:
            self.predict(motion, pose_frame, timestamp - self.last_time)
        self.last_time = timestamp
        used_left, used_right, rejected = self.update(left, right, center)
        used_center = self.used_center
        if used_left + used_right + used_center:
            self.last_update_time = timestamp
            self.blind_m = 0.
        if self.x is None:
            return CorridorEstimate(False, 'PERDIDO', rejected=rejected,
                                    reason='Estimador aguardando cones (>=3, ou dos dois lados)')
        sigma = tuple(float(v) for v in np.sqrt(np.maximum(np.diag(self.P), 0.)))
        lost = (self.blind_m > cfg.memory_m or self.last_update_time is None
                or timestamp - self.last_update_time > cfg.memory_s)
        if lost:
            state = 'PERDIDO'
        elif used_center and not (used_left or used_right):
            state = 'LINHA'
        elif used_left and used_right:
            state = 'AMBAS'
            self._learn_width(sigma[WIDTH])
        elif used_left or used_right:
            kappa = self.x[KAPPA]
            if abs(kappa) < cfg.straight_curvature:
                state = 'UMA_BORDA_RETA'
            else:
                # Borda esquerda numa curva a direita (kappa < 0) e a de fora.
                outer = (used_left and kappa < 0) or (used_right and kappa > 0)
                state = 'SO_EXTERNA' if outer else 'SO_INTERNA'
        else:
            state = 'MEMORIA'
        # Incerteza a frente DADA a largura: com uma borda so e a largura ainda
        # nao medida, segue a borda a meia largura do palpite (como a recuperacao
        # por uma borda do detector), devagar (slow, pelo sigma do eixo).
        lateral_sigma = self.lateral_sigma(cfg.check_distance_m, given_width=True)
        # Uma borda so e a largura nunca medida nesta corrida: o centro sai do
        # palpite de largura (lateral_sigma acima o supoe conhecido): devagar.
        width_guessed = (state in ('SO_EXTERNA', 'SO_INTERNA', 'UMA_BORDA_RETA') and self.learned_width is None
                         and sigma[WIDTH] > cfg.width_prior_sigma)
        # Em MEMORIA o carro segue o trecho ja visto; o criterio a frente vale
        # para liberar a partida e com cones, e o limite e memory_m.
        uncertain = ((state != 'MEMORIA' and lateral_sigma > cfg.max_sigma_lateral)
                     or sigma[E] > cfg.max_sigma_e or sigma[PSI] > cfg.max_sigma_psi)
        remembered = self.last_visual_estimate if state == 'MEMORIA' else None
        valid = state != 'PERDIDO' and not uncertain and (state != 'MEMORIA' or remembered is not None)
        if state == 'PERDIDO':
            reason = f'Corredor perdido: {self.blind_m:.2f} m sem cones aceitos'
        elif uncertain:
            reason = (f'Corredor incerto (sigma lateral a {cfg.check_distance_m:.1f} m={lateral_sigma:.2f} m,'
                      f' eixo={sigma[E]:.2f} m, rumo={sigma[PSI]:.2f} rad)')
        elif not valid:
            reason = 'Corredor sem trajetoria visual valida na memoria'
        else:
            reason = {'LINHA': 'Linha amarela (estimador)', 'AMBAS': 'Corredor pelas duas bordas',
                      'SO_EXTERNA': 'Corredor pela borda externa da curva',
                      'SO_INTERNA': 'Corredor pela borda interna da curva',
                      'UMA_BORDA_RETA': 'Corredor por uma borda (reta)',
                      'MEMORIA': f'Corredor pela memoria ({self.blind_m:.2f} m sem cones)'}[state]
        errors = (float(self.x[E]), float(self.x[PSI]), float(self.x[KAPPA])) if valid else None
        # Continuidade da mesma trajetoria: conserva a cautela da ultima
        # observacao e o teto por incerteza do eixo durante a propagacao.
        slow_path = remembered.slow if remembered is not None else (
            lateral_sigma > cfg.slow_sigma_lateral or width_guessed)
        estimate = CorridorEstimate(valid, state, errors, float(self.x[WIDTH]), sigma, (used_left, used_right),
                                    rejected, self.blind_m, reason, tuple(float(v) for v in self.x),
                                    slow=valid and (slow_path or sigma[E] > cfg.slow_sigma_e),
                                    near_uncertain=valid and sigma[E] > cfg.near_sigma_e,
                                    learned_width_m=math.nan if self.learned_width is None else self.learned_width,
                                    remembered_state=remembered.state if remembered is not None else None)
        if state != 'MEMORIA':
            self.last_visual_estimate = estimate if valid else None
        if state == 'PERDIDO':
            self.reset()
        return estimate

    def lateral_sigma(self, distance, given_width=False):
        """Desvio da posicao lateral da linha central `distance` m a frente (caminho).

        given_width: como se a largura fosse conhecida (covariancia condicionada a W).
        """
        if self.x is None:
            return math.inf
        P = self.P
        if given_width and P[WIDTH, WIDTH] > 1e-12:
            P = P - np.outer(P[:, WIDTH], P[WIDTH, :])/P[WIDTH, WIDTH]
        line, _ = centerlines(self.x + PROBES, self.settings.max_curvature)
        lateral = np.array([np.interp(distance, ARC, row) for row in line[:, :, 1]])
        gradient = np.zeros(5)
        gradient[:4] = (lateral[1:] - lateral[0])/PROBE
        return float(math.sqrt(max(gradient @ P @ gradient, 0.)))

    def sample(self, arc_lengths, offset=0.):
        if self.x is None:
            return None
        return sample_centerline(self.x, arc_lengths, offset, self.settings.max_curvature)
