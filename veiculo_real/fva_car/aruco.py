"""Placas ArUco: deteccao (processo de visao) e acao por ID (laco de controle).

Os marcadores sao do mesmo dicionario do camera.py (DICT_5X5_100, IDs 0-99).
Cada ID tem uma acao em aruco_actions (main.py), decidida em SignController
por match/case:

    "pare"     placa de PARE (ID 1), AO LADO da pista: reduz para approach_speed
               e para com a frente do carro na altura da placa (stop_at_m). A
               posicao da placa vem da imagem (camera_pose.marker_ground: direcao
               pela homografia, distancia pelo tamanho do marcador, marker_m) e
               e lembrada pela odometria depois que ela sai da imagem (a ~1,5 m).
               Se ela estiver NA FRENTE, para antes do cone (stop_gap_m entre o
               pneu e o cone), com o ultrassom como protecao adicional. Fica parado
               hold_s segundos CONTADOS COM O CARRO JA PARADO (5 s no main.py) e
               segue sozinho; o botao "Liberar parada" da GUI libera antes (hold_s
               None: so o botao). Depois ignora a placa por ignore_s.
    "ignorar"  so registra.

A placa vale vista em confirm_frames das ultimas confirm_window buscas (antes:
seguidas; em 26/11 so 22-65% das buscas achavam a placa a vista). Confirmada
sem posicao utilizavel, pede parada imediatamente. Na aproximacao, a vref desce
num perfil suave ate a parada (approach_limit), nao em degrau.
"""
from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

ACTIONS = ("pare", "ignorar")
STILL_SPEED = .05      # m/s: abaixo disto o carro esta parado (conta hold_s)
STILL_TIMEOUT_S = 3.   # s: sem chegar a STILL_SPEED, conta hold_s depois disto
APPROACH_CRAWL_S = .6  # s em approach_speed no fim do perfil de aproximacao
STATES = ("LIVRE", "APROXIMANDO", "PARADO", "SAINDO")


@dataclass(frozen=True)
class Marker:
    """Marcador visto num quadro (pixels da imagem cheia)."""
    marker_id: int
    cx: float
    cy: float
    side_px: float   # lado VERTICAL medio (placa vista de lado: o horizontal encolhe)
    x: float = math.nan   # posicao no chao (m, eixo dianteiro, x frente, y esquerda)
    y: float = math.nan
    z: float = math.nan   # altura do centro estimada: confere o tamanho configurado
    size_m: float = math.nan  # lado real implicito pela posicao (confere marker_m)
    source: str = ""      # "altura", "tamanho" ou "chao" (tamanho dava a placa abaixo do chao)

    def to_tuple(self):
        r = lambda value, digits: round(value, digits) if math.isfinite(value) else math.nan
        return (self.marker_id, round(self.cx, 1), round(self.cy, 1), round(self.side_px, 1),
                r(self.x, 3), r(self.y, 3), r(self.z, 3), r(self.size_m, 3), self.source)


class ArucoDetector:
    """Marcadores ArUco de um quadro BGR (OpenCV novo ou antigo)."""
    def __init__(self, dictionary=None):
        dictionary = cv2.aruco.DICT_5X5_100 if dictionary is None else dictionary
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                               cv2.aruco.DetectorParameters())
            self._detect = detector.detectMarkers
        else:  # OpenCV < 4.7
            table = cv2.aruco.Dictionary_get(dictionary)
            parameters = cv2.aruco.DetectorParameters_create()
            self._detect = lambda gray: cv2.aruco.detectMarkers(gray, table, parameters=parameters)

    def detect(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        if ids is None:
            return ()
        markers = []
        for quad, marker_id in zip(corners, ids.ravel()):
            quad = np.asarray(quad, float).reshape(4, 2)  # sup-esq, sup-dir, inf-dir, inf-esq
            side = float((np.linalg.norm(quad[2] - quad[1]) + np.linalg.norm(quad[3] - quad[0]))/2)
            cx, cy = quad.mean(axis=0)
            markers.append(Marker(int(marker_id), float(cx), float(cy), side))
        return tuple(markers)


@dataclass(frozen=True)
class SignCommand:
    """O que o laco de controle faz neste passo por causa das placas."""
    state: str = "LIVRE"
    speed_limit: float = math.inf      # teto da vref (m/s)
    marker_id: int = -1
    message: str = ""
    stop: bool = False                 # entrou em PARADO neste passo
    remaining_m: float = math.nan      # quanto falta ate a posicao de parada
    in_path: bool = False


class SignController:
    def __init__(self, actions=None, confirm_frames=2, approach_speed=.3, stop_gap_m=.08,
                 front_overhang_m=.07, reaction_s=.3, deceleration=1.75, hold_s=None, ignore_s=4.,
                 stop_at_m=0., wheelbase=.36, front_half_width_m=.25, max_frame_age=1.,
                 confirm_window=None, approach_deceleration=0.):
        """stop_at_m: onde a FRENTE do carro para em relacao a placa (0: na altura
        dela; negativo: antes)."""
        self.actions = {}
        for key, action in dict({1: "pare"} if actions is None else actions).items():
            if action not in ACTIONS:
                raise ValueError(f"Acao ArUco desconhecida: {action} (use {', '.join(ACTIONS)})")
            self.actions[int(key)] = action
        self.confirm_frames = int(confirm_frames)
        # Janela de buscas (ids vistos em cada): confirma com confirm_frames nela.
        self.confirm_window = max(self.confirm_frames, int(confirm_window or self.confirm_frames))
        self.recent = deque(maxlen=self.confirm_window)
        # Desaceleracao do perfil de aproximacao (m/s2); 0: degrau para approach_speed.
        self.approach_deceleration = float(approach_deceleration)
        self.approach_speed = float(approach_speed)
        self.stop_gap_m = float(stop_gap_m)
        self.front_overhang_m = float(front_overhang_m)
        self.reaction_s = float(reaction_s)
        self.deceleration = float(deceleration)
        self.hold_s = None if hold_s is None else float(hold_s)
        self.ignore_s = float(ignore_s)
        self.state = "LIVRE"
        self.marker_id = -1
        self.seen = {}          # id -> quadros seguidos em que apareceu
        self.last_frame = None
        self.last_scan = None
        self.stopped_at = None
        self.still_since = None  # quando a velocidade zerou em PARADO (conta hold_s)
        self.ignore_until = {}  # id -> instante ate quando ignorar (depois de parar nela)
        self.reported = set()
        self.stop_at_m = float(stop_at_m)
        self.wheelbase = float(wheelbase)
        self.front_half_width_m = float(front_half_width_m)
        self.max_frame_age = float(max_frame_age)
        self.sign_world = None  # posicao da placa no referencial da odometria (media ponderada)
        self.sign_weight = 0.

    def _front(self, pose):
        x, y, th = pose
        return x + self.wheelbase*math.cos(th), y + self.wheelbase*math.sin(th), th

    def _remember(self, marker, capture_pose):
        """Placa no referencial da odometria; perto pesa mais (o tamanho erra menos)."""
        if capture_pose is None or not (math.isfinite(marker.x) and math.isfinite(marker.y)):
            return
        fx, fy, th = self._front(capture_pose)
        c, s = math.cos(th), math.sin(th)
        world = np.array([fx + c*marker.x - s*marker.y, fy + s*marker.x + c*marker.y])
        weight = 1./max(math.hypot(marker.x, marker.y), .5)**2
        if self.sign_world is None:
            self.sign_world, self.sign_weight = world, weight
        else:
            self.sign_world = (self.sign_weight*self.sign_world + weight*world)/(self.sign_weight + weight)
            self.sign_weight += weight

    def _remaining(self, pose):
        """Distancia ate o ponto de parada e se o cone intercepta a frente."""
        if self.sign_world is None or pose is None:
            return math.nan, False
        fx, fy, th = self._front(pose)
        dx, dy = self.sign_world[0] - fx, self.sign_world[1] - fy
        c, s = math.cos(th), math.sin(th)
        ahead = c*dx + s*dy
        in_path = abs(-s*dx + c*dy) <= self.front_half_width_m
        gap = max(-self.stop_at_m, self.stop_gap_m) if in_path else -self.stop_at_m
        return ahead - self.front_overhang_m - gap, in_path

    def braking_distance(self, speed):
        speed = abs(float(speed)) if math.isfinite(speed) else self.approach_speed
        return speed*self.reaction_s + speed*speed/(2.*self.deceleration)

    def approach_limit(self, remaining):
        """Teto da vref na aproximacao: perfil v^2 = vf^2 + 2*a*(espaco ate a
        distancia de parada a vf), suave em vez do degrau para approach_speed."""
        if self.approach_deceleration <= 0 or not math.isfinite(remaining):
            return self.approach_speed
        # Termina APPROACH_CRAWL_S antes da distancia de parada: o carro so rola
        # para desacelerar, chega la ja em approach_speed e para no ponto certo.
        room = max(0., remaining - self.braking_distance(self.approach_speed)
                   - APPROACH_CRAWL_S*self.approach_speed)
        return math.sqrt(self.approach_speed**2 + 2.*self.approach_deceleration*room)

    def _new_frame(self, markers, now):
        ids = {m.marker_id for m in markers}
        self.recent.append(ids)
        self.seen = {i: sum(i in scan for scan in self.recent) for i in ids}
        confirmed = []
        for marker in markers:
            if self.seen[marker.marker_id] >= self.confirm_frames and now >= self.ignore_until.get(marker.marker_id, -1):
                confirmed.append(marker)
        return confirmed

    def update(self, observation, now, pose=None, pose_at=None, speed=math.nan):
        """Comando deste passo. observation: a ultima da visao (com .markers);
        pose: (x, y, th) do eixo traseiro agora; pose_at(t): a do instante da foto."""
        stamp = getattr(observation, "marker_timestamp", getattr(observation, "timestamp", None))
        scan = getattr(observation, "marker_scan", None)
        message = ""
        if stamp is not None and 0. <= now - stamp <= self.max_frame_age and (
                self.last_frame is None or stamp > self.last_frame):
            if self.last_frame is not None and (stamp - self.last_frame > self.max_frame_age
                    or (scan is not None and self.last_scan is not None and scan != self.last_scan + 1)):
                self.seen.clear()
                self.recent.clear()
            self.last_frame = stamp
            self.last_scan = scan
            markers = tuple(Marker(*m) for m in (getattr(observation, "markers", ()) or ()))
            confirmed = self._new_frame(markers, now)
            capture_pose = pose_at(stamp) if confirmed and pose_at is not None else None
            for marker in confirmed:
                action = self.actions.get(marker.marker_id)
                match action:
                    case "pare":
                        if self.state == "LIVRE":
                            self.state, self.marker_id = "APROXIMANDO", marker.marker_id
                            self.sign_world, self.sign_weight = None, 0.
                            where = (f" a {marker.x:.2f} m, {'esquerda' if marker.y > 0 else 'direita'}"
                                     f" {abs(marker.y):.2f} m, centro a {100*marker.z:.0f} cm do chao"
                                     f" (posicao pela {marker.source or 'tamanho'}; lado implicito"
                                     f" {100*marker.size_m:.1f} cm)"
                                     if math.isfinite(marker.x) else " (sem posicao)")
                            if marker.source == "chao":
                                where += (". ATENCAO: pelo tamanho configurado a placa ficaria abaixo do chao:"
                                          " o marcador e menor que marker_m (meca o quadrado preto)")
                            message = f"PARE confirmado (ArUco {marker.marker_id}, lado {marker.side_px:.0f} px){where}"
                        if self.state == "APROXIMANDO" and marker.marker_id == self.marker_id:
                            self._remember(marker, capture_pose)
                    case "ignorar":
                        pass
                    case _:
                        if marker.marker_id not in self.reported:
                            self.reported.add(marker.marker_id)
                            message = f"ArUco {marker.marker_id} visto, sem acao configurada"
        remaining, in_path = self._remaining(pose)
        if self.state == "APROXIMANDO":
            if not math.isfinite(remaining) or remaining <= self.braking_distance(speed):
                self.stopped(now)
                return SignCommand(self.state, 0., self.marker_id, message, True, remaining, in_path)
            return SignCommand(self.state, self.approach_limit(remaining), self.marker_id, message,
                               remaining_m=remaining, in_path=in_path)
        if self.state == "PARADO" and self.hold_s is not None:
            # O tempo conta com o carro parado: PARADO comeca ainda freando. Sem
            # velocidade, conta da ordem de parar; com o encoder sem zerar
            # (ruido), dela mais STILL_TIMEOUT_S, para nao ficar preso.
            if self.still_since is None and math.isfinite(speed) and abs(speed) < STILL_SPEED:
                self.still_since = now
            if self.still_since is not None:
                start = self.still_since
            else:
                start = self.stopped_at + (STILL_TIMEOUT_S if math.isfinite(speed) else 0.)
            if now - start >= self.hold_s:
                message = self.release(now)
        if self.state == "SAINDO" and now >= self.ignore_until.get(self.marker_id, -1):
            self.state = "LIVRE"
        if self.state == "PARADO":
            return SignCommand(self.state, 0., self.marker_id, message,
                               remaining_m=remaining, in_path=in_path)
        return SignCommand(self.state, math.inf, self.marker_id, message)

    def stopped(self, now):
        """Trava a parada da placa, pela visao ou pelo ultrassom."""
        if self.state == "APROXIMANDO":
            self.state, self.stopped_at, self.still_since = "PARADO", now, None
            return True
        return False

    def release(self, now):
        """Botao Liberar parada (ou hold_s): segue e ignora esta placa por ignore_s."""
        if self.state not in ("PARADO", "APROXIMANDO"):
            return ""
        self.ignore_until[self.marker_id] = now + self.ignore_s
        self.state, self.seen = "SAINDO", {}
        self.recent.clear()
        return f"PARE (ArUco {self.marker_id}) liberado: ignorando esta placa por {self.ignore_s:.0f} s"
