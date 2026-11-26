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

from .corridor import offsets

ACTIONS = ("pare", "ignorar")
STILL_SPEED = .05      # m/s: abaixo disto o carro esta parado (conta hold_s)
STILL_TIMEOUT_S = 3.   # s: sem chegar a STILL_SPEED, conta hold_s depois disto
APPROACH_CRAWL_S = .6  # s em approach_speed no fim do perfil de aproximacao
DISTINCT_SIGN_M = 1.0  # tolerancia da localizacao de leituras da mesma placa
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
        self._detect = self._detector(dictionary)
        self._detect_small = self._detector(dictionary, small=True)

    @staticmethod
    def _detector(dictionary, small=False):
        aruco = cv2.aruco
        parameters = (aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters")
                      else aruco.DetectorParameters_create())
        if small:
            parameters.minMarkerPerimeterRate = .018
            parameters.adaptiveThreshWinSizeMax = 19
            parameters.adaptiveThreshWinSizeStep = 4
            parameters.polygonalApproxAccuracyRate = .06
            parameters.perspectiveRemovePixelPerCell = 8
        if hasattr(cv2.aruco, "ArucoDetector"):
            return aruco.ArucoDetector(aruco.getPredefinedDictionary(dictionary), parameters).detectMarkers
        table = aruco.Dictionary_get(dictionary)  # OpenCV < 4.7
        return lambda gray: aruco.detectMarkers(gray, table, parameters=parameters)

    def detect(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        markers = self._markers(corners, ids)
        if not any(marker.marker_id == 1 for marker in markers):
            # O PARE dos cones aparece na metade inferior. Amplia so essa regiao,
            # com a mesma decodificacao do dicionario e sem afrouxar a correcao de bits.
            top = int(gray.shape[0] * .62)
            enlarged = cv2.resize(gray[top:], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = self._detect_small(enlarged)
            markers.extend(marker for marker in self._markers(corners, ids, scale=2, top=top)
                           if marker.marker_id == 1)
        return tuple(markers)

    @staticmethod
    def _markers(corners, ids, scale=1, top=0):
        markers = []
        if ids is not None:
            for quad, marker_id in zip(corners, ids.ravel()):
                quad = (np.asarray(quad, float).reshape(4, 2) + .5) / scale - .5 + (0, top)
                side = (np.linalg.norm(quad[2] - quad[1]) + np.linalg.norm(quad[3] - quad[0])) / 2
                cx, cy = quad.mean(axis=0)
                markers.append(Marker(int(marker_id), float(cx), float(cy), float(side)))
        return markers


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
    cone_position: tuple | None = None  # posicao no referencial atual do eixo dianteiro


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

    def _world_position(self, marker, capture_pose):
        if capture_pose is None or not (math.isfinite(marker.x) and math.isfinite(marker.y)):
            return None
        fx, fy, th = self._front(capture_pose)
        c, s = math.cos(th), math.sin(th)
        return np.array([fx + c*marker.x - s*marker.y, fy + s*marker.x + c*marker.y])

    def _remember(self, marker, world):
        """Combina leituras da mesma placa; de perto a localizacao pesa mais."""
        if world is None:
            return
        weight = 1./max(math.hypot(marker.x, marker.y), .5)**2
        if self.sign_world is None:
            self.sign_world, self.sign_weight = world, weight
        elif np.linalg.norm(world - self.sign_world) < DISTINCT_SIGN_M:
            self.sign_world = (self.sign_weight*self.sign_world + weight*world)/(self.sign_weight + weight)
            self.sign_weight += weight

    def _relative_position(self, pose):
        if self.sign_world is None or pose is None:
            return None
        fx, fy, th = self._front(pose)
        dx, dy = self.sign_world[0] - fx, self.sign_world[1] - fy
        c, s = math.cos(th), math.sin(th)
        return c*dx + s*dy, -s*dx + c*dy

    def _remaining(self, position, observation, pose, pose_at, now):
        """Distancia de parada pela linha central quando a curva ja foi estimada."""
        if position is None:
            return math.nan, False
        ahead, lateral = position
        corridor = getattr(observation, "corridor", None)
        if (corridor is not None and corridor.valid and corridor.state_vector is not None
                and pose_at is not None and 0. <= now - observation.timestamp <= .5):
            capture = pose_at(observation.timestamp)
            if capture is not None:
                fx, fy, th = self._front(capture)
                x, y, _ = self._front(pose)
                c, s = math.cos(th), math.sin(th)
                front = ((x - fx)*c + (y - fy)*s, -(x - fx)*s + (y - fy)*c)
                marker = self._relative_position(capture)
                cross, arc, _, _ = offsets([front, marker], corridor.state_vector)
                if abs(cross[1]) <= corridor.width_m / 2 + .30 and -.5 <= arc[1] <= 3.5:
                    ahead, lateral = float(arc[1] - arc[0]), float(cross[1])
        in_path = abs(lateral) <= self.front_half_width_m
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
            if self.seen[marker.marker_id] >= self.confirm_frames:
                confirmed.append(marker)
        return confirmed

    def _observe(self, observation, now, pose, pose_at, speed):
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
            if (capture_pose is None and pose is not None and math.isfinite(speed)
                    and now - stamp <= .12 and abs(speed)*(now - stamp) <= .06):
                # Na primeira imagem ainda pode faltar uma segunda pose para
                # interpolar. A geometria local basta com deslocamento pequeno.
                capture_pose = pose
            for marker in confirmed:
                action = self.actions.get(marker.marker_id)
                match action:
                    case "pare":
                        world = self._world_position(marker, capture_pose)
                        previous = self._relative_position(pose) if self.state == "SAINDO" else None
                        distinct = (previous is not None and previous[0] < -.30 and world is not None and marker.x > 0.
                                    and np.linalg.norm(world - self.sign_world) >= DISTINCT_SIGN_M)
                        armed = self.state == "LIVRE" and now >= self.ignore_until.get(marker.marker_id, -1)
                        if armed or distinct:
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
                            self._remember(marker, world)
                    case "ignorar":
                        pass
                    case _:
                        if marker.marker_id not in self.reported:
                            self.reported.add(marker.marker_id)
                            message = f"ArUco {marker.marker_id} visto, sem acao configurada"
        return message

    def update(self, observation, now, pose=None, pose_at=None, speed=math.nan, marker_observations=()):
        """Processa as leituras da placa e calcula o comando pela trajetoria atual."""
        if self.state == "SAINDO" and now >= self.ignore_until.get(self.marker_id, -1):
            self.state = "LIVRE"
        message = ""
        for sample in (*marker_observations, observation):
            message = self._observe(sample, now, pose, pose_at, speed) or message
        position = self._relative_position(pose) if self.state != "LIVRE" else None
        remaining, in_path = self._remaining(position, observation, pose, pose_at, now)
        if self.state == "APROXIMANDO":
            if not math.isfinite(remaining) or remaining <= self.braking_distance(speed):
                self.stopped(now)
                return SignCommand(self.state, 0., self.marker_id, message, True, remaining, in_path, position)
            return SignCommand(self.state, self.approach_limit(remaining), self.marker_id, message,
                               remaining_m=remaining, in_path=in_path, cone_position=position)
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
                               remaining_m=remaining, in_path=in_path, cone_position=position)
        if self.state == "SAINDO":
            return SignCommand(self.state, math.inf, self.marker_id, message,
                               in_path=in_path, cone_position=position)
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
