"""Placas ArUco: deteccao (processo de visao) e acao por ID (laco de controle).

Os marcadores sao do mesmo dicionario do camera.py (DICT_5X5_100, IDs 0-99).
Cada ID tem uma acao em aruco_actions (main.py), decidida em SignController
por match/case:

    "pare"     placa de PARE (ID 1), AO LADO da pista: reduz para approach_speed
               e para com a frente do carro na altura da placa (stop_at_m). A
               posicao da placa vem da imagem (camera_pose.marker_ground: direcao
               pela homografia, distancia pelo tamanho do marcador, marker_m) e
               e lembrada pela odometria depois que ela sai da imagem (a ~1,5 m).
               Se ela estiver NA FRENTE, o ultrassom tambem para (stop_gap_m entre
               o pneu e o cone). Fica parado ate o botao "Liberar parada" da GUI
               (ou hold_s); depois ignora a placa por ignore_s.
    "ignorar"  so registra.

A placa so vale vista em confirm_frames quadros seguidos (um reflexo isolado
nao para o carro) e e esquecida se sumir por memory_s sem o carro parar.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

ACTIONS = ("pare", "ignorar")
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

    def to_tuple(self):
        r = lambda value, digits: round(value, digits) if math.isfinite(value) else math.nan
        return (self.marker_id, round(self.cx, 1), round(self.cy, 1), round(self.side_px, 1),
                r(self.x, 3), r(self.y, 3), r(self.z, 3))


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
    stop: bool = False                 # parar agora (placa ao lado alcancada)
    remaining_m: float = math.nan      # quanto falta ate a frente do carro chegar a placa


class SignController:
    def __init__(self, actions=None, confirm_frames=2, memory_s=2.5, approach_speed=.3, stop_gap_m=.08,
                 front_overhang_m=.07, reaction_s=.3, deceleration=1.75, hold_s=None, ignore_s=4.,
                 stop_at_m=0., wheelbase=.36):
        """stop_at_m: onde a FRENTE do carro para em relacao a placa (0: na altura
        dela; negativo: antes)."""
        self.actions = {}
        for key, action in dict({1: "pare"} if actions is None else actions).items():
            if action not in ACTIONS:
                raise ValueError(f"Acao ArUco desconhecida: {action} (use {', '.join(ACTIONS)})")
            self.actions[int(key)] = action
        self.confirm_frames = int(confirm_frames)
        self.memory_s = float(memory_s)
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
        self.last_seen_at = -math.inf
        self.stopped_at = None
        self.ignore_until = {}  # id -> instante ate quando ignorar (depois de parar nela)
        self.reported = set()
        self.stop_at_m = float(stop_at_m)
        self.wheelbase = float(wheelbase)
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
            total = self.sign_weight + weight
            self.sign_world = (self.sign_weight*self.sign_world + weight*world)/total
            self.sign_weight = total

    def remaining(self, pose):
        """Quanto a frente do carro ainda anda ate a altura da placa (m), ou nan."""
        if self.sign_world is None or pose is None:
            return math.nan
        fx, fy, th = self._front(pose)
        dx, dy = self.sign_world[0] - fx, self.sign_world[1] - fy
        ahead = math.cos(th)*dx + math.sin(th)*dy
        return ahead - self.front_overhang_m - self.stop_at_m

    def braking_distance(self, speed):
        speed = abs(float(speed)) if math.isfinite(speed) else self.approach_speed
        return speed*self.reaction_s + speed*speed/(2.*self.deceleration)

    def trigger_distance(self, speed):
        """Leitura do ultrassom em que comecar a parar para sobrar stop_gap_m na frente do pneu."""
        speed = abs(float(speed)) if math.isfinite(speed) else self.approach_speed
        return (self.stop_gap_m + self.front_overhang_m + speed*self.reaction_s
                + speed*speed/(2.*self.deceleration))

    def _new_frame(self, markers, now):
        ids = {m.marker_id for m in markers}
        self.seen = {i: self.seen.get(i, 0) + 1 for i in ids}
        confirmed = []
        for marker in markers:
            if self.seen[marker.marker_id] >= self.confirm_frames and now >= self.ignore_until.get(marker.marker_id, -1):
                confirmed.append(marker)
        return confirmed

    def update(self, observation, now, pose=None, pose_at=None, speed=math.nan):
        """Comando deste passo. observation: a ultima da visao (com .markers);
        pose: (x, y, th) do eixo traseiro agora; pose_at(t): a do instante da foto."""
        stamp = getattr(observation, "timestamp", None)
        message = ""
        if stamp is not None and stamp != self.last_frame:
            self.last_frame = stamp
            markers = tuple(Marker(*m) for m in (getattr(observation, "markers", ()) or ()))
            capture_pose = pose_at(stamp) if pose_at is not None else None
            for marker in self._new_frame(markers, now):
                action = self.actions.get(marker.marker_id)
                if marker.marker_id == self.marker_id:
                    self.last_seen_at = now
                    if self.state == "APROXIMANDO":
                        self._remember(marker, capture_pose)
                match action:
                    case "pare":
                        if self.state == "LIVRE":
                            self.state, self.marker_id, self.last_seen_at = "APROXIMANDO", marker.marker_id, now
                            self.sign_world, self.sign_weight = None, 0.
                            self._remember(marker, capture_pose)
                            where = (f" a {marker.x:.2f} m, {'esquerda' if marker.y > 0 else 'direita'}"
                                     f" {abs(marker.y):.2f} m, centro a {100*marker.z:.0f} cm do chao"
                                     if math.isfinite(marker.x) else " (sem posicao: so o ultrassom para)")
                            message = (f"PARE (ArUco {marker.marker_id}, lado {marker.side_px:.0f} px){where}:"
                                       f" reduzindo para {self.approach_speed:.2f} m/s")
                    case "ignorar":
                        pass
                    case _:
                        if marker.marker_id not in self.reported:
                            self.reported.add(marker.marker_id)
                            message = f"ArUco {marker.marker_id} visto, sem acao configurada"
        remaining = self.remaining(pose) if self.state == "APROXIMANDO" else math.nan
        if self.state == "APROXIMANDO" and math.isfinite(remaining) and remaining <= self.braking_distance(speed):
            # A placa ja saiu da imagem: a odometria leva o carro ate ela.
            return SignCommand(self.state, self.approach_speed, self.marker_id, message, True, remaining)
        if (self.state == "APROXIMANDO" and now - self.last_seen_at > self.memory_s
                and not (math.isfinite(remaining) and remaining > -.5)):
            message = f"PARE (ArUco {self.marker_id}) sumiu ha {self.memory_s:.1f} s sem parada: seguindo"
            self.state = "LIVRE"
        if self.state == "PARADO" and self.hold_s is not None and now - self.stopped_at >= self.hold_s:
            message = self.release(now)
        if self.state == "SAINDO" and now >= self.ignore_until.get(self.marker_id, -1):
            self.state = "LIVRE"
        if self.state == "APROXIMANDO":
            return SignCommand(self.state, self.approach_speed, self.marker_id, message, False, remaining)
        if self.state == "PARADO":
            return SignCommand(self.state, 0., self.marker_id, message)
        return SignCommand(self.state, math.inf, self.marker_id, message)

    def stopped(self, now):
        """O ultrassom parou o carro durante a aproximacao: e a parada da placa."""
        if self.state == "APROXIMANDO":
            self.state, self.stopped_at = "PARADO", now
            return True
        return False

    def release(self, now):
        """Botao Liberar parada (ou hold_s): segue e ignora esta placa por ignore_s."""
        if self.state not in ("PARADO", "APROXIMANDO"):
            return ""
        self.ignore_until[self.marker_id] = now + self.ignore_s
        self.state, self.seen = "SAINDO", {}
        return f"PARE (ArUco {self.marker_id}) liberado: ignorando esta placa por {self.ignore_s:.0f} s"
