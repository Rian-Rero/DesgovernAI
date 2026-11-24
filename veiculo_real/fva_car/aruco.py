"""Placas ArUco: deteccao (processo de visao) e acao por ID (laco de controle).

Os marcadores sao do mesmo dicionario do camera.py (DICT_5X5_100, IDs 0-99).
Cada ID tem uma acao em aruco_actions (main.py), decidida em SignController
por match/case:

    "pare"     placa de PARE (ID 1): reduz para approach_speed e para BEM PERTO
               do cone pelo ultrassom (stop_gap_m entre a frente do pneu e o
               cone), sem a distancia normal de parada do ultrassom (~0,4 m a
               0,3 m/s). Fica parado ate o botao "Liberar parada" da GUI (ou
               hold_s, se dado); depois ignora a placa por ignore_s.
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
    side_px: float   # lado medio do quadrado: cresce ao chegar perto

    def to_tuple(self):
        return (self.marker_id, round(self.cx, 1), round(self.cy, 1), round(self.side_px, 1))


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
            quad = np.asarray(quad, float).reshape(4, 2)
            side = float(np.mean(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)))
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


class SignController:
    def __init__(self, actions=None, confirm_frames=2, memory_s=2.5, approach_speed=.3, stop_gap_m=.08,
                 front_overhang_m=.07, reaction_s=.3, deceleration=1.75, hold_s=None, ignore_s=4.):
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

    def update(self, observation, now):
        """Comando deste passo; observation: a ultima observacao da visao (com .markers)."""
        stamp = getattr(observation, "timestamp", None)
        message = ""
        if stamp is not None and stamp != self.last_frame:
            self.last_frame = stamp
            markers = tuple(Marker(*m) for m in (getattr(observation, "markers", ()) or ()))
            for marker in self._new_frame(markers, now):
                action = self.actions.get(marker.marker_id)
                if marker.marker_id == self.marker_id:
                    self.last_seen_at = now
                match action:
                    case "pare":
                        if self.state == "LIVRE":
                            self.state, self.marker_id, self.last_seen_at = "APROXIMANDO", marker.marker_id, now
                            message = (f"PARE (ArUco {marker.marker_id}, lado {marker.side_px:.0f} px):"
                                       f" reduzindo para {self.approach_speed:.2f} m/s e parando perto do cone")
                    case "ignorar":
                        pass
                    case _:
                        if marker.marker_id not in self.reported:
                            self.reported.add(marker.marker_id)
                            message = f"ArUco {marker.marker_id} visto, sem acao configurada"
        if self.state == "APROXIMANDO" and now - self.last_seen_at > self.memory_s:
            message = f"PARE (ArUco {self.marker_id}) sumiu ha {self.memory_s:.1f} s sem parada: seguindo"
            self.state = "LIVRE"
        if self.state == "PARADO" and self.hold_s is not None and now - self.stopped_at >= self.hold_s:
            message = self.release(now)
        if self.state == "SAINDO" and now >= self.ignore_until.get(self.marker_id, -1):
            self.state = "LIVRE"
        if self.state == "APROXIMANDO":
            return SignCommand(self.state, self.approach_speed, self.marker_id, message)
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
