"""Stanley no eixo dianteiro: x para a frente, y e delta para a esquerda."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StanleyConfig:
    gain: float = 1.5  # 1/s
    soft_speed: float = 0.2  # m/s, regulariza a partida
    max_steer: float = math.radians(20)
    max_steer_rate: float = math.radians(90)  # rad/s

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (
            self.gain, self.soft_speed, self.max_steer, self.max_steer_rate
        )) or self.max_steer >= math.pi / 2:
            raise ValueError("Parametros do Stanley devem ser positivos e finitos.")


class StanleyController:
    def __init__(self, config=None):
        self.config = config or StanleyConfig()
        self.reset()

    def reset(self):
        self.steering = 0.0
        self.heading_term = 0.0
        self.cross_track_term = 0.0
        self.raw_steering = 0.0
        self.target_steering = 0.0

    def update(self, cross_track, heading_error, velocity, dt):
        """Erros em metros/radianos; valido somente para marcha a frente.

        cross_track > 0: trajetoria a esquerda do eixo dianteiro.
        heading_error > 0: tangente da trajetoria aponta para a esquerda.
        """
        if not all(math.isfinite(v) for v in (cross_track, heading_error, velocity, dt)):
            raise ValueError("Entradas do Stanley devem ser finitas.")
        if dt <= 0 or velocity < -0.05:
            raise ValueError("Stanley requer dt > 0 e deslocamento para a frente.")
        c = self.config
        heading = math.atan2(math.sin(heading_error), math.cos(heading_error))
        self.heading_term = heading
        self.cross_track_term = math.atan2(c.gain * cross_track, c.soft_speed + abs(velocity))
        self.raw_steering = heading + self.cross_track_term
        target = max(-c.max_steer, min(c.max_steer, self.raw_steering))
        self.target_steering = target
        increment = c.max_steer_rate * dt
        self.steering += max(-increment, min(increment, target - self.steering))
        return self.steering


def curve_speed(cruise_speed, curvature, lateral_acceleration=0.5):
    """Limita v pela aceleracao lateral v^2*|kappa|, em m/s e 1/m."""
    if not all(math.isfinite(v) for v in (cruise_speed, curvature, lateral_acceleration)):
        raise ValueError("Velocidade, curvatura e aceleracao devem ser finitas.")
    if cruise_speed <= 0 or lateral_acceleration <= 0:
        raise ValueError("Velocidade e aceleracao lateral devem ser positivas.")
    return min(cruise_speed, math.sqrt(lateral_acceleration / max(abs(curvature), 1e-6)))
