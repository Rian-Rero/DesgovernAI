"""Frenagem frontal: coast-down quando possivel, freio quando necessario."""

from dataclasses import dataclass
from enum import IntEnum
import math


class BrakingMode(IntEnum):
    CRUISE = 0
    COAST = 1
    BRAKE = 2
    HOLD = 3
    FAULT = 4


@dataclass(frozen=True)
class BrakingConfig:
    # Valores nominais: medir coast-down e resposta do ESC antes de calibrar.
    coast_deceleration: float = 0.2  # m/s^2, limite inferior esperado
    clearance: float = 0.20  # m entre sensor e obstaculo na parada
    reaction_time: float = 0.40  # s, inclui sensor, filtro e atuador
    anticipation_time: float = 0.30  # s, inicia coast antes do limite
    stop_speed: float = 0.05  # m/s
    release_margin: float = 0.15  # m, histerese na liberacao
    release_time: float = 0.50  # s com caminho livre antes de retomar
    max_brake: float = 1.0  # fracao maxima do comando de freio

    def __post_init__(self):
        positive = ("coast_deceleration", "clearance", "stop_speed", "release_time")
        nonnegative = ("reaction_time", "anticipation_time", "release_margin")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Parametro ADAS invalido: {name}={value}")
        if not math.isfinite(self.max_brake) or not 0 < self.max_brake <= 1:
            raise ValueError("max_brake deve estar entre 0 e 1")


@dataclass(frozen=True)
class BrakingDecision:
    mode: BrakingMode
    reason: str
    coast_stop_distance: float
    activation_distance: float
    required_deceleration: float
    target_speed: float = 0.0
    release_distance: float = 0.0


class BrakingADAS:
    """Mantem a intervencao ate parar; nao comanda deslocamento para tras."""

    def __init__(self, config=None):
        self.config = config or BrakingConfig()
        self.mode = BrakingMode.CRUISE
        self._clear_since = None

    def stopping_distance(self, speed):
        speed = max(0.0, speed)
        return (
            self.config.clearance
            + speed * self.config.reaction_time
            + speed * speed / (2.0 * self.config.coast_deceleration)
        )

    def safe_speed(self, distance):
        """Inverte a distancia de parada, incluindo antecipacao e histerese."""
        cfg = self.config
        available = max(0.0, distance - cfg.clearance - cfg.release_margin)
        if available == 0:
            return 0.0
        delay = cfg.reaction_time + cfg.anticipation_time
        return 2.0 * available / (
            math.sqrt(delay * delay + 2.0 * available / cfg.coast_deceleration) + delay
        )

    def update(self, speed, distance, distance_valid, desired_speed, now, speed_valid=True):
        if not math.isfinite(desired_speed) or desired_speed < 0:
            raise ValueError("O ADAS frontal exige referencia de velocidade >= 0")
        if not math.isfinite(now):
            raise ValueError("Instante ADAS invalido")

        cfg = self.config
        usable_speed = speed_valid and math.isfinite(speed)
        usable_distance = distance_valid and math.isfinite(distance) and distance >= 0
        forward_speed = max(0.0, speed) if usable_speed else 0.0
        coast_stop = self.stopping_distance(forward_speed)
        activation = coast_stop + forward_speed * cfg.anticipation_time
        available = distance - cfg.clearance - forward_speed * cfg.reaction_time
        required = (
            forward_speed * forward_speed / (2.0 * available)
            if usable_distance and available > 0 else math.inf
        )
        target_speed = min(desired_speed, self.safe_speed(distance)) if usable_distance else 0.0
        release_distance = (
            self.stopping_distance(target_speed)
            + target_speed * cfg.anticipation_time + cfg.release_margin
        )

        def decide(mode, reason):
            self.mode = mode
            if mode != BrakingMode.HOLD:
                self._clear_since = None
            return BrakingDecision(
                mode, reason, coast_stop, activation, required,
                target_speed if mode == BrakingMode.CRUISE else 0.0, release_distance,
            )

        if not usable_speed:
            return decide(BrakingMode.FAULT, "encoder invalido: corta tracao e bloqueia retomada")
        # O sensor e frontal; uma eventual inversao corta o freio de imediato.
        if speed < -cfg.stop_speed:
            return decide(BrakingMode.HOLD, "movimento para tras: comando neutro")
        if not usable_distance:
            mode = BrakingMode.BRAKE if forward_speed > cfg.stop_speed else BrakingMode.FAULT
            return decide(mode, "ultrassom invalido: parada preventiva")

        stopped = abs(speed) <= cfg.stop_speed
        if self.mode != BrakingMode.CRUISE and stopped:
            if target_speed > cfg.stop_speed and distance + 1e-9 >= release_distance:
                if self._clear_since is None:
                    self._clear_since = now
                if now - self._clear_since >= cfg.release_time:
                    return decide(BrakingMode.CRUISE, "caminho livre confirmado: retomada com velocidade adaptada")
            else:
                self._clear_since = None
            reason = (
                "parado: confirmando leituras validas para retomar"
                if self._clear_since is not None else "parado: espaco insuficiente para retomar"
            )
            return decide(BrakingMode.HOLD, reason)

        if self.mode == BrakingMode.BRAKE:
            return decide(BrakingMode.BRAKE, "mantem frenagem ate a parada")
        if self.mode in (BrakingMode.COAST, BrakingMode.FAULT, BrakingMode.HOLD):
            if distance < coast_stop:
                return decide(BrakingMode.BRAKE, "coast-down deixou de ser suficiente")
            return decide(BrakingMode.COAST, "mantem roda livre ate a parada")
        if desired_speed == 0:
            return decide(BrakingMode.COAST, "referencia zero: roda livre")
        if stopped and target_speed <= cfg.stop_speed:
            return decide(BrakingMode.HOLD, "obstaculo proximo: impede partida")
        if distance <= activation:
            if distance >= coast_stop:
                return decide(BrakingMode.COAST, "coast-down suficiente: roda livre")
            return decide(BrakingMode.BRAKE, "distancia insuficiente para coast-down")
        reason = (
            "referencia reduzida conforme espaco disponivel"
            if target_speed < desired_speed else "distancia livre"
        )
        return decide(BrakingMode.CRUISE, reason)
