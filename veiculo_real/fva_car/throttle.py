"""Feedforward do acelerador aprendido: throttle de regime por faixa de velocidade.

O ESC so move o carro acima de ~0,35 de acelerador (zona morta), e o throttle de
regime nao e uma reta: nos ensaios de 10/11 foi ~0,40 de 0,3 a 0,55 m/s, 0,45 a
0,65 e 0,59 a 0,9 m/s. Ele muda com o carro, o ESC e a carga da bateria.

Em regime (velocidade parada na referencia, sem saturar), o throttle aplicado e
exatamente o que aquela velocidade pede. O mapa guarda uma media movel por faixa
de 0,1 m/s e o feedforward passa a sair dele, interpolado. Fora das faixas ja
medidas, segue o modelo offset + ganho*v deslocado para encostar na faixa medida
mais proxima. Salvo entre corridas (main.py, logs/acelerador.json), como o trim
da direcao.
"""
import math

import numpy as np

AUTO_MAX_SPEED_M_S = 6.0  # teto de referencia e de atuacao no modo automatico
SPEEDS = np.round(np.arange(.2, 1.6001, .1), 2)  # centros das faixas, m/s
MIN_SAMPLES = 25    # passos em regime (0,5 s a 50 Hz) para a faixa valer
TAU_S = 2.          # constante de tempo da media de cada faixa
U_LIMITS = (.1, .95)


class ThrottleMap:
    def __init__(self, offset, gain, u=None, n=None, u_max=1.):
        self.offset, self.gain = float(offset), float(gain)
        # Maior comando (servos.Servos.max_command): acima de 1,5 m/s passa de 1.
        self.u_max = float(u_max)
        self.u = np.full(len(SPEEDS), np.nan) if u is None else np.asarray(u, float).copy()
        self.n = np.zeros(len(SPEEDS), int) if n is None else np.asarray(n, int).copy()
        self.learned_s = 0.  # tempo em regime nesta corrida

    @classmethod
    def from_dict(cls, data, offset, gain):
        """Mapa salvo (ou None) -> ThrottleMap; arquivo invalido comeca vazio."""
        try:
            speeds = np.asarray(data["v"], float)
            u = np.asarray([math.nan if value is None else value for value in data["u"]], float)
            n = np.asarray(data["n"], int)
            if speeds.shape != SPEEDS.shape or not np.allclose(speeds, SPEEDS) or u.shape != n.shape:
                raise ValueError
            bad = ~np.isfinite(u) | (u < U_LIMITS[0]) | (u > U_LIMITS[1])
            u[bad], n[bad] = math.nan, 0
            return cls(offset, gain, u, np.maximum(n, 0))
        except (TypeError, KeyError, ValueError):
            return cls(offset, gain)

    def to_dict(self):
        return {"v": SPEEDS.tolist(), "u": [None if not math.isfinite(v) else round(float(v), 4) for v in self.u],
                "n": self.n.tolist()}

    def _base(self, v):
        return self.offset + self.gain*v

    def feedforward(self, v):
        v = abs(float(v))
        if v <= .05:
            return 0.
        known = (self.n >= MIN_SAMPLES) & np.isfinite(self.u)
        if not np.any(known):
            return float(np.clip(self._base(v), 0., self.u_max))
        speeds, values = SPEEDS[known], self.u[known]
        if v <= speeds[0]:
            value = self._base(v) + values[0] - self._base(speeds[0])
        elif v >= speeds[-1]:
            value = self._base(v) + values[-1] - self._base(speeds[-1])
        else:
            value = float(np.interp(v, speeds, values))
        return float(np.clip(value, 0., self.u_max))

    def update(self, v, u, dt):
        """Uma amostra em regime: velocidade medida v (m/s) com throttle u."""
        index = int(round((abs(float(v)) - SPEEDS[0])/(SPEEDS[1] - SPEEDS[0])))
        if not 0 <= index < len(SPEEDS) or not U_LIMITS[0] <= u <= U_LIMITS[1] or dt <= 0:
            return
        if self.n[index] == 0 or not math.isfinite(self.u[index]):
            self.u[index] = float(u)
        else:
            self.u[index] += (1. - math.exp(-dt/TAU_S))*(float(u) - self.u[index])
        self.n[index] += 1
        self.learned_s += float(dt)

    def summary(self):
        known = (self.n >= MIN_SAMPLES) & np.isfinite(self.u)
        return ", ".join(f"{v:.1f} m/s: {u:.2f}" for v, u in zip(SPEEDS[known], self.u[known])) or "nenhuma faixa"
