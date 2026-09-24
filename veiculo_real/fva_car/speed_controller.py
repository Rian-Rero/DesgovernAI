"""Controlador PI de velocidade usado pelo carrinho real.

Os ganhos sao os mesmos validados inicialmente no simulador. No veiculo real
eles devem ser refinados a partir dos logs dos ensaios, sem alterar a logica de
anti-windup.
"""

import math

SPEED_KP = 4.0  # 1/s
SPEED_KI = 2.0  # 1/s^2
CONTROL_SAMPLE_TIME = 0.05  # s: 20 Hz, igual ao projeto no simulador
VELOCITY_FILTER_TIME = 0.03  # s
CONTROL_FILTER_TIME = 0.05  # s: suaviza a acao sem atrasar muito a resposta
STOP_SPEED_THRESHOLD = 0.03  # m/s


class SpeedPI:
	"""PI com saturacao e integracao condicional (anti-windup)."""

	def __init__(
		self,
		kp: float = SPEED_KP,
		ki: float = SPEED_KI,
		output_filter_time: float = CONTROL_FILTER_TIME,
	):
		if kp < 0.0 or ki < 0.0:
			raise ValueError("Os ganhos do PI devem ser nao negativos.")
		if output_filter_time < 0.0:
			raise ValueError("A constante de tempo do filtro nao pode ser negativa.")
		self.kp = float(kp)
		self.ki = float(ki)
		self.output_filter_time = float(output_filter_time)
		self.reset()

	def reset(self) -> None:
		self.integral = 0.0
		self.output = 0.0

	def update(
		self, error: float, dt: float, lower_limit: float, upper_limit: float
	) -> float:
		if dt < 0.0:
			raise ValueError("O intervalo de amostragem nao pode ser negativo.")
		if lower_limit > upper_limit:
			raise ValueError("O limite inferior deve ser menor que o superior.")

		unrestricted = self.kp * error + self.ki * self.integral
		target = min(max(unrestricted, lower_limit), upper_limit)

		# Integra somente sem saturacao ou quando o erro desfaz a saturacao.
		if (
			unrestricted == target
			or (unrestricted > upper_limit and error < 0.0)
			or (unrestricted < lower_limit and error > 0.0)
		):
			self.integral += error * dt

		if self.output_filter_time > 0.0 and dt > 0.0:
			alpha = -math.expm1(-dt / self.output_filter_time)
			self.output += alpha * (target - self.output)
		else:
			self.output = target

		# Garante os limites mesmo se eles mudarem entre duas amostras.
		self.output = min(max(self.output, lower_limit), upper_limit)
		return self.output
