"""Controlador PI de velocidade usado pelo carrinho real.

Os ganhos sao os mesmos validados inicialmente no simulador. No veiculo real
eles devem ser refinados a partir dos logs dos ensaios, sem alterar a logica de
anti-windup.
"""

SPEED_KP = 8.0  # 1/s: compromisso entre acomodacao rapida e amortecimento
SPEED_KI = 4.0  # 1/s^2
CONTROL_SAMPLE_TIME = 0.05  # s: 20 Hz, igual ao projeto no simulador
VELOCITY_FILTER_TIME = 0.03  # s
STOP_SPEED_THRESHOLD = 0.03  # m/s


class SpeedPI:
	"""PI com saturacao e integracao condicional (anti-windup)."""

	def __init__(self, kp: float = SPEED_KP, ki: float = SPEED_KI):
		if kp < 0.0 or ki < 0.0:
			raise ValueError("Os ganhos do PI devem ser nao negativos.")
		self.kp = float(kp)
		self.ki = float(ki)
		self.reset()

	def reset(self) -> None:
		self.integral = 0.0

	def update(
		self, error: float, dt: float, lower_limit: float, upper_limit: float
	) -> float:
		if dt < 0.0:
			raise ValueError("O intervalo de amostragem nao pode ser negativo.")
		if lower_limit > upper_limit:
			raise ValueError("O limite inferior deve ser menor que o superior.")

		unrestricted = self.kp * error + self.ki * self.integral
		output = min(max(unrestricted, lower_limit), upper_limit)

		# Integra somente sem saturacao ou quando o erro desfaz a saturacao.
		if (
			unrestricted == output
			or (unrestricted > upper_limit and error < 0.0)
			or (unrestricted < lower_limit and error > 0.0)
		):
			self.integral += error * dt

		return output
