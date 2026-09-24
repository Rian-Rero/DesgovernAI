"""PI para a entrada normalizada, com velocidade filtrada e anti-windup."""

SPEED_KP = 8.0  # 1/s: compromisso entre acomodacao rapida e amortecimento
SPEED_KI = 4.0  # 1/s^2
CONTROL_SAMPLE_TIME = 0.05  # s: 20 Hz para a banda do PI
VELOCITY_FILTER_TIME = 0.03  # s
MOTOR_INPUT_GAIN = 0.77
COAST_DECELERATION = 0.5880832138991398
MOTOR_TORQUE_FACTOR = 0.63


class SpeedPI:
    def __init__(self, kp: float = SPEED_KP, ki: float = SPEED_KI):
        self.kp = kp
        self.ki = ki
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0

    def update(
        self, error: float, dt: float, lower_limit: float, upper_limit: float
    ) -> float:
        if dt < 0:
            raise ValueError("O intervalo de amostragem nao pode ser negativo.")

        unrestricted = self.kp * error + self.ki * self.integral
        output = min(max(unrestricted, lower_limit), upper_limit)

        # Integra somente sem saturacao ou quando o erro desfaz a saturacao.
        if (
            unrestricted == output
            or (unrestricted > upper_limit and error < 0)
            or (unrestricted < lower_limit and error > 0)
        ):
            self.integral += error * dt

        return output
