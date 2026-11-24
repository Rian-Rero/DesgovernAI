"""Guinada medida pelo giroscopio da IMU, com tudo o que pode dar errado tratado.

Sem IMU, a guinada do Car vem do modelo da bicicleta, v*tan(esterco)/L, com o
esterco COMANDADO. Nos logs de 19/11 a roda fazia so 88-90% da curva pedida, e
o pneu escorrega mais quanto mais rapido. A guinada medida corrige isso direto
na odometria: na previsao do estimador entre quadros, na compensacao dos ~220 ms
de atraso da imagem e no mapa (SLAM). Ela tambem permite amortecer o Stanley
pela guinada (fva_car/lateral.py, stanley_yaw_damping).

A IMU deste carro ja queimou uma vez. Por isso, ela nunca e obrigatoria:

  partida    o chip precisa se identificar (WHO_AM_I, em imu.py). Parado, a
             gravidade da o eixo vertical em qualquer montagem (de lado, de
             cabeca para baixo) e o giroscopio da o vies. A partida falha (e o
             Car usa o modelo) se o acelerometro nao mede ~1 g, se as leituras
             sao todas iguais (travada) ou se o vies e absurdo (queimada).
  na volta   erro de I2C: aquele passo usa o modelo; 10 seguidos desligam a IMU
             ate o fim. Leituras identicas por 1 s (travada) ou saturadas: o
             mesmo. Parado, o vies e reaprendido devagar (deriva com a
             temperatura).
  coerencia  em curva, a guinada medida tem que acompanhar a do modelo: razao
             abaixo de 0,3 (eixo errado, sensor morto) desliga a IMU. Essa
             razao, separada por lado, e a RESPOSTA DA DIRECAO medida na volta:
             vale para steering_response e para as escalas por lado.

Sinais: rad/s, esquerda (anti-horario visto de cima) positiva, como o Car.
"""
import math
import time

import numpy as np

DEG = math.pi / 180.0


class GyroYaw:
    def __init__(self, sensor, full_scale_dps=500.0, max_failures=10, stuck_reads=50, still_speed=0.03,
                 still_s=0.5, bias_tau_s=3.0, min_turn_rate=0.25, min_turn_speed=0.3, check_samples=150,
                 min_ratio=0.3):
        """sensor: objeto com read_motion() -> ((ax, ay, az) em g, (gx, gy, gz) em graus/s)."""
        self.sensor = sensor
        self.full_scale_dps = float(full_scale_dps)
        self.max_failures = int(max_failures)
        self.stuck_reads = int(stuck_reads)
        self.still_speed = float(still_speed)
        self.still_s = float(still_s)
        self.bias_tau_s = float(bias_tau_s)
        self.min_turn_rate = float(min_turn_rate)
        self.min_turn_speed = float(min_turn_speed)
        self.check_samples = int(check_samples)
        self.min_ratio = float(min_ratio)
        self.enabled = False
        self.reason = "nao calibrada"
        self.up = np.array([0.0, 0.0, 1.0])
        self.bias = np.zeros(3)         # graus/s
        self.bias_start = None
        self.failures = self.consecutive = self.saturated = self.used = 0
        self._previous = None
        self._same = 0
        self._still_for = 0.0
        self._turns = {1: [0.0, 0.0, 0], -1: [0.0, 0.0, 0]}  # lado: [soma w*m, soma m*m, n]
        self.max_rate = 0.0
        self.last = math.nan

    # ------------------------------------------------------------------ partida
    def calibrate(self, samples=100, delay=0.01, sleep=time.sleep):
        """Carro parado: eixo vertical pela gravidade e vies do giroscopio.

        Levanta RuntimeError explicando o problema; o Car entao segue sem IMU."""
        accel, gyro = [], []
        for _ in range(int(samples)):
            a, g = self.sensor.read_motion()
            accel.append(a)
            gyro.append(g)
            sleep(delay)
        accel, gyro = np.asarray(accel, float), np.asarray(gyro, float)
        if not (np.all(np.isfinite(accel)) and np.all(np.isfinite(gyro))):
            raise RuntimeError("leituras invalidas (nan)")
        if np.all(np.ptp(accel, axis=0) == 0) and np.all(np.ptp(gyro, axis=0) == 0):
            raise RuntimeError("leituras todas iguais: sensor travado ou queimado")
        mean = accel.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if not 0.8 <= norm <= 1.2:
            raise RuntimeError(f"acelerometro mede {norm:.2f} g parado (esperado ~1 g): sensor com defeito")
        noise = float(np.max(gyro.std(axis=0)))
        if noise > 2.0:
            raise RuntimeError(f"giroscopio oscila {noise:.1f} graus/s parado: carro mexendo ou sensor ruim")
        bias = gyro.mean(axis=0)
        if float(np.max(np.abs(bias))) > 20.0:
            raise RuntimeError(f"vies de {np.max(np.abs(bias)):.0f} graus/s: sensor com defeito")
        self.up = mean / norm
        self.bias = bias
        self.bias_start = float(bias @ self.up)
        self.enabled, self.reason = True, ""
        return self

    def disable(self, reason):
        if self.enabled:
            print(f"\033[33mIMU desligada: {reason}. Guinada pelo modelo da direcao.\033[0m", flush=True)
        self.enabled, self.reason = False, reason

    # ------------------------------------------------------------------ na volta
    def read(self, speed, model_rate, dt):
        """Guinada medida (rad/s) ou nan (aquele passo usa o modelo).

        speed: m/s do encoder; model_rate: guinada do modelo SEM a correcao de
        resposta (v*tan(esterco)/L), para medir a resposta da direcao."""
        if not self.enabled:
            return math.nan
        try:
            accel, gyro = self.sensor.read_motion()
            gyro = np.asarray(gyro, float)
            if gyro.shape != (3,) or not np.all(np.isfinite(gyro)):
                raise ValueError("leitura invalida")
        except Exception as error:  # I2C (OSError), leitura curta, etc.
            self.failures += 1
            self.consecutive += 1
            if self.consecutive >= self.max_failures:
                self.disable(f"{self.consecutive} leituras seguidas falharam ({error})")
            return math.nan
        self.consecutive = 0
        sample = (tuple(accel), tuple(gyro))
        self._same = self._same + 1 if sample == self._previous else 0
        self._previous = sample
        if self._same >= self.stuck_reads:
            self.disable(f"leituras identicas por {self._same} passos (sensor travado)")
            return math.nan
        if float(np.max(np.abs(gyro))) >= 0.98 * self.full_scale_dps:
            self.saturated += 1
            return math.nan
        speed = float(speed) if math.isfinite(speed) else 0.0
        dt = float(dt) if math.isfinite(dt) and dt > 0 else 0.0
        self._still_for = self._still_for + dt if abs(speed) < self.still_speed else 0.0
        if self._still_for >= self.still_s and float(np.max(np.abs(gyro - self.bias))) < 3.0:
            # Parado ha um tempo: o que o giroscopio mede e vies (deriva com a temperatura).
            self.bias += (1.0 - math.exp(-dt / self.bias_tau_s)) * (gyro - self.bias)
        rate = float((gyro - self.bias) @ self.up) * DEG
        self.used += 1
        self.max_rate = max(self.max_rate, abs(rate))
        self.last = rate
        if abs(speed) >= self.min_turn_speed and math.isfinite(model_rate) and abs(model_rate) >= self.min_turn_rate:
            side = self._turns[1 if model_rate > 0 else -1]
            side[0] += rate * model_rate
            side[1] += model_rate * model_rate
            side[2] += 1
            ratio = self.response_ratio()
            if ratio is not None and ratio < self.min_ratio:
                self.disable(f"nao acompanha as curvas (resposta {ratio:.2f}): eixo errado ou sensor ruim")
                return math.nan
        return rate

    # ------------------------------------------------------------------ saida
    def response_ratio(self, side=None, min_samples=None):
        """Guinada medida / guinada do comando nas curvas (um lado ou os dois)."""
        min_samples = self.check_samples if min_samples is None else min_samples
        sides = [self._turns[side]] if side else list(self._turns.values())
        xy, xx, n = (sum(s[i] for s in sides) for i in range(3))
        return xy / xx if n >= min_samples and xx > 0 else None

    def response(self, min_samples=50):
        """{'esquerda': razao, 'direita': razao}, None onde ha poucas curvas."""
        return {"esquerda": self.response_ratio(1, min_samples), "direita": self.response_ratio(-1, min_samples)}

    def summary(self):
        response = self.response()
        return {
            "ativa": self.enabled,
            "motivo": self.reason,
            "leituras_usadas": self.used,
            "falhas_i2c": self.failures,
            "saturadas": self.saturated,
            "vies_inicio_graus_s": None if self.bias_start is None else round(self.bias_start, 3),
            "vies_fim_graus_s": None if self.bias_start is None else round(float(self.bias @ self.up), 3),
            "eixo_vertical": [round(float(v), 3) for v in self.up],
            "guinada_max_rad_s": round(self.max_rate, 3),
            "resposta_direcao": {k: None if v is None else round(v, 3) for k, v in response.items()},
        }


def start_gyro(parameters, imu_factory=None, log=print):
    """(sensor, GyroYaw, situacao) pela configuracao; sem IMU, (None, None, motivo).

    Qualquer falha (sem smbus, sem o chip no barramento, chip queimado, carro
    mexendo na calibracao) deixa o Car no modelo da direcao, como antes."""
    if not parameters.get("imu", True):
        return None, None, "desligada na configuracao"
    sensor = None
    try:
        if imu_factory is None:
            try:
                from . import imu as imu_module
            except ImportError:
                import imu as imu_module
            imu_factory = imu_module.IMU
        full_scale = float(parameters.get("imu_gyro_range_dps", 500))
        sensor = imu_factory(bus_id=int(parameters.get("imu_bus", 4)), auto_calibrate=False,
                             gyro_range_dps=int(full_scale))
        gyro = GyroYaw(sensor, full_scale_dps=full_scale).calibrate()
        status = (f"{getattr(sensor, 'chip', 'ok')} | eixo vertical {np.round(gyro.up, 2).tolist()}"
                  f" | vies {gyro.bias_start:+.2f} graus/s")
        log(f"\033[32mIMU: {status}\033[0m")
        return sensor, gyro, status
    except Exception as error:  # noqa: BLE001 - IMU nunca impede a missao
        status = f"{type(error).__name__}: {error}"
        log(f"\033[33mIMU indisponivel ({status}). Guinada pelo modelo da direcao.\033[0m")
        if sensor is not None:
            try:
                sensor.close()
            except Exception:  # noqa: BLE001
                pass
        return None, None, status
