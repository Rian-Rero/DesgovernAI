# -*- coding: utf-8 -*-
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075):
# Fundamentos de Veículos Autônomos - 2026/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
import os
import math
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import matplotlib.pyplot as plt
from fva_car import Car
from fva_car.speed_controller import CONTROL_SAMPLE_TIME
from fva_car.lateral import YellowLineFollower

VELOCITY_REFERENCE = 1.5  # m/s
DISPLAY_RATE = 10.0  # Hz
TELEMETRY_RESOLUTION = (2304, 1344)  # pixels
TELEMETRY_DPI = 100
parameters = {
    "ts": 120.0,
    "save": True,
    "logfile": "logs/",
    "beep": True,
    "sample_time": CONTROL_SAMPLE_TIME,
    "explicit_vision": True,
    "real_time": True,
    "velocity_reference": VELOCITY_REFERENCE,
    "stanley_gain": 1.5,  # 1/s: peso do erro de distancia lateral
    "stanley_softening": 0.3,  # m/s: evita divisao por zero na partida
    "stanley_max_steer_deg": 20.0,  # graus: limite fisico do carrinho
    "stanley_max_rate_deg_s": 60.0,  # graus/s: limita movimentos bruscos
    "curve_min_speed": 0.5,  # m/s durante recuperacao de erro/curvas
    "curve_lateral_accel": 0.6,  # m/s^2: limite nominal v^2*|curvatura|
    "steering_sign": -1.0,  # juntas positivas da cena Car viram para a direita
    "line_ground_z": 0.005,  # altura da pintura da pista no mundo, em metros
    # Segmentacao amarela em HSV: H em graus (0..360), S/V entre 0 e 1.
    "yellow_hsv_lower": (40.0, 0.4, 0.2),
    "yellow_hsv_upper": (75.0, 1.0, 1.0),
}


def control_func(car):
    if not hasattr(car, 'line_follower'):
        car.line_follower = YellowLineFollower(car)
    car.line_follower(car)


def vision_func(car):
    image = getattr(car, 'line_image', None)
    return image if image is not None else car.get_image(gray=False)


class TelemetryPlot:
    def __init__(self, car, duration):
        plt.ion()
        width, height = TELEMETRY_RESOLUTION
        self.figure = plt.figure(
            figsize=(width / TELEMETRY_DPI, height / TELEMETRY_DPI), dpi=TELEMETRY_DPI
        )
        grid = self.figure.add_gridspec(3, 2, height_ratios=(1.2, 1, 1))
        camera = self.figure.add_subplot(grid[0, :])
        self.lateral_axis = self.figure.add_subplot(grid[1, 0])
        self.heading_axis = self.lateral_axis.twinx()
        self.velocity_axis = self.figure.add_subplot(grid[1, 1], sharex=self.lateral_axis)
        self.control_axis = self.figure.add_subplot(grid[2, 0], sharex=self.lateral_axis)
        self.steering_axis = self.figure.add_subplot(grid[2, 1], sharex=self.lateral_axis)
        self.image = camera.imshow(vision_func(car))
        camera.axis("off")
        (self.lateral_error,) = self.lateral_axis.plot([], [], label="Erro lateral")
        (self.heading_error,) = self.heading_axis.plot(
            [], [], color="tab:orange", linestyle="--", label="Erro de orientacao"
        )
        zero_error = self.lateral_axis.axhline(
            0, color="gray", linestyle=":", label="Erro zero"
        )
        self.lateral_axis.set(
            xlabel="Tempo de simulacao (s)", ylabel="Erro lateral (m)",
            xlim=(0, duration), ylim=(-0.5, 0.5),
            title="Erros do Stanley (+ linha a esquerda)",
        )
        self.heading_axis.set(ylabel="Erro de orientacao (graus)", ylim=(-10, 10))
        self.heading_axis.tick_params(axis="y", colors="tab:orange", labelsize=12)
        self.heading_axis.yaxis.label.set_color("tab:orange")
        self.heading_axis.yaxis.label.set_size(14)
        self.lateral_axis.grid(True)
        self.lateral_axis.legend(handles=[self.lateral_error, self.heading_error, zero_error])
        (self.measured,) = self.velocity_axis.plot([], [], label="Velocidade medida")
        self.velocity_axis.axhline(
            car.parameters.get("velocity_reference", VELOCITY_REFERENCE),
            color="tab:red", linestyle="--", linewidth=1.5, label="Referencia nominal",
        )
        (self.reference,) = self.velocity_axis.plot(
            [], [], color="tab:orange", linestyle=":", drawstyle="steps-post",
            label="Referencia ajustada (curvas/erro)",
        )
        self.velocity_axis.set(
            xlabel="Tempo de simulacao (s)", ylabel="Velocidade (m/s)",
            xlim=(0, duration), ylim=(-0.1, 2.7), title="Velocidade e referencia",
        )
        self.velocity_axis.grid(True)
        self.velocity_axis.legend()
        (self.pwm,) = self.control_axis.plot(
            [], [], drawstyle="steps-pre", color="tab:green", label="PWM equivalente"
        )
        self.control_axis.set(
            xlabel="Tempo de simulacao (s)",
            ylabel="PWM equivalente (%)",
            xlim=(0, duration),
            ylim=(-105, 105),
            yticks=(-100, -50, 0, 50, 100),
        )
        self.control_axis.set_title("Acao de controle dos motores", fontsize=16)
        self.control_axis.axhline(0, color="gray", linewidth=0.8)
        self.control_axis.grid(True)
        self.control_axis.legend()
        (self.steering_command,) = self.steering_axis.plot(
            [], [], drawstyle="steps-post", label="Comando equivalente"
        )
        (self.steering_left,) = self.steering_axis.plot([], [], "--", label="Roda esquerda")
        (self.steering_right,) = self.steering_axis.plot([], [], ":", label="Roda direita")
        self.steering_axis.set(
            xlabel="Tempo de simulacao (s)", ylabel="Esterçamento (graus)",
            xlim=(0, duration), ylim=(-30, 30), title="Esterçamento (sinal das juntas)",
        )
        self.steering_axis.axhline(0, color="gray", linewidth=0.8)
        self.steering_axis.grid(True)
        self.steering_axis.legend()
        for axis in (self.lateral_axis, self.velocity_axis, self.control_axis, self.steering_axis):
            axis.title.set_fontsize(16)
            axis.tick_params(labelsize=12)
            axis.xaxis.label.set_size(14)
            axis.yaxis.label.set_size(14)
        self.title = self.figure.suptitle("Telemetria")
        self.figure.subplots_adjust(
            left=0.07, right=0.97, bottom=0.06, top=0.94, wspace=0.25, hspace=0.55
        )
        plt.show(block=False)
        window = getattr(self.figure.canvas.manager, "window", None)
        if window is not None and hasattr(window, "showMaximized"):
            window.showMaximized()

    def update(self, car, elapsed):
        self.image.set_data(vision_func(car))
        stride = max(1, len(car.traj) // 2000)
        trajectory = car.traj[::stride]
        t = [state["t"] for state in trajectory]
        # NaN interrompe a curva quando a linha nao foi detectada.
        errors = [state.get("cross_track_m", float("nan"))
                  if state.get("line_valid") == 1 else float("nan")
                  for state in trajectory]
        self.lateral_error.set_data(t, errors)
        finite = [abs(error) for error in errors if math.isfinite(error)]
        limit = max(0.5, 1.15 * max(finite, default=0.0))
        self.lateral_axis.set_ylim(-limit, limit)
        headings = [math.degrees(state.get("heading_error_rad", float("nan")))
                    if state.get("line_valid") == 1 else float("nan")
                    for state in trajectory]
        self.heading_error.set_data(t, headings)
        finite_headings = [abs(error) for error in headings if math.isfinite(error)]
        heading_limit = max(10., 1.15 * max(finite_headings, default=0.))
        self.heading_axis.set_ylim(-heading_limit, heading_limit)
        self.measured.set_data(t, [state["v"] for state in trajectory])
        self.reference.set_data(t, [state["vref"] for state in trajectory])
        self.pwm.set_data(t, [state["motor_pwm_percent"] for state in trajectory])
        for line, column in ((self.steering_command, "steer_cmd_rad"),
                             (self.steering_left, "steer_left_rad"),
                             (self.steering_right, "steer_right_rad")):
            line.set_data(t, [math.degrees(state.get(column, float("nan")))
                              for state in trajectory])
        self.title.set_text(
            f"t = {car.t:.2f} s | ritmo = {car.t/max(elapsed, 1e-9):.2f}x"
            + (f" | linha: {'OK' if car.traj[-1].get('line_valid') else 'PERDIDA - PARANDO'}"
               f" | erro = {car.traj[-1].get('cross_track_m', float('nan')):.3f} m"
               if 'line_valid' in car.traj[-1] else '')
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


def run_mission(settings=None, display=True, control=None):
    """Executa a missao; control(car) roda a cada passo (padrao: control_func)."""
    config = dict(parameters if settings is None else settings)
    control = control_func if control is None else control
    car = Car(config)
    try:
        car.start_mission()
        telemetry = TelemetryPlot(car, config["ts"]) if display else None
        wall_start = time.perf_counter()
        car.wall_start = wall_start
        next_display = wall_start + 1 / DISPLAY_RATE
        while car.t < config["ts"]:
            if not car.step():
                break
            control(car)
            now = time.perf_counter()
            if telemetry is not None and now >= next_display:
                telemetry.update(car, now - wall_start)
                next_display = time.perf_counter() + 1 / DISPLAY_RATE
            if config.get("real_time", False):
                wait = wall_start + car.t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
    except KeyboardInterrupt:
        print("\nMissao interrompida pelo usuario.")
    finally:
        try:
            if config["save"] and getattr(car, "traj", None):
                car.save()
                print(f"Log salvo em {car.logfile}/car.csv")
        finally:
            car.close()


if __name__ == "__main__":
    run_mission()
