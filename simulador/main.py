# -*- coding: utf-8 -*-
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075):
# Fundamentos de Veículos Autônomos - 2026/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import matplotlib.pyplot as plt
from fva_car import Car
from fva_car.speed_controller import CONTROL_SAMPLE_TIME

VELOCITY_REFERENCE = 1.5  # m/s
DISPLAY_RATE = 10.0  # Hz
TELEMETRY_RESOLUTION = (2304, 1344)  # pixels
TELEMETRY_DPI = 100
parameters = {
    "ts": 20.0,
    "save": True,
    "logfile": "logs/",
    "beep": True,
    "sample_time": CONTROL_SAMPLE_TIME,
    "explicit_vision": True,
    "real_time": True,
}


def control_func(car):
    car.set_steer(0.0)
    car.set_vel(VELOCITY_REFERENCE)


def vision_func(car):
    return car.get_image(gray=False)


class TelemetryPlot:
    def __init__(self, car, duration):
        plt.ion()
        width, height = TELEMETRY_RESOLUTION
        self.figure = plt.figure(
            figsize=(width / TELEMETRY_DPI, height / TELEMETRY_DPI), dpi=TELEMETRY_DPI
        )
        grid = self.figure.add_gridspec(2, 2, height_ratios=(1.2, 1))
        camera = self.figure.add_subplot(grid[0, :])
        self.velocity_axis = self.figure.add_subplot(grid[1, 0])
        self.control_axis = self.figure.add_subplot(grid[1, 1], sharex=self.velocity_axis)
        self.image = camera.imshow(vision_func(car))
        camera.axis("off")
        (self.measured,) = self.velocity_axis.plot([], [], label="Velocidade medida")
        (self.reference,) = self.velocity_axis.plot([], [], "--", label="Referencia")
        self.velocity_axis.set(
            xlabel="Tempo de simulacao (s)",
            ylabel="Velocidade (m/s)",
            xlim=(0, duration),
            ylim=(-0.1, 2.7),
        )
        self.velocity_axis.grid(True)
        self.velocity_axis.legend()
        self.velocity_axis.set_title("Velocidade e referencia", fontsize=16)
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
        for axis in (self.velocity_axis, self.control_axis):
            axis.tick_params(labelsize=12)
            axis.xaxis.label.set_size(14)
            axis.yaxis.label.set_size(14)
        self.title = self.figure.suptitle("Telemetria")
        self.figure.subplots_adjust(
            left=0.07, right=0.97, bottom=0.07, top=0.94, wspace=0.25, hspace=0.25
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
        self.measured.set_data(t, [state["v"] for state in trajectory])
        self.reference.set_data(t, [state["vref"] for state in trajectory])
        self.pwm.set_data(t, [state["motor_pwm_percent"] for state in trajectory])
        self.title.set_text(
            f"t = {car.t:.2f} s | ritmo = {car.t/max(elapsed, 1e-9):.2f}x"
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


def run_mission(settings=None, display=True):
    config = dict(parameters if settings is None else settings)
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
            control_func(car)
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
