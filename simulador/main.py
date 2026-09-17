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

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
import matplotlib.pyplot as plt
from fva_car import Car
from fva_car.speed_controller import CONTROL_SAMPLE_TIME

VELOCITY_REFERENCE = 1.5  # m/s
DISPLAY_RATE = 10.0  # Hz
parameters = {
    'ts': 10.0,
    'save': True,
    'logfile': 'logs/',
    'beep': True,
    'sample_time': CONTROL_SAMPLE_TIME,
    'explicit_vision': True,
    'real_time': True,
}


def control_func(car):
    car.set_steer(0.0)
    car.set_vel(VELOCITY_REFERENCE)


def vision_func(car):
    return car.get_image(gray=False)


class TelemetryPlot:
    def __init__(self, car, duration):
        plt.ion()
        self.figure, (camera, self.velocity_axis) = plt.subplots(2, 1, figsize=(6, 8))
        self.image = camera.imshow(vision_func(car))
        camera.axis('off')
        self.measured, = self.velocity_axis.plot([], [], label='Velocidade medida')
        self.reference, = self.velocity_axis.plot([], [], '--', label='Referencia')
        self.velocity_axis.set(xlabel='Tempo de simulacao (s)', ylabel='Velocidade (m/s)',
                               xlim=(0, duration), ylim=(-0.1, 2.7))
        self.velocity_axis.grid(True)
        self.velocity_axis.legend()
        self.title = self.figure.suptitle('Telemetria')
        plt.show(block=False)

    def update(self, car, elapsed):
        self.image.set_data(vision_func(car))
        stride = max(1, len(car.traj) // 2000)
        trajectory = car.traj[::stride]
        t = [state['t'] for state in trajectory]
        self.measured.set_data(t, [state['v'] for state in trajectory])
        self.reference.set_data(t, [state['vref'] for state in trajectory])
        self.title.set_text(f't = {car.t:.2f} s | ritmo = {car.t/max(elapsed, 1e-9):.2f}x')
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


def run_mission(settings=None, display=True):
    config = dict(parameters if settings is None else settings)
    car = Car(config)
    try:
        car.start_mission()
        telemetry = TelemetryPlot(car, config['ts']) if display else None
        wall_start = time.perf_counter()
        car.wall_start = wall_start
        next_display = wall_start + 1 / DISPLAY_RATE
        while car.t < config['ts']:
            if not car.step():
                break
            control_func(car)
            now = time.perf_counter()
            if telemetry is not None and now >= next_display:
                telemetry.update(car, now - wall_start)
                next_display = time.perf_counter() + 1 / DISPLAY_RATE
            if config.get('real_time', False):
                wait = wall_start + car.t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
    except KeyboardInterrupt:
        print('\nMissao interrompida pelo usuario.')
    finally:
        try:
            if config['save'] and getattr(car, 'traj', None):
                car.save()
                print(f'Log salvo em {car.logfile}/car.csv')
        finally:
            car.close()


if __name__ == '__main__':
    run_mission()
