# -*- coding: utf-8 -*-
########################################
# Disciplina: Topicos em Engenharia de Controle e Automacao IV (ENG075):
# Fundamentos de Veiculos Autonomos - 2026/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automacao
# DELT - Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
# -*- coding: utf-8 -*-
from fva_car import Car
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

MAIN_VEL = 1.5  # m/s
OBSTACLE_DISTANCE = 0.20  # m: margem desejada entre o carro parado e o obstaculo
STOP_REACTION_TIME = 0.30  # s: atraso de leitura e atuacao
STOP_DECELERATION = 1.75  # m/s²: ajuste experimental, calibrar com a parada real


def stopping_distance(velocity):
    """Distancia de parada [m] para velocidade [m/s].

    De v_final² = v² - 2*a*d, com v_final = 0:
    d = margem + |v|*tempo_reacao + v²/(2*a).
    a representa a desaceleracao efetiva ao cortar o throttle.
    """
    speed = abs(velocity)
    return (
        OBSTACLE_DISTANCE
        + speed * STOP_REACTION_TIME
        + speed ** 2 / (2.0 * STOP_DECELERATION)
    )


########################################
# thread de visao
def vision_func(car, vision_data, stop_event):

    W, H = car.cam.get_resolution()

    while not stop_event.is_set():

        # pega imagem
        frame = car.get_image(gray=True)

        # detecta aruco
        frame, point = car.cam.detect_aruco(frame, aruco_id=23)

        # disponibiliza imagem para o main
        vision_data["frame"] = frame

        if point is None:
            continue

        # esterçamento aponta para o aruco
        cx = point[0] - W / 2

        vision_data["refste"] = -np.deg2rad(20.0 * cx / (W / 2))


########################################
# main
########################################
if __name__ == "__main__":

    parameters = {
        "ts": 20.0,
        "save": True,
        "logfile": "logs/",
        "camera": False,
        "ultrasonic_steering": False,
        "us_buzzer": False,
        "initial_position": [0, 0, np.deg2rad(0)],
    }

    car = Car(parameters)

    vision_data = {"refste": 0.0, "frame": None}

    stop_event = threading.Event()
    thread_vision = None

    try:
        car.start_mission()
        print(
            f"Ensaio: {parameters['ts']:.0f} s com vref = {MAIN_VEL:.1f} m/s",
            flush=True,
        )

        # inicia visao somente se solicitada
        if parameters["camera"]:
            thread_vision = threading.Thread(
                target=vision_func, args=(car, vision_data, stop_event), daemon=True
            )
            thread_vision.start()

        if parameters["camera"]:
            plt.ion()
            plt.figure(1)

        t_plot = time.monotonic()
        obstacle_detected = False
        next_stop_beep = 0.0

        # controle fica na thread principal
        while car.t < parameters["ts"]:

            # atualiza sensores
            if not car.step():
                break

            # direcao
            car.set_steer(vision_data["refste"])

            # ultrassom
            dist, valid = car.get_distance()

            # margem fixa + percurso durante o atraso + distancia de desaceleracao
            speed = abs(car.v)
            stop_distance = stopping_distance(car.v)

            # uma deteccao valida trava a parada ate o fim do ensaio
            if not obstacle_detected and valid and dist <= stop_distance:
                obstacle_detected = True
                print(
                    f"Obstaculo a {dist:.2f} m | velocidade {speed:.2f} m/s | "
                    f"limiar {stop_distance:.2f} m: parada travada ate o fim do ensaio.",
                    flush=True,
                )

            if obstacle_detected:
                car._set_ref(0.0)
                car.vel_error_integral = 0.0
                # corta o throttle mesmo se o PI tiver erro acumulado
                car.set_u(0.0)
                # aviso periodico em thread, sem bloquear o controle de parada
                now = time.monotonic()
                if now >= next_stop_beep:
                    if car.bz.beep([0.15, 0.15], silence=0.15):
                        next_stop_beep = now + 1.0
            else:
                car.set_vel(MAIN_VEL)

            # telemetria para plots remotos
            print(
                f"DATA,"
                f"{car.t:.3f},"
                f"{car.p[0]:.3f},"
                f"{car.p[1]:.3f},"
                f"{car.v:.3f},"
                f"{car.vref:.3f},"
                f"{car.a:.3f},"
                f"{car.u:.3f},"
                f"{car.w:.3f},"
                f"{car.th:.3f}",
                flush=True,
            )

            # atualiza grafico aproximadamente 1 Hz
            if time.monotonic() - t_plot >= 1.0:

                if parameters["camera"]:
                    frame = vision_data["frame"]

                    if frame is not None:
                        plt.cla()
                        plt.imshow(frame, cmap="gray")
                        plt.pause(0.001)

                t_plot = time.monotonic()

        # salva dados
        if parameters["save"]:
            car.save()

    finally:
        # termina a thread de visao
        stop_event.set()

        if thread_vision is not None:
            thread_vision.join(timeout=1.0)

        car.close()

    print("Terminou...")
