# -*- coding: utf-8 -*-
"""Stanley no veiculo real, com a percepcao HSV de tools/yellow_path.py."""
import argparse
from pathlib import Path
import signal
import threading
import time

import numpy as np

from fva_car.lateral import (
    GroundCalibration, LineObservation, RealLineDetector, RealStanleyFollower,
)

MAIN_VEL = 1.0  # m/s: referencia inicial para ajuste no veiculo real
OBSTACLE_DISTANCE = 0.20
STOP_REACTION_TIME = 0.30
STOP_DECELERATION = 1.75  # m/s^2: ajuste experimental ja usado no veiculo

parameters = {
    "ts": 20.0,
    "save": True,
    "logfile": "logs/",
    "camera": True,
    "ultrasonic_steering": False,  # camera fixa apos calibrada
    "us_buzzer": False,
    "initial_position": [0, 0, 0.0],
    "camera_calibration_file": str(
        Path(__file__).resolve().parent / "calibracao" / "camera_lateral.json"
    ),
    "velocity_reference": MAIN_VEL,
    "stanley_gain": 1.5,
    "stanley_softening": 0.3,
    "stanley_max_steer_deg": 20.0,
    "stanley_max_rate_deg_s": 60.0,
    "steering_sign": 1.0,  # convencao do Car real: positivo para esquerda
    "curve_min_speed": 0.3,
    "curve_lateral_accel": 0.4,
    "vision_timeout": 0.3,  # segundos desde o inicio da captura
    # Mesmo modo de reconstrucao do detector atualizado em tools/yellow_path.py.
    "path_settings": {"s_min": 25, "dashed": True, "dash_gap": 120},
    "path_anchor": 0.5,  # preferencia horizontal da faixa, entre 0 e 1
}


def stopping_distance(velocity):
    speed = abs(velocity)
    return (OBSTACLE_DISTANCE + speed * STOP_REACTION_TIME
            + speed ** 2 / (2.0 * STOP_DECELERATION))


def stop_vehicle(car):
    """Neutro imediato: set_vel(0) sozinho poderia conservar integral do PI."""
    car._set_ref(0.0)
    car.vel_error_integral = 0.0
    car.set_u(0.0)
    car.set_steer(0.0)


def vision_func(car, vision_data, stop_event, detector, clock=time.monotonic):
    """Apenas percebe; a thread principal e a unica que envia comandos."""
    while not stop_event.is_set():
        captured_at = clock()
        try:
            frame = car.get_image(gray=False)  # a Camera real entrega BGR
            observation = detector.observe_frame(frame, captured_at)
        except Exception as error:
            vision_data["observation"] = LineObservation(
                timestamp=captured_at, reason=f"Falha de visao: {error}"
            )
            print(f"Visao interrompida: {error}", flush=True)
            return
        # Erros, imagem e horario pertencem ao mesmo frame.
        vision_data["observation"] = observation
        if frame is None:
            stop_event.wait(0.02)


def run_mission(settings=None, display=False, car_factory=None, clock=time.monotonic):
    config = dict(parameters)
    if settings is not None:
        config.update(settings)
    if not config["camera"] or config["ultrasonic_steering"]:
        raise ValueError("Stanley requer camera ligada e fixa (ultrasonic_steering=False).")
    # Valida geometria e ajustes antes de abrir sensores ou atuar nos motores.
    calibration = GroundCalibration.from_file(config["camera_calibration_file"])
    detector = RealLineDetector(
        calibration, path_settings=config.get("path_settings", {}),
        anchor=config.get("path_anchor", 0.5),
    )
    follower = RealStanleyFollower(config)
    if car_factory is None:
        from fva_car import Car
        car_factory = Car
    car = car_factory(config)
    vision_data = {"observation": None}
    stop_event = threading.Event()
    thread_vision = None
    plot = None
    try:
        car.start_mission()
        print(f"Stanley HSV | {config['ts']:.0f} s | teto {config['velocity_reference']:.2f} m/s", flush=True)
        thread_vision = threading.Thread(
            target=vision_func, args=(car, vision_data, stop_event, detector, clock),
            daemon=True, name="linha-hsv",
        )
        thread_vision.start()
        if display:
            import matplotlib.pyplot as plt
            plt.ion()
            figure, axis = plt.subplots()
            plot = (plt, figure, axis)
        t_plot = clock()
        obstacle_detected = False
        next_stop_beep = 0.0
        while car.t < config["ts"]:
            if not car.step():
                break
            dist, valid = car.get_distance()
            stop_distance = stopping_distance(car.v)
            if (not obstacle_detected and valid and np.isfinite(dist)
                    and dist <= stop_distance):
                obstacle_detected = True
                print(f"Obstaculo a {dist:.2f} m: parada travada ate o fim do ensaio.", flush=True)
            observation = vision_data["observation"]
            follower.apply(car, observation, clock(), obstacle_detected=obstacle_detected)
            car.log(obstacle_stop=int(obstacle_detected), distance_m=dist,
                    stopping_distance_m=stop_distance)
            if not thread_vision.is_alive():
                stop_vehicle(car)
                print("Thread de visao encerrada; missao interrompida.", flush=True)
                break
            now = clock()
            if obstacle_detected and now >= next_stop_beep:
                if car.bz.beep([0.15, 0.15], silence=0.15):
                    next_stop_beep = now + 1.0
            # Conserva o formato DATA lido pela GUI remota existente.
            print(f"DATA,{car.t:.3f},{car.p[0]:.3f},{car.p[1]:.3f},"
                  f"{car.v:.3f},{car.vref:.3f},{car.a:.3f},{car.u:.3f},"
                  f"{car.w:.3f},{car.th:.3f}", flush=True)
            if plot is not None and now - t_plot >= 1.0:
                plt, _, axis = plot
                if observation is not None and observation.frame is not None:
                    axis.clear()
                    axis.imshow(observation.frame)
                    axis.set_title(observation.reason or "Linha amarela HSV")
                    axis.axis("off")
                    plt.pause(0.001)
                t_plot = now
    except KeyboardInterrupt:
        print("\nMissao interrompida pelo usuario.", flush=True)
    finally:
        # Neutraliza antes de esperar a thread, inclusive se cap.read estiver bloqueado.
        try:
            stop_vehicle(car)
        finally:
            stop_event.set()
            if thread_vision is not None:
                thread_vision.join(timeout=1.0)
            try:
                if config["save"] and getattr(car, "traj", None):
                    car.save()
            finally:
                car.close()
                if plot is not None:
                    plot[0].close(plot[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", help="JSON da camera calibrada em metros")
    parser.add_argument("--speed", type=float, default=MAIN_VEL, help="Teto de velocidade em m/s")
    parser.add_argument("--display", action="store_true", help="Mostrar a faixa detectada")
    args = parser.parse_args()
    settings = {"velocity_reference": args.speed}
    if args.calibration:
        settings["camera_calibration_file"] = args.calibration
    def interrupt_mission(_signum, _frame):
        # A GUI remota encerra com SIGTERM. Usa o mesmo finally do Ctrl+C.
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt_mission)
    try:
        try:
            run_mission(settings, display=args.display)
        except (FileNotFoundError, ValueError) as error:
            parser.exit(2, f"{error}\nCalibre com veiculo_real/tools/calibrar_camera_lateral.py antes de rodar.\n")
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
