# -*- coding: utf-8 -*-
"""Stanley no veiculo real: linha amarela ou centro de cones pela camera."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import queue
import signal
import socket
import threading
import time

import numpy as np
import cv2

from fva_car.cone_debug import ConeDebugRecorder
from fva_car.cones import RealConeDetector
from fva_car.odometry import PoseHistory
from fva_car.lateral import (
    GroundCalibration,
    LineObservation,
    RealLineDetector,
    RealStanleyFollower,
)

MAIN_VEL = 0.7  # m/s: velocidade padrao da missao
FAST_VEL = 1.5  # m/s: selecionada com --fast
TELEMETRY_HZ = 10.0
TELEMETRY_PERIOD = 1.0 / TELEMETRY_HZ
OBSTACLE_DISTANCE = 0.20
STOP_REACTION_TIME = 0.30
STOP_DECELERATION = 1.75  # m/s^2: ajuste experimental ja usado no veiculo

parameters = {
    "ts": 30.0,
    "navigation_mode": "cones",  # somente cones; linha requer selecao explicita
    "cone_settings": {"roi_top": 0, "single_border_recovery": True},
    "save": True,
    "logfile": "logs/",
    "camera": True,
    "ultrasonic_steering": False,  # camera fixa apos calibrada
    "us_buzzer": False,
    "initial_position": [0, 0, 0.0],
    "encoder_protocol": "rc",  # mesmo formato de cinco campos da upstream/rc-input
    "camera_calibration_file": str(
        Path(__file__).resolve().parent / "calibracao" / "camera_lateral.json"
    ),
    "velocity_reference": MAIN_VEL,
    "stanley_gain": 1.5,
    "stanley_softening": 0.3,
    "stanley_use_vref": True,  # True: vref no denominador; False: velocidade medida (car.v)
    "stanley_max_steer_deg": 20.0,
    # 60 graus/s levava 0,33 s (0,5 m a 1,5 m/s) para chegar ao batente:
    # quase todo o trecho visivel antes da curva.
    "stanley_max_rate_deg_s": 120.0,
    # Esterco de regime atan(L*kappa) somado ao Stanley; 0 desliga.
    "stanley_curvature_ff": 1.0,
    "wheelbase": 0.36,  # m: mesmo CAR["L"] de fva_car/car.py
    # Passa-baixas (s) do limite de velocidade em curva; 0 desliga. Evitam os
    # trancos de acelerador ("engasgos"); o esterco nao passa por eles.
    "curvature_filter_s": 0.2,
    "speed_filter_s": 0.3,
    # Leva os erros da imagem ao instante do comando por odometria (ate 0,25 s).
    "vision_latency_compensation": True,
    "steering_sign": 1.0,  # convencao do Car real: positivo para esquerda
    "steering_filter_size": 3,  # Stanley ja limita a taxa; reduz atraso extra do servo
    "curve_min_speed": 0.15,
    "curve_lateral_accel": 0.4,
    "vision_timeout": 1.0,  # segundos desde o inicio da captura
    # Perdendo a faixa, segue a ultima trajetoria vista (odometria) por ate
    # este numero de quadros seguidos sem deteccao (~16-20 quadros/s) antes de
    # parar; 0 volta a parar no primeiro quadro sem faixa.
    "line_loss_frames": 8,
    # Teto em segundos da mesma tolerancia, caso a visao pare de entregar quadros.
    "line_loss_tolerance": 1.0,
    # Durante a perda, soma esterco para o lado da curva (ou da faixa) a esta
    # taxa, ate o batente: ~18 graus apos 8 quadros. 0 so segue a trajetoria.
    "line_search_rate_deg_s": 40.0,
    # Cones: perdendo o corredor, segue a trajetoria prevista por odometria por
    # ate cone_loss_frames quadros (e cone_loss_tolerance s), reduzindo a vref
    # em rampa ate curve_min_speed; depois neutraliza. 0 para no 1o quadro.
    "cone_loss_frames": 6,
    "cone_loss_tolerance": 0.6,
    "cone_max_lateral_jump_m": 0.12,
    "cone_max_heading_jump_rad": 0.35,
    # Debug por quadro em logs/<ensaio>/cones_debug.jsonl + frames JPG so em eventos.
    "cone_debug": True,
    "cone_max_event_frames": 300,
    # Coleta para rotular/avaliar (tools/rotular_cones.py): quadros originais em
    # logs/<ensaio>/frames/ a esta taxa (Hz). 0 desliga; nao usar na prova.
    "record_frames_hz": 0.0,
    "vision_width": 0,  # 0: grade original; 320/480: visao reduzida, homografia original
    # Mesmo modo de reconstrucao do detector atualizado em tools/yellow_path.py.
    # ROI 0: usa a imagem inteira, como no detector HSV de referencia.
    "path_settings": {"s_min": 25, "dashed": True, "dash_gap": 120, "roi_top": 50},
    "path_anchor": 0.5,  # preferencia horizontal da faixa, entre 0 e 1
    # Porta HTTP com a visao ao vivo (http://<ip-do-carro>:porta/); 0 desliga.
    "stream_port": 0,
}


def cone_calibration_warnings(calibration, detector):
    """Avisos de cobertura da homografia para a faixa em que os cones sao usados."""
    from tools.cone_geometry import calibration_report

    settings = getattr(detector, "settings", None)
    max_range = getattr(settings, "max_range_cm", 300)
    if isinstance(max_range, bool) or not isinstance(max_range, (int, float)):
        max_range = 300
    try:
        return calibration_report(calibration, max_range_m=max_range / 100)["warnings"]
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        return [f"nao foi possivel verificar a calibracao: {error}"]


def stopping_distance(velocity):
    speed = abs(velocity)
    return (
        OBSTACLE_DISTANCE
        + speed * STOP_REACTION_TIME
        + speed**2 / (2.0 * STOP_DECELERATION)
    )


def stop_vehicle(car):
    """Neutro imediato: set_vel(0) sozinho poderia conservar integral do PI."""
    car._set_ref(0.0)
    car.vel_error_integral = 0.0
    car.set_u(0.0)
    car.set_steer(0.0)


def capture_func(car, frames, stop_event, clock=time.monotonic):
    """Captura continuamente; a fila guarda somente o frame mais recente."""
    while not stop_event.is_set():
        captured_at = clock()
        error = None
        try:
            frame = car.get_image(gray=False)
        except Exception as failure:
            frame, error = None, str(failure)
        sample = (frame, captured_at, clock() - captured_at, error)
        try:
            frames.put_nowait(sample)
        except queue.Full:
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
            frames.put_nowait(sample)
        if error is not None:
            return
        if frame is None:
            stop_event.wait(0.02)


def vision_func(
    car, vision_data, stop_event, detector, clock=time.monotonic, frames=None
):
    """Apenas percebe; a thread principal e a unica que envia comandos."""
    sequence = 0
    while not stop_event.is_set():
        if frames is not None:
            try:
                frame, captured_at, capture_seconds, capture_error = frames.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue
        else:
            captured_at = clock()
            capture_seconds, capture_error = float("nan"), None
        try:
            if capture_error is not None:
                raise RuntimeError(capture_error)
            if frames is None:
                frame = car.get_image(gray=False)  # a Camera real entrega BGR
                capture_seconds = clock() - captured_at
            processing_at = clock()
            observation = detector.observe_frame(frame, captured_at)
            sequence += 1
            observation = replace(
                observation,
                capture_seconds=capture_seconds,
                processing_seconds=clock() - processing_at,
                sequence=sequence,
            )
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


def save_vision_snapshot(car, observation, status, now):
    """Salva o ultimo frame somente apos neutralizar os motores."""
    folder = getattr(car, "logfile", None)
    if not isinstance(folder, (str, Path)) or observation is None:
        return
    folder = Path(folder)

    def finite(value):
        return float(value) if value is not None and np.isfinite(value) else None

    try:
        for name, frame in (
            ("vision_camera.jpg", observation.source_frame),
            ("vision_detection.jpg", observation.frame),
        ):
            if frame is not None:
                if name == "vision_detection.jpg":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(folder / name), frame):
                    raise OSError(f"Nao foi possivel salvar {name}")
        data = {
            "control_status": status,
            "vision_reason": observation.reason,
            "metric_line_valid": bool(observation.valid),
            "image_size": (
                None
                if observation.source_frame is None
                else list(observation.source_frame.shape[1::-1])
            ),
            "processing_size": observation.processing_size,
            "vision_frame_id": observation.sequence,
            "vision_age_s": finite(now - observation.timestamp),
            "capture_s": finite(observation.capture_seconds),
            "processing_s": finite(observation.processing_seconds),
            "detection_s": finite(observation.detection_seconds),
            "geometry_s": finite(observation.geometry_seconds),
            "errors": None if observation.errors is None else list(observation.errors),
            "color_support": finite(observation.color_support),
            "cone_debug": getattr(observation, "debug", None),
            "encoder_valid": bool(getattr(car, "encoder_valid", False)),
            "sel_direcao": getattr(car, "sel_direcao", None),
            "sel_tracao": getattr(car, "sel_tracao", None),
        }
        from fva_car.cone_debug import clean

        (folder / "vision_status.json").write_text(
            json.dumps(clean(data), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Diagnostico de visao salvo em {folder}", flush=True)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Nao foi possivel salvar diagnostico de visao: {error}", flush=True)


def stream_text(car, status, observation):
    """Legenda do quadro transmitido: o que o controle decidiu neste passo."""
    lines = [status[:48]]
    if len(status) > 48:  # tolerancia de perda: quadros, tempo e fator da vref
        lines.append(status[48:110])
    if observation is not None:
        lines.append("Referencia: " + ("cones" if observation.reference_source == "cones" else "linha amarela"))
    if observation is not None and observation.reference_source == "cones":
        lines.append(f"Cones: verdes={len(observation.green_pixels)} vermelhos={len(observation.red_pixels)}")
        debug = getattr(observation, "debug", None) or {}
        sides = debug.get("sides") or {}
        if sides:
            lines.append(f"Lados: verde={'E' if sides.get('green_left') else 'D'}"
                         f" ({'auto' if sides.get('auto') else 'manual'} {sides.get('score') or 0:+.2f})")
        trajectory = debug.get("trajectory") or {}
        if trajectory:
            text = {"reta": "reta", "curva_esq": "curva esq", "curva_dir": "curva dir"}.get(
                trajectory.get("segment"), "?")
            if trajectory.get("radius_m") is not None and trajectory.get("segment") != "reta":
                text += f" R={abs(trajectory['radius_m']):.1f}m"
            if trajectory.get("curve_ahead_m"):
                text += f" | curva em {trajectory['curve_ahead_m']:.1f}m"
            if debug.get("branch"):
                text += " | BIFURCACAO"
            lines.append("Trecho: " + text)
    if observation is not None and observation.valid:
        lateral, heading, curvature = observation.errors
        lines.append(f"e={lateral:+.3f}m psi={np.degrees(heading):+.1f} k={curvature:+.2f}")
    lines.append(f"v={car.v:.2f} vref={car.vref:.2f} st={np.degrees(car.st):+.1f}")
    return lines


def render_stream_frame(detector, observation, lines):
    image = detector.preview_bgr(observation)
    if image is None:
        width, height = detector.processing_size
        image = np.zeros((height, width, 3), np.uint8)
    else:
        image = image.copy()
    for index, line in enumerate(lines):
        origin = (4, 14 + 14 * index)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run_mission(settings=None, display=False, car_factory=None, clock=time.monotonic):
    config = dict(parameters)
    if settings is not None:
        config.update(settings)
    if not config["camera"] or config["ultrasonic_steering"]:
        raise ValueError(
            "Stanley requer camera ligada e fixa (ultrasonic_steering=False)."
        )
    if config["encoder_protocol"] not in ("rpm", "rc"):
        raise ValueError("encoder_protocol deve ser 'rpm' ou 'rc'.")
    if (
        isinstance(config["steering_filter_size"], bool)
        or not isinstance(config["steering_filter_size"], int)
        or config["steering_filter_size"] < 1
    ):
        raise ValueError("steering_filter_size deve ser inteiro positivo.")
    # Valida geometria e ajustes antes de abrir sensores ou atuar nos motores.
    calibration = GroundCalibration.from_file(config["camera_calibration_file"])
    mode = config.get("navigation_mode", "cones")
    if mode == "cones":
        detector = RealConeDetector(
            calibration, settings=config.get("cone_settings"), render_preview=False,
            processing_width=config.get("vision_width", 0),
        )
        calibration_warnings = cone_calibration_warnings(calibration, detector)
        for warning in calibration_warnings:
            print(f"AVISO calibracao: {warning}", flush=True)
        # Sem chaveamento para linha nem esterco de busca: durante a tolerancia
        # apenas continua a trajetoria ja identificada, com a vref em rampa.
        config["line_loss_frames"] = config.get("cone_loss_frames", 6)
        config["line_loss_tolerance"] = config.get("cone_loss_tolerance", 0.6)
        config["line_search_rate_deg_s"] = 0.
        config["loss_speed_ramp"] = True
        config["max_lateral_jump_m"] = config.get("cone_max_lateral_jump_m", 0.12)
        config["max_heading_jump_rad"] = config.get("cone_max_heading_jump_rad", 0.35)
    elif mode == "line":
        detector = RealLineDetector(
            calibration,
            path_settings=config.get("path_settings", {}),
            anchor=config.get("path_anchor", 0.5),
            render_preview=False,
            processing_width=config.get("vision_width", 0),
        )
    else:
        raise ValueError("navigation_mode deve ser line ou cones")
    follower = RealStanleyFollower(config)
    stream = None
    if config.get("stream_port"):
        from fva_car.stream import VisionStream

        # Abre a porta antes dos motores: porta ocupada falha sem mover o carro.
        stream = VisionStream(
            lambda observation, lines: render_stream_frame(detector, observation, lines),
            port=config["stream_port"],
        )
    if car_factory is None:
        from fva_car import Car

        car_factory = Car
    try:
        car = car_factory(config)
    except BaseException:
        if stream is not None:
            stream.close()
        raise
    recorder = None
    folder = getattr(car, "logfile", None)
    record_hz = float(config.get("record_frames_hz", 0.0) or 0.0)
    if (mode == "cones" and (config.get("cone_debug", True) or record_hz > 0) and config["save"]
            and isinstance(folder, (str, Path))):
        try:
            recorder = ConeDebugRecorder(folder, max_event_frames=config.get("cone_max_event_frames", 300),
                                         record_hz=record_hz)
            recorder.notes["calibration_warnings"] = calibration_warnings
        except OSError as error:
            print(f"Debug de cones desativado: {error}", flush=True)
    poses = None
    if mode == "cones":
        poses = PoseHistory()
        detector.odometry = poses  # a visao leva o mapa de cones pela odometria
    vision_data = {"observation": None}
    stop_event = threading.Event()
    frames = queue.Queue(maxsize=1)
    thread_vision = None
    thread_capture = None
    plot = None
    observation = None
    control_status = "Missao ainda nao iniciada"
    last_control_at = None
    try:
        car.start_mission()
        print(
            f"Stanley {mode} | {config['ts']:.0f} s | teto {config['velocity_reference']:.2f} m/s",
            flush=True,
        )
        print(f"Encoder: protocolo {config['encoder_protocol']}", flush=True)
        print(
            (
                "Stanley: velocidade de referencia (vref)"
                if config["stanley_use_vref"]
                else "Stanley: velocidade medida (encoder)"
            ),
            flush=True,
        )
        print(
            f"Direcao: filtro de {config['steering_filter_size']} amostras", flush=True
        )
        thread_capture = threading.Thread(
            target=capture_func,
            args=(car, frames, stop_event, clock),
            daemon=True,
            name="captura-camera",
        )
        thread_capture.start()
        thread_vision = threading.Thread(
            target=vision_func,
            args=(car, vision_data, stop_event, detector, clock),
            kwargs={"frames": frames},
            daemon=True,
            name="linha-hsv",
        )
        thread_vision.start()
        if stream is not None:
            stream.start()
            print(
                f"Camera ao vivo: http://{socket.gethostname()}.local:{stream.port}/"
                f" (ou http://<ip-do-carro>:{stream.port}/)",
                flush=True,
            )
        if display:
            import matplotlib.pyplot as plt

            plt.ion()
            figure, axis = plt.subplots()
            plot = (plt, figure, axis)
        t_plot = clock()
        t_telemetry = t_plot - TELEMETRY_PERIOD
        obstacle_detected = False
        next_stop_beep = 0.0
        last_encoder_valid = None
        t_diagnostic = t_plot - 1.0
        last_diagnostic_sequence = 0
        while car.t < config["ts"]:
            if not car.step():
                break
            if poses is not None:
                try:
                    poses.append(clock(), float(car.p[0]), float(car.p[1]), float(car.th))
                except (TypeError, ValueError, IndexError):
                    pass
            encoder_valid = bool(getattr(car, "encoder_valid", True))
            if encoder_valid != last_encoder_valid:
                print(
                    (
                        "Encoder: leitura OK."
                        if encoder_valid
                        else "Encoder sem leitura valida: comandos em neutro. Confira o protocolo e a serial."
                    ),
                    flush=True,
                )
                last_encoder_valid = encoder_valid
            dist, valid = car.get_distance()
            stop_distance = stopping_distance(car.v)
            if (
                not obstacle_detected
                and valid
                and np.isfinite(dist)
                and dist <= stop_distance
            ):
                obstacle_detected = True
                print(
                    f"Obstaculo a {dist:.2f} m: parada travada ate o fim do ensaio.",
                    flush=True,
                )
            observation = vision_data["observation"]
            manual = (
                getattr(car, "sel_direcao", 1) == 0
                and getattr(car, "sel_tracao", 1) == 0
            )
            if (
                manual
                and encoder_valid
                and not car.emergencia
                and not obstacle_detected
                and np.isfinite(car.v)
                and np.isfinite(car.dt)
                and car.dt > 0
            ):
                # Car aplica os dois canais RC; a perda da faixa nao impede
                # controle manual. Obstaculos e emergencia conservam prioridade.
                follower.reset()
                car.set_steer(0.0)
                car.set_vel(0.0)
                car.log(lateral_active=0)
                control_status = "Controle manual RC"
            else:
                follower.apply(
                    car, observation, clock(), obstacle_detected=obstacle_detected
                )
                control_status = follower.status
            if mode == "cones" and observation is not None:
                debug = getattr(observation, "debug", None) or {}
                sides = debug.get("sides") or {}
                trajectory = debug.get("trajectory") or {}
                segment = {"reta": 0, "curva_esq": 1, "curva_dir": -1}.get(trajectory.get("segment"))
                car.log(navigation_cones=int(observation.reference_source == "cones"),
                        cone_red_count=len(getattr(observation, "red_pixels", ())),
                        cone_green_count=len(getattr(observation, "green_pixels", ())),
                        corridor_width_m=getattr(observation, "corridor_width_m", float("nan")),
                        cone_green_left=float(sides["green_left"]) if "green_left" in sides else float("nan"),
                        cone_estimated=float(bool(debug.get("estimated"))),
                        cone_branch=float(bool(debug.get("branch"))),
                        cone_segment=float("nan") if segment is None else float(segment))
            if recorder is not None and observation is not None:
                recorder.record(observation, dict(
                    {"active": False} if manual else follower.last_state, status=control_status, manual=bool(manual),
                    obstacle=bool(obstacle_detected), t=float(car.t), v=float(car.v),
                    vref=float(car.vref), steer_cmd_rad=float(car.st)))
            car.log(
                obstacle_stop=int(obstacle_detected),
                distance_m=dist,
                stopping_distance_m=stop_distance,
            )
            if stream is not None:
                stream.publish(observation, stream_text(car, control_status, observation))
            now = clock()
            last_control_at = now
            if not thread_vision.is_alive():
                stop_vehicle(car)
                print("Thread de visao encerrada; missao interrompida.", flush=True)
                break
            if now - t_diagnostic >= 1.0:
                timing = ""
                sequence = 0 if observation is None else observation.sequence
                vision_fps = (sequence - last_diagnostic_sequence) / (
                    now - t_diagnostic
                )
                if observation is not None:
                    timing = (
                        f" | imagem={now-observation.timestamp:.3f}s"
                        f" captura={observation.capture_seconds:.3f}s"
                        f" processamento={observation.processing_seconds:.3f}s"
                        f" deteccao={observation.detection_seconds:.3f}s"
                        f" geometria={observation.geometry_seconds:.3f}s"
                    )
                    if observation.processing_size:
                        timing += f" grade={observation.processing_size[0]}x{observation.processing_size[1]}"
                print(
                    f"CONTROLE: {control_status}{timing} visao={vision_fps:.1f}fps"
                    f" | vref={car.vref:.2f} u={car.u:.2f}",
                    flush=True,
                )
                t_diagnostic = now
                last_diagnostic_sequence = sequence
            if obstacle_detected and now >= next_stop_beep:
                if car.bz.beep([0.15, 0.15], silence=0.15):
                    next_stop_beep = now + 1.0
            # O controle roda em todo passo; somente a telemetria usa 10 Hz.
            if now - t_telemetry >= TELEMETRY_PERIOD:
                print(
                    f"DATA,{car.t:.3f},{car.p[0]:.3f},{car.p[1]:.3f},"
                    f"{car.v:.3f},{car.vref:.3f},{car.a:.3f},{car.u:.3f},"
                    f"{car.w:.3f},{car.th:.3f}",
                    flush=True,
                )
                t_telemetry = now
            if plot is not None and now - t_plot >= 1.0:
                plt, _, axis = plot
                preview = detector.with_preview(observation)
                if preview is not None and preview.frame is not None:
                    axis.clear()
                    axis.imshow(preview.frame)
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
            if stream is not None:
                stream.close()
            if recorder is not None:
                recorder.close()
            if thread_vision is not None:
                thread_vision.join(timeout=1.0)
            if thread_capture is not None:
                thread_capture.join(timeout=1.0)
            try:
                if config["save"]:
                    save_vision_snapshot(
                        car,
                        detector.with_preview(observation),
                        control_status,
                        last_control_at,
                    )
                if config["save"] and getattr(car, "traj", None):
                    car.save()
            finally:
                car.close()
                if plot is not None:
                    plot[0].close(plot[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", help="JSON da camera calibrada em metros")
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument(
        "--speed",
        type=float,
        default=MAIN_VEL,
        help=f"Teto de velocidade em m/s (padrao {MAIN_VEL})",
    )
    speed.add_argument(
        "--fast",
        action="store_const",
        dest="speed",
        const=FAST_VEL,
        help=f"Atalho para --speed {FAST_VEL}",
    )
    parser.add_argument(
        "--encoder-protocol",
        choices=("rpm", "rc"),
        help="rpm: Arduino original; rc: Arduino com receptor e chaves",
    )
    parser.add_argument(
        "--display", action="store_true", help="Mostrar a faixa detectada"
    )
    parser.add_argument(
        "--vision-width",
        type=int,
        help="Largura do processamento: 0 original, 320 ou 480 para reduzir custo",
    )
    parser.add_argument(
        "--steering-filter",
        type=int,
        help="Amostras do filtro da direcao: padrao 3; 1 desativa a media movel",
    )
    parser.add_argument(
        "--loss-frames",
        type=int,
        metavar="N",
        help="Quadros seguidos sem faixa/cones em que segue a ultima trajetoria (padrao 8 linha, 6 cones)",
    )
    parser.add_argument(
        "--stream",
        type=int,
        nargs="?",
        const=8080,
        metavar="PORTA",
        help="Transmitir a visao ao vivo por HTTP (padrao 8080), para o navegador do notebook",
    )
    parser.add_argument(
        "--record-frames",
        type=float,
        metavar="HZ",
        help="Coleta: grava quadros originais em logs/<ensaio>/frames/ a HZ (ex.: 5)",
    )
    parser.add_argument(
        "--no-cone-debug",
        action="store_true",
        help="Nao gravar cones_debug.jsonl nem frames de eventos",
    )
    parser.add_argument("--navigation", choices=("line", "cones"),
                        help="cones: somente cones (padrao); line: somente linha amarela")
    args = parser.parse_args()
    settings = {"velocity_reference": args.speed}
    if args.navigation is not None:
        settings["navigation_mode"] = args.navigation
    if args.vision_width is not None:
        settings["vision_width"] = args.vision_width
    if args.steering_filter is not None:
        settings["steering_filter_size"] = args.steering_filter
    if args.stream is not None:
        settings["stream_port"] = args.stream
    if args.loss_frames is not None:
        settings["line_loss_frames"] = args.loss_frames
        settings["cone_loss_frames"] = args.loss_frames
    if args.no_cone_debug:
        settings["cone_debug"] = False
    if args.record_frames is not None:
        if args.record_frames < 0:
            parser.error("--record-frames deve ser >= 0")
        settings["record_frames_hz"] = args.record_frames
    if args.encoder_protocol:
        settings["encoder_protocol"] = args.encoder_protocol
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
            parser.exit(
                2,
                f"{error}\nCalibre com veiculo_real/tools/calibrar_camera_lateral.py antes de rodar.\n",
            )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
