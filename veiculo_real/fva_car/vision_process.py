"""Captura e visao fora do laco de controle: em threads ou num processo proprio.

No Raspberry Pi 4 a visao, o laco de controle a 50 Hz, a gravacao do debug e a
captura disputavam o GIL de um unico processo Python: o detector levava ~12 ms
no notebook e 115-150 ms no carro. VisionProcess roda camera e detector em
outro processo (outro interpretador, outro nucleo); o controle recebe so a
observacao pronta. capture_func/vision_func sao as mesmas nos dois modos.
"""
from dataclasses import replace
import multiprocessing
import os
import queue
import signal
import threading
import time

import numpy as np

from .lateral import LineObservation

# Mascaras so servem ao detector; o preview e o debug usam pontos e curvas.
_EMPTY_MASK = np.zeros((0, 0), np.uint8)


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



def _compact(observation):
    """Observacao sem as mascaras do detector (3 x 300 KB por quadro no pipe)."""
    result = getattr(observation, "cone_result", None)
    if result is None or not hasattr(result, "raw_mask"):
        return observation
    return replace(observation, cone_result=replace(
        result, raw_mask=_EMPTY_MASK, green_mask=_EMPTY_MASK, red_mask=_EMPTY_MASK))


def _build_detector(mode, calibration_file, detector_kwargs):
    from .lateral import GroundCalibration, RealLineDetector

    calibration = GroundCalibration.from_file(calibration_file)
    if mode == "cones":
        from .cones import RealConeDetector

        return RealConeDetector(calibration, render_preview=False, **detector_kwargs)
    return RealLineDetector(calibration, render_preview=False, **detector_kwargs)


class _Sender:
    """Faz o papel de vision_data no processo filho: cada quadro vai ao pipe."""
    def __init__(self, connection):
        self.connection = connection

    def __setitem__(self, key, observation):
        self.connection.send(("obs", _compact(observation)))


def _vision_main(connection, stop, mode, calibration_file, detector_kwargs, camera_settings, poses,
                 camera_factory=None):
    """Processo filho: abre a camera, captura em thread e detecta."""
    # Ctrl+C chega ao grupo inteiro: so o processo principal decide parar.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import cv2

    cv2.setNumThreads(2)
    parent = os.getppid()
    camera = None
    try:
        detector = _build_detector(mode, calibration_file, detector_kwargs)
        if poses is not None:
            detector.odometry = poses
        if camera_factory is None:
            from .camera import Camera

            camera_factory = Camera
        camera = camera_factory(**camera_settings)
    except Exception as error:  # informa antes de qualquer motor ser ligado
        connection.send(("error", f"{type(error).__name__}: {error}"))
        if camera is not None:
            camera.close()
        connection.close()
        return
    connection.send(("ready", None))
    halt = threading.Event()

    def watchdog():
        # Sai se o principal pedir ou morrer (camera nao fica presa a um orfao).
        while not halt.is_set():
            if stop.wait(.2) or os.getppid() != parent:
                halt.set()

    frames = queue.Queue(maxsize=1)
    threads = [threading.Thread(target=watchdog, daemon=True, name="visao-vigia"),
               threading.Thread(target=capture_func, args=(camera, frames, halt), daemon=True,
                                name="visao-captura")]
    for thread in threads:
        thread.start()
    try:
        vision_func(camera, _Sender(connection), halt, detector, frames=frames)
    except (BrokenPipeError, EOFError, OSError):
        pass  # principal encerrou
    finally:
        halt.set()
        for thread in threads:
            thread.join(timeout=1.)
        camera.close()
        connection.close()


class VisionProcess:
    """Camera + detector num processo separado; entrega observacoes em vision_data.

    Iniciar antes dos motores: start() espera a camera abrir e falha cedo.
    """
    def __init__(self, vision_data, mode, calibration_file, detector_kwargs=None,
                 camera_settings=None, poses=None, start_timeout=30., camera_factory=None):
        """camera_factory: funcao de modulo (picklavel) que cria a camera; padrao Camera."""
        context = multiprocessing.get_context("spawn")
        self.vision_data = vision_data
        self.start_timeout = float(start_timeout)
        self._receiver_end, sender_end = context.Pipe(duplex=False)
        self._stop = context.Event()
        self.process = context.Process(
            target=_vision_main, name="visao",
            args=(sender_end, self._stop, mode, str(calibration_file), dict(detector_kwargs or {}),
                  dict(camera_settings or {}), poses, camera_factory),
            daemon=True)
        self._sender_end = sender_end
        # Process.start() descarta os args: sem esta referencia a trava do
        # historico seria coletada (e o semaforo apagado) antes do filho abrir.
        self.poses = poses
        self.receiver = None
        self.frames_received = 0

    def start(self):
        self.process.start()
        self._sender_end.close()  # so o filho escreve; EOF chega quando ele sair
        if not self._receiver_end.poll(self.start_timeout):
            self.stop()
            raise RuntimeError(f"Processo de visao nao respondeu em {self.start_timeout:.0f} s")
        try:
            kind, message = self._receiver_end.recv()
        except EOFError:
            kind, message = "error", "processo encerrou ao iniciar"
        if kind != "ready":
            self.stop()
            raise RuntimeError(f"Camera/visao: {message}")
        self.receiver = threading.Thread(target=self._receive, daemon=True, name="visao-recebe")
        self.receiver.start()
        return self

    def _receive(self):
        while True:
            try:
                kind, observation = self._receiver_end.recv()
            except (EOFError, OSError):
                return
            if kind == "obs":
                self.frames_received += 1
                self.vision_data["observation"] = observation

    def is_alive(self):
        return (self.process.is_alive() and self.receiver is not None and self.receiver.is_alive())

    def stop(self, timeout=2.):
        self._stop.set()
        if self.process.pid is not None:
            self.process.join(timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(1.)
        if self.receiver is not None:
            self.receiver.join(timeout=1.)
        self._receiver_end.close()
