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
# Horario do driver aceito se o quadro chegou ate isto antes do fim do read().
MAX_DRIVER_DELAY_S = 0.5


def frame_time(car, read_started, read_ended):
    """(horario da foto, atraso ate o fim da leitura, origem).

    Antes de 24/11 o horario era o de ANTES do read(), que espera o proximo
    quadro (~38 ms nos logs de 19/11): a idade da imagem incluia essa espera, e
    nao o tempo real desde a foto. A camera USB marca cada quadro (V4L2,
    mesmo relogio de time.monotonic); fora do plausivel, volta ao antigo."""
    stamp = getattr(car, "frame_time", None)
    if (isinstance(stamp, (int, float)) and np.isfinite(stamp)
            and 0.0 <= read_ended - stamp <= MAX_DRIVER_DELAY_S):
        return float(stamp), read_ended - float(stamp), "v4l2"
    return read_started, read_ended - read_started, "leitura"


def capture_func(car, frames, stop_event, clock=time.monotonic):
    """Captura continuamente; a fila guarda somente o frame mais recente."""
    while not stop_event.is_set():
        read_started = clock()
        error = None
        try:
            frame = car.get_image(gray=False)
        except Exception as failure:
            frame, error = None, str(failure)
        captured_at, capture_seconds, source = frame_time(car, read_started, clock())
        sample = (frame, captured_at, capture_seconds, error, source)
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
                frame, captured_at, capture_seconds, capture_error, source = frames.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue
        else:
            captured_at = clock()
            capture_seconds, capture_error, source = float("nan"), None, "leitura"
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
                capture_clock=source,
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


class SharedFrames:
    """Anel de quadros BGR em memoria compartilhada entre os processos.

    Mandar o quadro (640x480x3 = 900 KB) dentro da observacao custava 2,9 ms
    no notebook so no pipe (0,4 sem ele), mais no Pi, com o envio travando o
    processo de visao ate o controle ler e a copia disputando o GIL do laco de
    controle. Aqui o filho copia o quadro numa vaga e manda so (vaga, numero);
    o principal copia de volta. Trava entre processos: no ARM a ordem das
    escritas nao e garantida sem ela (como SharedPoseHistory)."""
    def __init__(self, shape, slots=4, context=None):
        context = context or multiprocessing.get_context("spawn")
        self.shape = tuple(int(v) for v in shape)
        self.slots = int(slots)
        self.size = int(np.prod(self.shape))
        self.buffer = context.RawArray("B", self.size * self.slots)
        self.numbers = context.RawArray("q", self.slots)
        self.lock = context.Lock()

    def _view(self, slot):
        return np.frombuffer(self.buffer, np.uint8, self.size, slot * self.size).reshape(self.shape)

    def put(self, frame, number):
        """Guarda o quadro; devolve a vaga, ou None se o formato nao cabe."""
        if not isinstance(frame, np.ndarray) or frame.shape != self.shape or frame.dtype != np.uint8:
            return None
        slot = int(number) % self.slots
        with self.lock:
            self._view(slot)[...] = frame
            self.numbers[slot] = int(number)
        return slot

    def get(self, slot, number):
        """Copia do quadro `number`, ou None se a vaga ja foi reescrita."""
        with self.lock:
            if self.numbers[slot] != number:
                return None
            return self._view(slot).copy()


def _frame_shape(calibration_file):
    """(altura, largura, 3) da calibracao, ou None."""
    try:
        from .lateral import GroundCalibration

        width, height = GroundCalibration.from_file(calibration_file).image_size
        return int(height), int(width), 3
    except Exception:
        return None


class _Sender:
    """Faz o papel de vision_data no processo filho: cada quadro vai ao pipe
    (a imagem, pelo anel compartilhado quando ha um)."""
    def __init__(self, connection, frames=None):
        self.connection = connection
        self.frames = frames
        self.number = 0

    def __setitem__(self, key, observation):
        observation = _compact(observation)
        reference = None
        if self.frames is not None and observation.source_frame is not None:
            self.number += 1
            slot = self.frames.put(observation.source_frame, self.number)
            if slot is not None:
                reference = (slot, self.number)
                observation = replace(observation, source_frame=None)
        self.connection.send(("obs", observation, reference))


def _vision_main(connection, stop, mode, calibration_file, detector_kwargs, camera_settings, poses,
                 camera_factory=None, shared_frames=None, cpu_cores=None):
    """Processo filho: abre a camera, captura em thread e detecta."""
    # Ctrl+C chega ao grupo inteiro: so o processo principal decide parar.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Nucleos proprios antes de criar threads (elas herdam a afinidade).
    from .system_health import pin_process

    cores = pin_process(cpu_cores) if cpu_cores else None
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
    connection.send(("ready", cores))
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
        vision_func(camera, _Sender(connection, shared_frames), halt, detector, frames=frames)
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
                 camera_settings=None, poses=None, start_timeout=30., camera_factory=None, cpu_cores=None):
        """camera_factory: funcao de modulo (picklavel) que cria a camera; padrao Camera."""
        context = multiprocessing.get_context("spawn")
        self.vision_data = vision_data
        self.start_timeout = float(start_timeout)
        self._receiver_end, sender_end = context.Pipe(duplex=False)
        self._stop = context.Event()
        shape = _frame_shape(calibration_file)
        self.shared_frames = SharedFrames(shape, context=context) if shape is not None else None
        self.process = context.Process(
            target=_vision_main, name="visao",
            args=(sender_end, self._stop, mode, str(calibration_file), dict(detector_kwargs or {}),
                  dict(camera_settings or {}), poses, camera_factory, self.shared_frames,
                  list(cpu_cores) if cpu_cores else None),
            daemon=True)
        self._sender_end = sender_end
        # Process.start() descarta os args: sem esta referencia a trava do
        # historico seria coletada (e o semaforo apagado) antes do filho abrir.
        self.poses = poses
        self.receiver = None
        self.frames_received = 0
        self.frames_lost = 0
        self.cpu_cores = None  # nucleos aplicados no filho (None: sem fixar)

    def start(self):
        self.process.start()
        self._sender_end.close()  # so o filho escreve; EOF chega quando ele sair
        if not self._receiver_end.poll(self.start_timeout):
            self.stop()
            raise RuntimeError(f"Processo de visao nao respondeu em {self.start_timeout:.0f} s")
        try:
            kind, message = self._receiver_end.recv()[:2]
        except EOFError:
            kind, message = "error", "processo encerrou ao iniciar"
        if kind != "ready":
            self.stop()
            raise RuntimeError(f"Camera/visao: {message}")
        self.cpu_cores = message
        self.receiver = threading.Thread(target=self._receive, daemon=True, name="visao-recebe")
        self.receiver.start()
        return self

    def _receive(self):
        while True:
            try:
                kind, observation, *extra = self._receiver_end.recv()
            except (EOFError, OSError):
                return
            if kind == "obs":
                reference = extra[0] if extra else None
                if reference is not None and self.shared_frames is not None:
                    frame = self.shared_frames.get(*reference)
                    if frame is None:
                        self.frames_lost += 1  # vaga reescrita: segue sem a imagem
                    observation = replace(observation, source_frame=frame)
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
