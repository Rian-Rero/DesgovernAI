"""Registro por quadro dos cones, leve para o Raspberry Pi 4.

Escreve logs/<ensaio>/cones_debug.jsonl (um JSON por quadro de visao, juntando
percepcao e decisao do controle) e, somente em eventos, alguns JPGs reduzidos em
cone_frames/. Uma thread de escrita faz JSON, JPEG e disco. O laco de controle
apenas enfileira e nunca bloqueia: com a fila cheia, o registro e descartado e
contado. No fim, cones_summary.json resume a corrida.
"""
from collections import Counter, deque
import json
import math
from pathlib import Path
import queue
import threading

import cv2
import numpy as np

EVENTS = ("perda_cones", "reaquisicao", "parada", "salto", "troca_lado", "bifurcacao")


def clean(value):
    """Converte para JSON estrito: numpy -> python, NaN/inf -> None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 4) if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


class ConeDebugRecorder:
    def __init__(self, folder, *, max_event_frames=300, pre_frames=5, post_frames=5,
                 queue_size=64, frame_size=(320, 240), jpeg_quality=70, record_hz=0.):
        self.folder = Path(folder)
        self.frames_folder = self.folder / "cone_frames"
        # Coleta para rotular/avaliar: quadro original em resolucao cheia, a
        # record_hz quadros/s (0 desliga). Somente em ensaios de coleta.
        self.raw_folder = self.folder / "frames"
        self.record_period = 1. / record_hz if record_hz and record_hz > 0 else None
        self.last_raw_t = -math.inf
        self.raw_saved = 0
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = open(self.folder / "cones_debug.jsonl", "w", encoding="utf-8")
        self.queue = queue.Queue(maxsize=queue_size)
        self.max_event_frames = int(max_event_frames)
        self.post_frames = int(post_frames)
        self.frame_size = tuple(frame_size)
        self.jpeg_quality = int(jpeg_quality)
        self.recent = deque(maxlen=max(0, int(pre_frames)))  # referencias, sem codificar
        self.post_left = 0
        self.post_event = None
        self.frames_reserved = 0
        self.last_sequence = None
        self.previous = None  # estado do quadro anterior, para detectar eventos
        self.dropped = 0
        self.write_errors = 0
        self.stats = Counter()
        self.events = Counter()
        self.segments = Counter()
        self.invalid_streak = 0
        self.longest_invalid = 0
        self.first_t = self.last_t = None
        self.timing = {"detection": [0., 0], "processing": [0., 0]}  # soma, quadros
        # Indicadores da corrida (resumo): estado do estimador, largura, oscilacao
        # da vref e do esterco, correcao de luz e cones por quadro.
        self.corridor_states = Counter()
        self.series = {"sigma_e": [], "learned_width": [], "vref": [], "steer": [], "light": [],
                       "cones": [], "preview_cut": [], "pursuit": []}
        self.closed = False
        self.notes = {}  # Informacoes da corrida copiadas para o resumo (ex.: avisos).
        self.thread = threading.Thread(target=self._writer, daemon=True, name="cones-debug")
        self.thread.start()

    # ------------------------------------------------------------------ laco
    def record(self, observation, control):
        """Chamado pelo laco principal a cada passo; registra so quadros novos."""
        if self.closed or observation is None:
            return False
        sequence = getattr(observation, "sequence", 0)
        if sequence == self.last_sequence:
            return False
        self.last_sequence = sequence
        debug = getattr(observation, "debug", None) or {}
        valid = bool(observation.valid)
        state = {
            "valid": valid,
            "active": bool(control.get("active")),
            "hold": bool(control.get("hold")),
            "branch": bool(debug.get("branch")),
        }
        events = self._events(state, debug, control)
        self._update_stats(observation, state, debug, events)
        self._indicators(debug, control)
        self.previous = state
        record = {
            "frame": sequence,
            "t_capture": observation.timestamp,
            "events": events,
            "valid": valid,
            "reason": observation.reason,
            "timing": {
                "capture_s": observation.capture_seconds,
                "processing_s": observation.processing_seconds,
                "detection_s": observation.detection_seconds,
                "geometry_s": observation.geometry_seconds,
            },
            "control": control,
            "vision": debug,
        }
        self._put(("json", record))
        frame = getattr(observation, "source_frame", None)
        self._frames(sequence, frame, events)
        t = observation.timestamp
        if (self.record_period is not None and frame is not None
                and isinstance(t, (int, float)) and t - self.last_raw_t >= self.record_period - 1e-6):
            if self._put(("raw", sequence, frame)):
                self.last_raw_t = t
                self.raw_saved += 1
        return True

    def _events(self, state, debug, control):
        previous = self.previous
        events = []
        if previous is not None:
            if previous["valid"] and not state["valid"]:
                events.append("perda_cones")
            if not previous["valid"] and state["valid"]:
                events.append("reaquisicao")
            if previous["active"] and not state["active"]:
                events.append("parada")
            if state["branch"] and not previous["branch"]:
                events.append("bifurcacao")
        if control.get("outlier"):
            events.append("salto")
        if (debug.get("sides") or {}).get("flipped"):
            events.append("troca_lado")
        return events

    def _update_stats(self, observation, state, debug, events):
        self.stats["frames"] += 1
        self.stats["valid"] += state["valid"]
        self.stats["estimated"] += bool(state["valid"] and debug.get("estimated"))
        self.stats["hold"] += state["hold"]
        self.stats["active"] += state["active"]
        self.stats["branch"] += state["branch"]
        self.events.update(events)
        segment = (debug.get("trajectory") or {}).get("segment")
        if state["valid"] and segment:
            self.segments[segment] += 1
        self.invalid_streak = 0 if state["valid"] else self.invalid_streak + 1
        self.longest_invalid = max(self.longest_invalid, self.invalid_streak)
        t = observation.timestamp
        if isinstance(t, (int, float)) and math.isfinite(t):
            self.first_t = t if self.first_t is None else self.first_t
            self.last_t = t
        for key, value in (("detection", observation.detection_seconds),
                           ("processing", observation.processing_seconds)):
            if isinstance(value, (int, float)) and math.isfinite(value):
                self.timing[key][0] += value
                self.timing[key][1] += 1

    def _frames(self, sequence, frame, events):
        if frame is None:
            return
        if events:
            name = events[0]
            for old_sequence, old_frame in self.recent:
                self._frame(old_sequence, old_frame, name + "_antes")
            self.recent.clear()
            self._frame(sequence, frame, name)
            self.post_left, self.post_event = self.post_frames, name
        elif self.post_left > 0:
            self._frame(sequence, frame, self.post_event + "_depois")
            self.post_left -= 1
        else:
            self.recent.append((sequence, frame))

    def _frame(self, sequence, frame, name):
        if self.frames_reserved >= self.max_event_frames:
            return
        if self._put(("frame", sequence, frame, name)):
            self.frames_reserved += 1

    def _put(self, item):
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    # ---------------------------------------------------------------- escrita
    def _writer(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            try:
                if item[0] == "json":
                    self.file.write(json.dumps(clean(item[1]), separators=(",", ":")) + "\n")
                elif item[0] == "raw":
                    _, sequence, frame = item
                    self.raw_folder.mkdir(exist_ok=True)
                    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        (self.raw_folder / f"f{sequence:06d}.jpg").write_bytes(data.tobytes())
                else:
                    _, sequence, frame, name = item
                    self.frames_folder.mkdir(exist_ok=True)
                    small = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
                    ok, data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if ok:
                        (self.frames_folder / f"f{sequence:06d}_{name}.jpg").write_bytes(data.tobytes())
            except (OSError, ValueError, TypeError, cv2.error):
                self.write_errors += 1
        self.file.flush()

    def _indicators(self, debug, control):
        finite = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        corridor = debug.get("corridor") or {}
        if corridor.get("state"):
            self.corridor_states[corridor["state"]] += 1
            sigma = corridor.get("sigma") or [None]
            if finite(sigma[0]):
                self.series["sigma_e"].append(sigma[0])
            if finite(corridor.get("learned_width_m")):
                self.series["learned_width"].append(corridor["learned_width_m"])
        detection = debug.get("detection") or {}
        light = detection.get("light")
        if light and all(finite(v) for v in light):
            self.series["light"].append(math.exp(sum(math.log(max(v, 1e-6)) for v in light)/len(light)))
        cones = sum(int((detection.get(c) or {}).get("bases", 0) or 0) for c in ("green", "red"))
        self.series["cones"].append(cones)
        if control.get("active"):
            for key, name in (("vref", "vref"), ("steer", "steer_cmd_rad")):
                if finite(control.get(name)):
                    self.series[key].append(control[name])
            preview = control.get("preview")
            target = control.get("speed_target")
            if preview and finite(target):
                self.series["preview_cut"].append(preview[0] <= target + 1e-6 and preview[0] < control.get("vref", 0) + 1e-6)
            self.series["pursuit"].append(control.get("steering_law") == "pure_pursuit")

    def _indicator_summary(self):
        import numpy as np

        series = self.series
        stat = lambda values, f: None if not values else float(f(np.asarray(values, float)))
        diff = lambda values: None if len(values) < 3 else float(np.median(np.abs(np.diff(values))))
        total = sum(self.corridor_states.values())
        return {
            "corridor_states": None if not total else {k: round(v/total, 3) for k, v in self.corridor_states.items()},
            "corridor_sigma_e_median_m": stat(series["sigma_e"], np.median),
            "learned_width_m": series["learned_width"][-1] if series["learned_width"] else None,
            # Variacao mediana quadro a quadro: vref aos trancos / direcao serrilhando.
            "vref_mean_m_s": stat(series["vref"], np.mean),
            "vref_step_median_m_s": diff(series["vref"]),
            "steer_step_median_rad": diff(series["steer"]),
            "steer_abs_p90_rad": stat(series["steer"], lambda v: np.percentile(np.abs(v), 90)),
            "preview_limited_fraction": stat(series["preview_cut"], np.mean),
            "pure_pursuit_fraction": stat(series["pursuit"], np.mean),
            # Ganho da normalizacao de luz (1 = nenhum); perto de 3,5: luz fraca demais.
            "light_gain_median": stat(series["light"], np.median),
            "light_gain_max": stat(series["light"], np.max),
            "cones_per_frame_median": stat(series["cones"], np.median),
        }

    def summary(self):
        duration = None if self.first_t is None else self.last_t - self.first_t
        frames = self.stats["frames"]
        return clean({
            "frames": frames,
            "valid": self.stats["valid"],
            "estimated": self.stats["estimated"],
            "hold": self.stats["hold"],
            "invalid": frames - self.stats["valid"],
            "active": self.stats["active"],
            "branch_frames": self.stats["branch"],
            "longest_invalid_frames": self.longest_invalid,
            "events": dict(self.events),
            "segments": dict(self.segments),
            "duration_s": duration,
            "vision_fps": frames / duration if duration else None,
            "mean_detection_s": self._mean("detection"),
            "mean_processing_s": self._mean("processing"),
            "event_frames_saved": self.frames_reserved,
            "raw_frames_saved": self.raw_saved,
            "dropped_records": self.dropped,
            "write_errors": self.write_errors,
            **self._indicator_summary(),
            **self.notes,
        })

    def _mean(self, key):
        total, count = self.timing[key]
        return total / count if count else None

    def close(self, timeout=2.):
        if self.closed:
            return
        self.closed = True
        try:
            self.queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        self.thread.join(timeout=timeout)
        try:
            self.file.close()
            (self.folder / "cones_summary.json").write_text(
                json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            print(f"Nao foi possivel salvar resumo dos cones: {error}", flush=True)
