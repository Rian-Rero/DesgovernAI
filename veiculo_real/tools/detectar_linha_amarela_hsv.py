#!/usr/bin/env python3
"""Interface de camera USB para acompanhar uma faixa amarela em retas e curvas.

Execute: venv/bin/python veiculo_real/tools/detectar_linha_amarela_hsv.py
Dependencias: numpy, pillow, opencv-python OU opencv-python-headless, tkinter.
Mantenha yellow_path.py ao lado deste arquivo. Consulte README_linha_hsv.md.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

from yellow_path import PathSettings, YellowPathDetector, render_path


def camera_source(value):
    return int(value) if value.isdecimal() else value


def available_cameras():
    """Lista dispositivos sem abri-los. Em outras plataformas aceita indices."""
    cameras = {}
    if sys.platform.startswith("linux"):
        devices = sorted(Path("/dev").glob("video[0-9]*"),
                         key=lambda p: int(p.name.removeprefix("video")))
        for device in devices:
            try:
                name = (Path("/sys/class/video4linux") / device.name / "name").read_text().strip()
            except OSError:
                name = "Camera USB / video"
            cameras[f"{device}  ·  {name}"] = str(device)
    if not cameras:
        cameras = {f"Camera {index}": index for index in range(6)}
    return cameras


def open_camera(source, width, height, fps):
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if sys.platform.startswith("linux") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(source, backend)
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(10):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        return cap, frame
        except cv2.error:
            pass
        cap.release()
    raise RuntimeError(f"Sem imagem da camera {source!r}. Confira a conexao e escolha outro dispositivo.")


class CaptureThread(threading.Thread):
    """Apenas esta thread possui a camera; a GUI recebe o frame mais novo."""
    def __init__(self, width, height, fps):
        super().__init__(daemon=True, name="captura-usb")
        self.width, self.height, self.fps = width, height, fps
        self.commands = queue.Queue()
        self.events = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.latest = None

    def request(self, action, generation, source=None):
        self.commands.put((action, generation, source))

    def take_frame(self):
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def publish(self, generation, frame):
        with self.lock:
            self.latest = generation, frame, time.monotonic()

    def run(self):
        cap, generation, failures = None, 0, 0
        try:
            while True:
                command = None
                try:
                    command = self.commands.get(timeout=.05) if cap is None else self.commands.get_nowait()
                    while True:
                        command = self.commands.get_nowait()
                except queue.Empty:
                    pass
                if command is not None:
                    action, generation, source = command
                    if cap is not None:
                        cap.release()
                        cap = None
                    with self.lock:
                        self.latest = None
                    if action == "quit":
                        return
                    if action == "stop":
                        self.events.put((generation, "stopped", "Camera desconectada"))
                        continue
                    try:
                        cap, frame = open_camera(source, self.width, self.height, self.fps)
                        failures = 0
                        self.events.put((generation, "ready", f"{frame.shape[1]} × {frame.shape[0]}"))
                        self.publish(generation, frame)
                    except (RuntimeError, cv2.error) as error:
                        self.events.put((generation, "error", str(error)))
                    continue
                if cap is None:
                    continue
                try:
                    ok, frame = cap.read()
                except cv2.error:
                    ok, frame = False, None
                if ok and frame is not None and frame.size:
                    failures = 0
                    self.publish(generation, frame)
                else:
                    failures += 1
                    if failures >= 10:
                        cap.release()
                        cap = None
                        self.events.put((generation, "error", "Sinal da camera perdido. Reconecte o dispositivo."))
                    else:
                        time.sleep(.02)
        finally:
            if cap is not None:
                cap.release()


class CameraApp:
    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        self.tk, self.ttk, self.Image, self.ImageTk = tk, ttk, Image, ImageTk
        self.root = root
        self.detector = YellowPathDetector()
        self.worker = CaptureThread(args.width, args.height, args.fps)
        self.generation = 0
        self.closed = False
        self.streaming = False
        self.last_frame = self.last_result = None
        self.photo = self.display_bounds = None
        self.previous_time = self.last_arrival = None
        self.fps = 0.
        self.state = tk.StringVar(value="Escolha a camera e clique em Conectar")
        self.detail = tk.StringVar(value="Pronto para iniciar")
        self.fps_text = tk.StringVar(value="— FPS")
        self.support_text = tk.StringVar(value="— suporte de cor")
        self.view = tk.StringVar(value="Destaque")
        self.camera_var = tk.StringVar()
        self.variables = {key: (tk.BooleanVar(value=value) if isinstance(value, bool)
                               else tk.IntVar(value=value)) for key, value in asdict(PathSettings()).items()}
        root.title("Faixa amarela | Visao do carrinho")
        root.geometry("1240x820")
        root.minsize(1000, 720)
        root.configure(bg="#0c1421")
        self._style()
        self._build()
        self.refresh_cameras()
        selected = next((label for label, value in self.cameras.items()
                         if value == args.camera or value == f"/dev/video{args.camera}"), None)
        self.camera_var.set(selected if selected is not None else str(args.camera))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.close())
        self.worker.start()
        self.root.after(30, self.tick)
        self.root.after(150, self.connect)

    def _style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#0c1421", foreground="#e6edf6", font=("DejaVu Sans", 10))
        style.configure("TButton", background="#25394f", padding=(13, 9), borderwidth=0)
        style.map("TButton", background=[("active", "#34516f")])
        style.configure("Accent.TButton", background="#137f75", foreground="white")
        style.map("Accent.TButton", background=[("active", "#15988a")])
        style.configure("TNotebook", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(13, 9), background="#1b2a3c")
        style.map("TNotebook.Tab", background=[("selected", "#29435a")])
        style.configure("TCombobox", fieldbackground="#e6edf6", foreground="#142033", padding=7)
        style.map("TCombobox", fieldbackground=[("readonly", "#e6edf6")], foreground=[("readonly", "#142033")])
        style.configure("TRadiobutton", padding=5)
        style.configure("TCheckbutton", padding=5)
        style.configure("TScale", troughcolor="#263b51")

    def _build(self):
        tk, ttk = self.tk, self.ttk
        header = ttk.Frame(self.root, padding=(22, 18, 22, 10))
        header.pack(fill="x")
        ttk.Label(header, text="FAIXA AMARELA", font=("DejaVu Sans", 21, "bold")).pack(anchor="w")
        ttk.Label(header, text="Uma trajetoria continua para retas e curvas", foreground="#9bb1c9").pack(anchor="w", pady=(3, 0))
        toolbar = ttk.Frame(self.root, padding=(22, 4, 22, 12))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Camera").pack(side="left", padx=(0, 9))
        self.camera_combo = ttk.Combobox(toolbar, textvariable=self.camera_var, width=37)
        self.camera_combo.pack(side="left", fill="x", expand=True)
        self.camera_combo.bind("<Return>", lambda _: self.connect())
        ttk.Button(toolbar, text="Atualizar lista", command=self.refresh_cameras).pack(side="left", padx=7)
        ttk.Button(toolbar, text="Conectar / trocar", style="Accent.TButton", command=self.connect).pack(side="left")
        ttk.Button(toolbar, text="Desconectar", command=self.disconnect).pack(side="left", padx=(7, 0))
        body = ttk.Frame(self.root, padding=(22, 0, 22, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        visual = ttk.Frame(body)
        visual.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        modes = ttk.Frame(visual)
        modes.pack(fill="x", pady=(0, 8))
        for text in ("Destaque", "Original", "HSV", "Caminho"):
            ttk.Radiobutton(modes, text=text, value=text, variable=self.view, command=self.redraw).pack(side="left")
        self.canvas = tk.Canvas(visual, bg="#080e18", highlightthickness=1, highlightbackground="#25394f")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        self.canvas.bind("<Button-1>", self.select_path)
        ttk.Label(visual, text="Clique perto da base da faixa desejada para selecionar o caminho.",
                  foreground="#9bb1c9", wraplength=650).pack(anchor="w", pady=(9, 3))
        ttk.Label(visual, text="Verde: cor observada   •   Laranja: falha reconstruida   •   Branco: centro",
                  foreground="#9bb1c9", wraplength=650).pack(anchor="w")
        sidebar = ttk.Frame(body, width=300)
        sidebar.grid(row=0, column=1, sticky="ns")
        notebook = ttk.Notebook(sidebar, width=300)
        notebook.pack(fill="both", expand=True)
        geometry, color = ttk.Frame(notebook, padding=12), ttk.Frame(notebook, padding=12)
        notebook.add(geometry, text="Caminho")
        notebook.add(color, text="Cor e luz")
        self.slider(geometry, "roi_top", "Ignorar topo", 0, 65, "%", "Recorta objetos fora da pista.")
        self.slider(geometry, "max_gap", "Conectar falhas", 0, 90, "px", "Limite vertical em imagem de 480 px.")
        self.slider(geometry, "max_width", "Largura maxima", 3, 40, "%", "Reduza para rejeitar objetos largos.")
        self.slider(geometry, "min_span", "Extensao minima", 12, 50, "%", "Aumente para rejeitar manchas curtas.")
        self.slider(geometry, "smoothing", "Estabilidade temporal", 0, 80, "%", "Suaviza somente uma faixa observada.")
        ttk.Button(geometry, text="Rebuscar no centro", command=self.recenter).pack(fill="x", pady=(12, 4))
        ttk.Label(geometry, text="Falhas longas interrompem a deteccao. O caminho acompanha curvas sem exigir uma reta.",
                  wraplength=270, foreground="#9bb1c9").pack(anchor="w", pady=8)
        self.slider(color, "h_min", "Matiz minimo (H)", 0, 179, "")
        self.slider(color, "h_max", "Matiz maximo (H)", 0, 179, "")
        self.slider(color, "s_min", "Saturacao minima (S)", 25, 255, "")
        self.slider(color, "v_min", "Brilho minimo (V)", 15, 255, "")
        ttk.Checkbutton(color, text="Compensar brilho local (CLAHE)", variable=self.variables["clahe"],
                        command=self.changed).pack(anchor="w", pady=8)
        ttk.Label(color, text="Tinta desbotada: reduza S. Sombras: reduza V. A recuperacao tolerante so vale dentro de uma trajetoria aceita.",
                  foreground="#9bb1c9", wraplength=270).pack(anchor="w", pady=8)
        ttk.Button(sidebar, text="Restaurar todos os ajustes", command=self.restore).pack(fill="x", pady=(12, 0))
        footer = ttk.Frame(self.root, padding=(22, 14, 22, 16))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.state, foreground="#77e2c6", font=("DejaVu Sans", 11, "bold")).pack(anchor="w")
        row = ttk.Frame(footer)
        row.pack(fill="x", pady=(5, 0))
        ttk.Label(row, textvariable=self.detail, foreground="#9bb1c9").pack(side="left", fill="x", expand=True)
        ttk.Label(row, textvariable=self.fps_text).pack(side="left", padx=14)
        ttk.Label(row, textvariable=self.support_text).pack(side="left")

    def slider(self, parent, key, label, low, high, suffix, hint=None):
        row = self.ttk.Frame(parent)
        row.pack(fill="x", pady=(9, 0))
        self.ttk.Label(row, text=label).pack(side="left")
        value_label = self.ttk.Label(row, foreground="#77e2c6", width=7, anchor="e")
        value_label.pack(side="right")
        variable = self.variables[key]

        def update(value):
            variable.set(round(float(value)))
            self.changed()

        self.ttk.Scale(parent, from_=low, to=high, variable=variable, command=update).pack(fill="x", pady=(2, 0))
        variable.trace_add("write", lambda *_: value_label.configure(text=f"{variable.get()}{suffix}"))
        value_label.configure(text=f"{variable.get()}{suffix}")
        if hint:
            self.ttk.Label(parent, text=hint, foreground="#9bb1c9", font=("DejaVu Sans", 9),
                           wraplength=270).pack(anchor="w", pady=(1, 2))

    def settings(self):
        return PathSettings(**{name: var.get() for name, var in self.variables.items()})

    def refresh_cameras(self):
        self.cameras = available_cameras()
        self.camera_combo.configure(values=list(self.cameras))
        if not self.camera_var.get():
            self.camera_var.set(next(iter(self.cameras)))

    def clear_stream(self):
        self.detector.reset()
        self.last_frame = self.last_result = None
        self.last_arrival = self.previous_time = None
        self.streaming = False
        self.fps = 0.
        self.fps_text.set("— FPS")
        self.support_text.set("— suporte de cor")
        self.redraw()

    def connect(self):
        selected = self.camera_var.get().strip()
        if not selected:
            self.state.set("Selecione uma camera ou digite seu indice")
            return
        source = self.cameras.get(selected, camera_source(selected))
        self.generation += 1
        self.clear_stream()
        self.state.set("Conectando camera…")
        self.detail.set(str(source))
        self.worker.request("connect", self.generation, source)

    def disconnect(self):
        self.generation += 1
        self.clear_stream()
        self.state.set("Camera desconectada")
        self.detail.set("Escolha um dispositivo e conecte novamente")
        self.worker.request("stop", self.generation)

    def changed(self):
        self.detector.reset()
        if self.last_frame is not None:
            self.last_result = self.detector.process(self.last_frame, self.settings())
            self.update_status()
            self.redraw()

    def restore(self):
        for name, value in asdict(PathSettings()).items():
            self.variables[name].set(value)
        self.detector.reset(anchor=.5)
        self.changed()

    def recenter(self):
        self.detector.reset(anchor=.5)
        self.changed()

    def select_path(self, event):
        if self.display_bounds is None or self.view.get() not in ("Destaque", "Original"):
            return
        left, top, width, height = self.display_bounds
        if left <= event.x < left + width and top <= event.y < top + height:
            self.detector.reset(anchor=(event.x - left) / width)
            self.changed()

    def update_status(self):
        if self.last_result is not None:
            self.state.set(self.last_result.status)
            self.support_text.set(f"{self.last_result.support:.0%} suporte de cor" if self.last_result.valid else "Sem caminho valido")

    def tick(self):
        if self.closed:
            return
        try:
            while True:
                generation, event, text = self.worker.events.get_nowait()
                if generation != self.generation:
                    continue
                if event == "ready":
                    self.streaming = True
                    self.detail.set(f"{text}  |  Captura ativa")
                else:
                    self.clear_stream()
                    self.state.set(text)
                    self.detail.set("Escolha a camera e clique em Conectar / trocar")
        except queue.Empty:
            pass
        packet = self.worker.take_frame()
        if packet is not None and packet[0] == self.generation and self.streaming:
            _, frame, captured_at = packet
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])), interpolation=cv2.INTER_AREA)
            now = time.monotonic()
            self.last_arrival = now
            self.last_frame = frame
            self.last_result = self.detector.process(frame, self.settings(), timestamp=captured_at)
            if self.previous_time is not None:
                instantaneous = 1 / max(now - self.previous_time, 1e-6)
                self.fps = instantaneous if not self.fps else .9 * self.fps + .1 * instantaneous
                self.fps_text.set(f"{self.fps:.1f} FPS")
            self.previous_time = now
            self.update_status()
            self.redraw()
        elif self.streaming and self.last_arrival is not None and time.monotonic() - self.last_arrival > 1.:
            self.last_result = self.last_frame = None
            self.state.set("Aguardando novas imagens da camera…")
            self.support_text.set("Sem caminho valido")
            self.redraw()
        self.root.after(15, self.tick)

    def redraw(self):
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self.last_frame is None or self.last_result is None:
            self.display_bounds = None
            self.canvas.create_text(cw / 2, ch / 2, text="Selecione sua camera USB\ne clique em Conectar / trocar",
                                    fill="#9bb1c9", font=("DejaVu Sans", 14), justify="center")
            return
        mode, result = self.view.get(), self.last_result
        if mode == "Original":
            bgr = self.last_frame
        elif mode == "HSV":
            bgr = cv2.cvtColor(result.raw_mask, cv2.COLOR_GRAY2BGR)
        elif mode == "Caminho":
            bgr = np.zeros_like(self.last_frame)
            bgr[result.path_mask > 0] = (0, 165, 255)
            bgr[result.observed_mask > 0] = (70, 245, 30)
            if result.valid:
                cv2.polylines(bgr, [result.center], False, (255, 255, 255), 2)
        else:
            bgr = render_path(self.last_frame, result, self.settings(), self.detector.anchor)
        image = self.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        factor = min(cw / image.width, ch / image.height)
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        image = image.resize(size, self.Image.Resampling.BILINEAR)
        self.photo = self.ImageTk.PhotoImage(image)
        left, top = (cw - size[0]) // 2, (ch - size[1]) // 2
        self.display_bounds = left, top, *size
        self.canvas.create_image(left, top, anchor="nw", image=self.photo)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.worker.request("quit", self.generation + 1)
        self.worker.join(timeout=.3)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=camera_source, default=0, help="Camera inicial; pode ser alterada na interface")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("width, height e fps devem ser positivos")
    cv2.setNumThreads(2)
    try:
        import tkinter as tk
        from PIL import ImageTk  # noqa: F401
    except ImportError as error:
        print(f"Dependencia ausente: {error}. Instale pillow e o suporte tkinter do Python.", file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Nao foi possivel abrir a interface: {error}. Execute em uma sessao grafica do PC.", file=sys.stderr)
        return 1
    app = CameraApp(root, args)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
