# -*- coding: utf-8 -*-
#!/usr/bin/env python3
########################################
# Disciplina: Topicos em Engenharia de Controle e Automacao IV (ENG075):
# Fundamentos de Veiculos Autonomos - 2026/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automacao
# DELT - Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
# GUI Tkinter para envio de arquivos e execução remota em múltiplas Raspberry Pis,
# com senha SSH padrão (DEFAULT_PASS) pré-preenchida no campo.
import hashlib
import os
import re
import posixpath
import shlex
import uuid
import platform
import subprocess
import threading
import stat
import time
import queue
import csv
import json
import base64
import webbrowser
import urllib.request
from collections import deque
from pathlib import Path

import paramiko
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

########################################
# Configurações (ajuste aqui)
########################################
SSH_USER = "alunos"
DEFAULT_PASS = "fva2023"  # <-- coloque aqui a senha padrão desejada, ex: "rasp123"
DEFAULT_DEST = "/home/alunos/Desktop/fva"
# m/s; nos cones e a velocidade MAXIMA NAS RETAS (o carro reduz sozinho antes
# das curvas e pelo limite de visibilidade, ~1,6 m/s com curva de R = 1,5 m).
SPEED_PRESETS = ("0.7", "1.0", "1.5", "2.0")
CURVE_RADII = ("1.2", "1.5", "2.0", "3.0")  # m: curva mais fechada da pista (--min-radius)
STREAM_PORT = 8080  # main.py --stream


# Opcoes da missao na aba de comandos: cada uma vira um argumento do main.py.
MISSION_DEFAULTS = {
    "navigation": "cones",        # cones | line
    "speed": SPEED_PRESETS[0],
    "stream": True,               # --stream 8080 (video no navegador)
    "estimator": True,            # False: --no-estimator
    "pure_pursuit": True,         # False: --stanley
    "preview": True,              # False: --no-preview
    "cone_debug": True,           # False: --no-cone-debug
    "record_frames": False,       # --record-frames HZ (coleta para a rede)
    "record_hz": "3",
    "classifier": False,          # --classifier CAMINHO
    "classifier_path": "modelos/cones.onnx",
    "vision_width": "0",
    "steering_filter": "3",
    "min_radius": "1.5",          # --min-radius (so cones; 1.5 e o padrao do main.py)
}
VISION_WIDTHS = ("0", "320", "480")
STEERING_FILTERS = ("1", "3", "5")
RECORD_RATES = ("1", "3", "5")
CONE_ONLY_OPTIONS = ("estimator", "pure_pursuit", "preview", "cone_debug", "record_frames", "classifier")


def build_main_command(options: dict) -> str:
    """Linha do main.py com as opcoes marcadas (padroes de MISSION_DEFAULTS).

    Opcoes so dos cones sao ignoradas na linha amarela; pure pursuit e previa
    de curva dependem do estimador (sem ele o main.py ja usa o Stanley).
    """
    o = {**MISSION_DEFAULTS, **options}
    parts = ["python3 -u main.py", f"--navigation {o['navigation']}",
             f"--vision-width {o['vision_width']}", f"--steering-filter {o['steering_filter']}",
             f"--speed {o['speed']}"]
    if o["stream"]:
        parts.append(f"--stream {STREAM_PORT}")
    if o["navigation"] == "cones":
        if not o["estimator"]:
            parts.append("--no-estimator")
        else:
            if not o["pure_pursuit"]:
                parts.append("--stanley")
            if not o["preview"]:
                parts.append("--no-preview")
        if not o["cone_debug"]:
            parts.append("--no-cone-debug")
        if o["record_frames"]:
            parts.append(f"--record-frames {o['record_hz']}")
        if str(o["min_radius"]) != MISSION_DEFAULTS["min_radius"]:
            parts.append(f"--min-radius {o['min_radius']}")
        if o["classifier"] and str(o["classifier_path"]).strip():
            parts.append(f"--classifier {shlex.quote(str(o['classifier_path']).strip())}")
    return " ".join(parts)


def with_main_command(command: str, main_line: str) -> str:
    """Troca as linhas que executam main.py pela nova; sem nenhuma, poe no inicio."""
    lines = command.splitlines()
    replaced = [main_line if re.search(r"\bmain\.py\b", line) else line for line in lines]
    if replaced == lines and not any(re.search(r"\bmain\.py\b", line) for line in lines):
        replaced = [main_line] + [line for line in lines if line.strip()]
    return "\n".join(replaced)


def with_speed(command: str, speed: str) -> str:
    """Troca --speed/--fast das linhas que executam main.py, sem tocar nas demais."""
    lines = []
    for line in command.splitlines():
        if re.search(r"\bmain\.py\b", line):
            line = re.sub(r"\s+--speed(?:=|\s+)\S+|\s+--fast\b", "", line).rstrip()
            line += f" --speed {speed}"
        lines.append(line)
    return "\n".join(lines)


# Colunas extras do car.csv no modo cones (main.py), lidas quando existirem.
CONE_COLUMNS = (
    "cone_green_count",
    "cone_red_count",
    "line_valid",
    "line_hold",
    "cone_segment",
    "cross_track_m",
    "curvature_m_inv",
)

MACS_CARS = {
    "verde": "2c:cf:67:1c:29:4a",
    "vermelho": "d8:3a:dd:f1:8a:4f",
    "roxo": "2c:cf:67:1c:29:07",
}

COLORS = {"verde": "#00cc00", "vermelho": "#cc0000", "roxo": "#8000cc"}

CAR_ICON = "🚗 "

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Saida de cada comando/teste remoto fica NO CARRO, dentro de logs/ (o que o
# "Coletar Dados" baixa): logs/terminal/<AAAA-MM-DD>/<HHhMMmSS>_<o-que-rodou>.log
# A hora e a do notebook: no hotspot, sem internet, o relogio do Pi pode estar errado.
REMOTE_TERMINAL_DIR = "logs/terminal"
# Logs antigos (antes de logs/terminal), recuperados pelo Coletar Dados.
LEGACY_TMP_LOGS = re.compile(r"^fva_(gui|teste|camera_http)_[0-9a-f]+\.log$")


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._]+", "-", text).strip("-._").lower() or "comando"


def remote_log_path(workdir: str, label: str, now=None):
    """(pasta, arquivo) do log no carro para um comando com este rotulo."""
    now = time.localtime() if now is None else now
    folder = posixpath.join(workdir, REMOTE_TERMINAL_DIR, time.strftime("%Y-%m-%d", now))
    return folder, posixpath.join(folder, f"{time.strftime('%Hh%Mm%S', now)}_{label}.log")


def command_label(command: str) -> str:
    """Nome curto e legivel do que o comando roda (para o nome do arquivo de log)."""
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = command.split()
    if not argv:
        return "comando"
    joined = " ".join(argv)
    script = next((Path(a).stem for a in argv if a.endswith(".py")), None)
    if script == "main":
        options = {}
        for index, arg in enumerate(argv):
            if arg.startswith("--"):
                following = argv[index + 1] if index + 1 < len(argv) else ""
                options[arg] = "" if following.startswith("--") else following
        navigation = {"line": "linha"}.get(options.get("--navigation"), "cones")
        parts = ["missao", navigation]
        if "--fast" in options:
            parts.append("1.5ms")
        elif options.get("--speed"):
            parts.append(f"{options['--speed']}ms")
        if navigation == "cones":
            if "--no-estimator" in options:
                parts.append("sem-estimador")
            else:
                parts.append("stanley" if "--stanley" in options else "purepursuit")
                if "--no-preview" in options:
                    parts.append("sem-previa")
            if "--record-frames" in options:
                parts.append("gravando")
            if "--classifier" in options and options["--classifier"].lower() != "none":
                parts.append("rede")
        return "_".join(parts)
    if "unittest" in argv:
        return "testes_unitarios"
    if script:
        return _slug(script)
    return _slug(Path(argv[0]).name if argv[0] not in ("sudo", "nohup") or len(argv) < 2 else argv[1])


# ---------------------------------------------------------------- envio incremental
# Nunca enviados: lixo do Python/sistema e as pastas de DADOS (o que o carro
# grava; a copia local nao pode sobrescrever o que esta la).
SYNC_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".idea", ".vscode"}
SYNC_SKIP_TOP_DIRS = {"logs", "experimentos", "ajuste_camera"}
SYNC_SKIP_SUFFIXES = (".pyc", ".pyo", ".swp")
SYNC_SKIP_FILES = {".DS_Store", "Thumbs.db"}


def local_tree(local_dir: str) -> dict:
    """{caminho relativo posix: (tamanho, mtime inteiro, caminho local)} do que deve ir ao carro."""
    files = {}
    for root, dirs, names in os.walk(local_dir):
        rel_root = os.path.relpath(root, local_dir)
        top = rel_root == "."
        dirs[:] = sorted(d for d in dirs if d not in SYNC_SKIP_DIRS
                         and not (top and d in SYNC_SKIP_TOP_DIRS))
        for name in names:
            if name in SYNC_SKIP_FILES or name.endswith(SYNC_SKIP_SUFFIXES):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            info = os.stat(path)
            rel = name if top else posixpath.join(*rel_root.split(os.sep), name)
            files[rel] = (info.st_size, int(info.st_mtime), path)
    return files


def _output_text(stream) -> str:
    data = stream.read()
    return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)


def remote_tree(client, remote_dir: str) -> dict:
    """{caminho relativo: (tamanho, mtime inteiro)} no carro, numa so ida e volta (find)."""
    command = (f"cd {shlex.quote(remote_dir)} 2>/dev/null && "
               "find . -type f -printf '%P\\t%s\\t%T@\\n' 2>/dev/null")
    _, out, _ = client.exec_command(command, timeout=60)
    files = {}
    for line in _output_text(out).splitlines():
        parts = line.rsplit("\t", 2)
        if len(parts) == 3:
            try:
                files[parts[0]] = (int(parts[1]), int(float(parts[2])))
            except ValueError:
                continue
    return files


def remote_md5(client, remote_dir: str, paths, chunk=150) -> dict:
    """md5 no carro de varios arquivos (lotes de `chunk` por comando)."""
    sums = {}
    paths = list(paths)
    for start in range(0, len(paths), chunk):
        names = " ".join(shlex.quote(p) for p in paths[start:start + chunk])
        _, out, _ = client.exec_command(f"cd {shlex.quote(remote_dir)} && md5sum -- {names} 2>/dev/null",
                                        timeout=60)
        for line in _output_text(out).splitlines():
            digest, _, name = line.partition("  ")
            if name:
                sums[name] = digest
    return sums


def local_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_plan(local: dict, remote: dict, force=False):
    """(enviar, conferir conteudo, iguais): mesmo tamanho e data = igual; mesmo
    tamanho e data diferente = conferir o md5 (1a vez depois de enviar sem data)."""
    send, check, same = [], [], []
    for rel, (size, mtime, _) in sorted(local.items()):
        there = remote.get(rel)
        if force or there is None or there[0] != size:
            send.append(rel)
        elif there[1] == mtime:
            same.append(rel)
        else:
            check.append(rel)
    return send, check, same


def sync_directory(client, sftp, local_dir, remote_dir, log, force=False, delete_py=False):
    """Envia so o que mudou de local_dir para remote_dir. Devolve um resumo (dict)."""
    started = time.monotonic()
    local = local_tree(local_dir)
    remote = remote_tree(client, remote_dir)
    send, check, same = sync_plan(local, remote, force)
    touched = []
    if check:
        sums = remote_md5(client, remote_dir, check)
        for rel in check:
            if sums.get(rel) == local_md5(local[rel][2]):
                touched.append(rel)  # conteudo igual: so acerta a data no carro
            else:
                send.append(rel)
    known_dirs = {""} | {posixpath.dirname(rel) for rel in remote}
    sent_bytes = 0
    for rel in send:
        folder = posixpath.dirname(rel)
        if folder not in known_dirs:
            path = ""
            for part in folder.split("/"):
                path = posixpath.join(path, part) if path else part
                if path not in known_dirs:
                    try:
                        sftp.mkdir(posixpath.join(remote_dir, path))
                    except IOError:
                        pass  # ja existe
                    known_dirs.add(path)
        size, mtime, path = local[rel]
        target = posixpath.join(remote_dir, rel)
        sftp.put(path, target)
        sftp.utime(target, (mtime, mtime))
        sent_bytes += size
        log(f"   ↑ {rel}")
    for rel in touched:
        mtime = local[rel][1]
        sftp.utime(posixpath.join(remote_dir, rel), (mtime, mtime))
    removed = []
    if delete_py:
        # So .py que sumiram daqui (arquivo renomeado/apagado, testes de outra
        # branch); nunca dentro das pastas de dados.
        for rel in sorted(set(remote) - set(local)):
            if (rel.endswith(".py") and rel.split("/")[0] not in SYNC_SKIP_TOP_DIRS
                    and not any(part in SYNC_SKIP_DIRS for part in rel.split("/"))):
                sftp.remove(posixpath.join(remote_dir, rel))
                removed.append(rel)
                log(f"   ✗ {rel} (não existe mais aqui)")
    return {"sent": len(send), "same": len(same) + len(touched), "bytes": sent_bytes,
            "removed": len(removed), "seconds": time.monotonic() - started}


def open_folder(path: Path):
    """Abre a pasta no gerenciador de arquivos do sistema."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        os.startfile(str(path))  # noqa: S606
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def parse_telemetry_line(line: str):
    """Converte uma linha DATA no protocolo oficial da telemetria FVA.

    Formato esperado:
    DATA,t,x,y,v,vref,a,u,w,th
    """
    clean_line = ANSI_ESCAPE_RE.sub("", line).strip()
    parts = [p.strip() for p in clean_line.split(",")]

    if len(parts) != 10 or parts[0] != "DATA":
        raise ValueError(
            f"esperados 10 campos iniciando por DATA, recebidos {len(parts)}"
        )

    _, t, x, y, v, vref, a, u, w, th = parts

    return {
        "t": float(t),
        "x": float(x),
        "y": float(y),
        "v": float(v),
        "vref": float(vref),
        "a": float(a),
        "u": float(u),
        "w": float(w),
        "th": float(th),
    }


########################################
# Utilitários de rede multiplataforma
########################################
def normalize_mac(mac: str) -> str:
    mac = mac.strip().lower()
    mac = re.sub(r"[^0-9a-f]", "", mac)
    if len(mac) != 12:
        raise ValueError(f"MAC inválido: {mac}")
    return ":".join(mac[i : i + 2] for i in range(0, 12, 2))


########################################
def ping_host(host: str) -> bool:
    """Envia um ping curto em Windows, Linux ou macOS."""
    if platform.system() == "Windows":
        cmd = ["ping", "-n", "1", "-w", "700", host]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", host]
    try:
        return (
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
    except Exception:
        return False


########################################
def parse_arp_table():
    """Retorna pares (IP, MAC) usando as tabelas ARP disponíveis no SO."""
    entries = []
    commands = []
    if platform.system() == "Windows":
        commands.append(["arp", "-a"])
    else:
        commands.extend([["ip", "neigh"], ["arp", "-n"], ["arp", "-a"]])

    for cmd in commands:
        try:
            out = subprocess.check_output(cmd, text=True, errors="ignore")
        except Exception:
            continue

        for line in out.splitlines():
            ip_match = re.search(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])", line)
            mac_match = re.search(r"([0-9a-fA-F]{2}(?:[:-][0-9a-fA-F]{2}){5})", line)
            if ip_match and mac_match:
                try:
                    entries.append(
                        (ip_match.group(1), normalize_mac(mac_match.group(1)))
                    )
                except ValueError:
                    pass

    return list(dict.fromkeys(entries))


########################################
def find_ip_by_mac_arptable(target_mac: str):
    try:
        target_mac = normalize_mac(target_mac)
    except Exception:
        return None

    for ip, mac in parse_arp_table():
        if mac == target_mac:
            return ip
    return None


########################################
# Interface Principal
########################################
class RsyncGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FVA - gerenciador de controle")
        self.geometry("1120x760")
        self.minsize(940, 620)
        # self.attributes("-fullscreen", True)
        # self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))
        self.devices = {}
        self.selected_files = []
        self.active_device = None

        # Estado exclusivo da aba "Testar Modulos"
        self.test_client = None
        self.test_channel = None
        self.test_remote_pid = None
        self.test_remote_ip = None
        self.test_run_id = None
        self.test_running = False
        self.camera_window = None
        self.camera_label = None
        self.camera_photo = None
        self.camera_frame = None
        self.live_video_running = False
        self.live_video_generation = 0
        self.live_video_photo = None
        self.live_video_frame = None
        self.live_video_queue = queue.Queue(maxsize=2)
        self.test_output_queue = queue.Queue()
        self.test_status_pending = None
        self.test_status_mark = None

        # Estilo definido antes de criar as abas para evitar larguras excessivas.
        style = ttk.Style(self)
        style.configure(".", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 11))
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TCheckbutton", font=("Arial", 12))
        style.configure("TNotebook.Tab", font=("Arial", 11, "bold"), padding=[7, 7])

        self._build_ui()
        self.after(50, self._flush_test_output)
        self.after(100, self._refresh_live_video)
        # preenche a senha padrão (se houver)
        if DEFAULT_PASS:
            self.pass_entry.insert(0, DEFAULT_PASS)
        self.after(
            200, lambda: threading.Thread(target=self.refresh_ips, daemon=True).start()
        )

        self.telemetry = {}
        self.telemetry_queue = queue.Queue()
        self.after(200, self._refresh_live_plot)

    ########################################
    def _build_ui(self):

        ########################################
        # Cabecalho
        ########################################
        header = ttk.Frame(self)
        header.pack(fill="x", padx=15, pady=(10, 5))

        # titulo
        title_frame = ttk.Frame(header)
        title_frame.pack(side="left")

        ttk.Label(
            title_frame,
            text="FVA — Fundamentos de Veículos Autônomos",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            title_frame,
            text="Gerenciador dos Veículos Experimentais",
            font=("Arial", 12),
        ).pack(anchor="w")

        # logo UFMG
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(BASE_DIR, "assets", "ufmg_logo.png")
        self.ufmg_logo = tk.PhotoImage(file=logo_path)
        # reduz a imagem pela metade
        self.ufmg_logo = self.ufmg_logo.subsample(6, 6)

        ttk.Label(header, image=self.ufmg_logo).pack(side="right")

        ########################################
        # Pasta remota global
        ########################################
        workdir_frame = ttk.Frame(self)
        workdir_frame.pack(fill="x", padx=15, pady=(3, 8))

        ttk.Label(workdir_frame, text="Pasta remota:").pack(side="left")
        self.dest_entry = ttk.Entry(workdir_frame)
        self.dest_entry.insert(0, DEFAULT_DEST)
        self.dest_entry.pack(side="left", fill="x", expand=True, padx=(8, 6))
        ttk.Button(
            workdir_frame, text="Restaurar padrão", command=self.restore_default_workdir
        ).pack(side="left")

        ########################################
        # Abas
        ########################################
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.tab_home = tk.Frame(notebook, bg="black")
        self.tab_files = ttk.Frame(notebook)
        self.tab_cmds = ttk.Frame(notebook)
        self.tab_video = ttk.Frame(notebook)
        self.tab_tests = ttk.Frame(notebook)
        self.tab_data = ttk.Frame(notebook)

        notebook.add(self.tab_home, text="🏠 Início")
        notebook.add(self.tab_files, text="📂 Enviar Arquivos")
        notebook.add(self.tab_cmds, text="💻 Comandos")
        notebook.add(self.tab_video, text="📷 Câmera")
        notebook.add(self.tab_data, text="📊 Coletar Dados")
        notebook.add(self.tab_tests, text="🧪 Módulos de Teste")

        self._build_tab_home(self.tab_home)
        self._build_tab_files(self.tab_files)
        self._build_tab_cmds(self.tab_cmds)
        self._build_tab_tests(self.tab_tests)
        self._build_tab_video(self.tab_video)
        self._build_tab_data(self.tab_data)

    ########################################
    def ui(self, func, *args, **kwargs):
        self.after(0, lambda: func(*args, **kwargs))

    ########################################
    def restore_default_workdir(self):
        self.dest_entry.delete(0, "end")
        self.dest_entry.insert(0, DEFAULT_DEST)

    ########################################
    def _build_tab_home(self, parent):

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        image_path = os.path.join(BASE_DIR, "assets", "wallpaper_fva.png")

        # carrega a imagem
        self.home_image = tk.PhotoImage(file=image_path)

        # label com fundo preto
        label = tk.Label(
            parent,
            image=self.home_image,
            bg="black",
            borderwidth=0,
            highlightthickness=0,
        )

        # centraliza horizontal e verticalmente
        label.place(relx=0.5, rely=0.5, anchor="center")

    ########################################
    def _build_tab_files(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Veículo:").pack(anchor="w")

        self.active_device_label = tk.Label(
            top, text="🔍 Procurando veículo...", font=("Arial", 12, "bold"), anchor="w"
        )
        self.active_device_label.pack(fill="x", padx=4, pady=6)

        # Cadastro interno dos veículos. Os MACs são usados apenas para detecção
        # e não são exibidos na interface.
        for name, mac in MACS_CARS.items():
            self.devices[name] = {"mac": mac, "ip": None}

        # Botões de ação
        btns = ttk.Frame(parent)
        btns.pack(fill="x", padx=6, pady=6)
        ttk.Button(
            btns,
            text="Atualizar IPs",
            command=lambda: threading.Thread(
                target=self.refresh_ips, daemon=True
            ).start(),
        ).pack(side="left", padx=4)
        ttk.Button(btns, text="Selecionar arquivos…", command=self.select_files).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Adicionar pasta…", command=self.add_directory).pack(
            side="left", padx=4
        )
        ttk.Button(
            btns, text="Remover selecionado(s)", command=self.remove_selected
        ).pack(side="left", padx=4)
        ttk.Button(btns, text="Limpar seleção", command=self.clear_files).pack(
            side="left", padx=4
        )

        # Lista de arquivos/pastas
        ttk.Label(parent, text="Arquivos/Pastas selecionados:").pack(
            anchor="w", padx=10
        )
        self.files_listbox = tk.Listbox(parent, height=6, selectmode="extended")
        self.files_listbox.pack(fill="x", padx=10, pady=(2, 6))

        # SSH / SFTP (Paramiko)
        opt = ttk.Frame(parent)
        opt.pack(fill="x", padx=10, pady=6)
        ttk.Label(opt, text="Usuário:").pack(side="left")
        self.user_entry = ttk.Entry(opt, width=14)
        self.user_entry.insert(0, SSH_USER)
        self.user_entry.pack(side="left", padx=4)
        ttk.Label(opt, text="Senha:").pack(side="left", padx=(8, 0))
        self.pass_entry = ttk.Entry(opt, width=14, show="*")
        self.pass_entry.pack(side="left", padx=4)
        ttk.Label(opt, text="Transferência: SFTP (multiplataforma)").pack(
            side="left", padx=(12, 0)
        )

        # Envio
        send = ttk.Frame(parent)
        send.pack(fill="x", padx=10, pady=6)
        ttk.Button(send, text="Enviar arquivos", command=self.send_to_selected).pack(
            side="left", padx=4
        )
        # Envio incremental: so o que mudou desde o ultimo envio vai pelo Wi-Fi.
        self.sync_force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(send, text="Enviar tudo (ignorar o que já está no carro)",
                        variable=self.sync_force_var).pack(side="left", padx=(12, 4))
        self.sync_delete_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(send, text="Apagar no carro .py que não existem mais aqui",
                        variable=self.sync_delete_var).pack(side="left", padx=(12, 4))

        # Log
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)
        ttk.Label(log_frame, text="Log:").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(log_frame, height=10)
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        # estilos
        style = ttk.Style()
        style.configure("Default.TCheckbutton", foreground="white", font=("Arial", 10))
        for n, c in COLORS.items():
            style.configure(
                f"{n}.TCheckbutton", foreground=c, font=("Arial", 10, "bold")
            )

    ########################################
    def _build_tab_cmds(self, parent):
        ttk.Label(parent, text="Executar comandos remotos no veículo ativo.").pack(
            anchor="w", padx=10, pady=(10, 4)
        )

        hint = ttk.Label(
            parent,
            text="Os comandos serão executados na pasta remota definida no topo da janela.",
            foreground="#888",
        )
        hint.pack(anchor="w", padx=10, pady=(0, 6))

        self._build_mission_options(parent)
        cmds_frame = ttk.Frame(parent)
        cmds_frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(cmds_frame, text="Comandos (1 por linha):").pack(anchor="w")

        ttk.Label(
            cmds_frame,
            text="As opções acima reescrevem a linha do main.py; dá para editar à mão antes de Executar.",
            foreground="#888",
        ).pack(anchor="w")
        self.cmd_text = scrolledtext.ScrolledText(
            cmds_frame,
            height=3,
            bg="black",
            fg="white",
            insertbackground="white",
            font=("Courier", 11),
        )

        self.cmd_text.insert("end", build_main_command(self.mission_options()) + "\n")
        self.cmd_text.pack(fill="x", pady=4)

        cmd_buttons = ttk.Frame(parent)
        cmd_buttons.pack(fill="x", padx=10, pady=6)

        ttk.Button(
            cmd_buttons, text="Executar", command=self.run_cmds_on_selected
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            cmd_buttons, text="Ver câmera", command=self.open_camera_stream
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            cmd_buttons, text="Limpar terminal", command=self.clear_cmd_log
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            cmd_buttons, text="📂 Dados coletados", command=self.open_collected_data
        ).pack(side="left")

        # area inferior: grafico e terminal lado a lado
        bottom = ttk.PanedWindow(parent, orient="horizontal")
        bottom.pack(fill="both", expand=True, padx=10, pady=6)

        ########################################
        # painel do grafico
        plot_frame = ttk.Frame(bottom)

        # seletor do grafico
        plot_select = ttk.Frame(plot_frame)
        plot_select.pack(fill="x", pady=(0, 5))

        ttk.Label(plot_select, text="Gráfico:").pack(side="left", padx=(0, 5))

        self.plot_var = tk.StringVar(value="Velocidade")

        self.plot_combo = ttk.Combobox(
            plot_select,
            textvariable=self.plot_var,
            state="readonly",
            values=[
                "Velocidade",
                "Aceleração / Controle",
                "Velocidade angular",
                "Orientação",
                "Trajetória XY",
            ],
            width=24,
        )

        self.plot_combo.pack(side="left")

        self.plot_combo.bind("<<ComboboxSelected>>", lambda event: self.update_plot())

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Velocidade [m/s]")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ########################################
        # painel do terminal
        terminal_frame = ttk.Frame(bottom)

        ttk.Label(terminal_frame, text="Terminal:").pack(anchor="w")

        self.cmd_log = scrolledtext.ScrolledText(
            terminal_frame,
            bg="black",
            fg="white",
            insertbackground="white",
            font=("Courier", 11),
        )
        self.cmd_log.pack(fill="both", expand=True)

        # cores ANSI do terminal
        self.cmd_log.tag_configure("ansi_black", foreground="#555555")
        self.cmd_log.tag_configure("ansi_red", foreground="#ff5555")
        self.cmd_log.tag_configure("ansi_green", foreground="#55ff55")
        self.cmd_log.tag_configure("ansi_yellow", foreground="#ffff55")
        self.cmd_log.tag_configure("ansi_blue", foreground="#5555ff")
        self.cmd_log.tag_configure("ansi_magenta", foreground="#ff55ff")
        self.cmd_log.tag_configure("ansi_cyan", foreground="#55ffff")
        self.cmd_log.tag_configure("ansi_white", foreground="white")

        self.cmd_log.configure(state="disabled")

        # adiciona os dois lados
        bottom.add(plot_frame, weight=1)
        bottom.add(terminal_frame, weight=3)

        # inicia com 60% para o gráfico e 40% para o terminal
        def set_initial_panes():
            width = bottom.winfo_width()
            if width > 1:
                bottom.sashpos(0, int(width * 0.60))

        self.after(100, set_initial_panes)

    def _build_mission_options(self, parent):
        """Caixas da missao (main.py): cada mudanca reescreve a linha do comando."""
        box = ttk.LabelFrame(parent, text="Opções da missão (main.py)")
        box.pack(fill="x", padx=10, pady=(4, 2))
        self.mission_vars = {}
        for key, value in MISSION_DEFAULTS.items():
            variable = tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=value)
            variable.trace_add("write", lambda *_: self.apply_mission_options())
            self.mission_vars[key] = variable
        self.speed_var = self.mission_vars["speed"]  # compatibilidade

        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Label(row, text="Navegação:").pack(side="left")
        for value, label in (("cones", "Cones"), ("line", "Linha amarela")):
            ttk.Radiobutton(row, text=label, value=value,
                            variable=self.mission_vars["navigation"]).pack(side="left", padx=(4, 8))
        for label, key, values, width in (("Vel. máx. retas [m/s]:", "speed", SPEED_PRESETS, 6),
                                          ("Curva mais fechada R [m]:", "min_radius", CURVE_RADII, 5),
                                          ("Largura da visão:", "vision_width", VISION_WIDTHS, 5),
                                          ("Filtro direção:", "steering_filter", STEERING_FILTERS, 4)):
            ttk.Label(row, text=label).pack(side="left", padx=(12, 4))
            combo = ttk.Combobox(row, textvariable=self.mission_vars[key], state="readonly",
                                 values=values, width=width)
            combo.pack(side="left")
            self.mission_combos = {**getattr(self, "mission_combos", {}), key: combo}

        checks = ttk.Frame(box)
        checks.pack(fill="x", padx=6, pady=2)
        self.mission_checks = {}
        items = (
            ("stream", "Transmitir vídeo (Ver câmera)"),
            ("estimator", "Estimador do corredor"),
            ("pure_pursuit", "Pure pursuit (desmarcado: Stanley)"),
            ("preview", "Reduzir antes da curva"),
            ("cone_debug", "Debug de cones (jsonl + quadros)"),
        )
        for index, (key, label) in enumerate(items):
            check = ttk.Checkbutton(checks, text=label, variable=self.mission_vars[key])
            check.grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 18), pady=1)
            self.mission_checks[key] = check

        extra = ttk.Frame(box)
        extra.pack(fill="x", padx=6, pady=(2, 6))
        check = ttk.Checkbutton(extra, text="Gravar quadros para a rede, Hz:",
                                variable=self.mission_vars["record_frames"])
        check.pack(side="left")
        self.mission_checks["record_frames"] = check
        self.record_combo = ttk.Combobox(extra, textvariable=self.mission_vars["record_hz"],
                                         state="readonly", values=RECORD_RATES, width=3)
        self.record_combo.pack(side="left", padx=(4, 18))
        check = ttk.Checkbutton(extra, text="Classificador (rede):",
                                variable=self.mission_vars["classifier"])
        check.pack(side="left")
        self.mission_checks["classifier"] = check
        self.classifier_entry = ttk.Entry(extra, textvariable=self.mission_vars["classifier_path"], width=26)
        self.classifier_entry.pack(side="left", padx=(4, 0))
        self._update_mission_widgets()

    def mission_options(self):
        return {key: variable.get() for key, variable in getattr(self, "mission_vars", {}).items()}

    def _update_mission_widgets(self):
        """Desabilita o que nao se aplica: opcoes de cones na linha amarela,
        pure pursuit e previa sem o estimador, Hz/caminho com a caixa desmarcada."""
        options = self.mission_options()
        cones = options.get("navigation") == "cones"
        for key, check in self.mission_checks.items():
            enabled = cones or key not in CONE_ONLY_OPTIONS
            if key in ("pure_pursuit", "preview"):
                enabled = enabled and options.get("estimator", True)
            check.state(["!disabled"] if enabled else ["disabled"])
        self.record_combo.state(["!disabled"] if cones and options.get("record_frames") else ["disabled"])
        radius = getattr(self, "mission_combos", {}).get("min_radius")
        if radius is not None:  # limite de visibilidade usa o caminho do estimador
            radius.state(["readonly", "!disabled"] if cones and options.get("estimator", True)
                         else ["disabled"])
        self.classifier_entry.state(["!disabled"] if cones and options.get("classifier") else ["disabled"])

    def apply_mission_options(self):
        if not hasattr(self, "cmd_text") or not hasattr(self, "mission_checks"):
            return
        self._update_mission_widgets()
        command = with_main_command(self.cmd_text.get("1.0", "end-1c"),
                                    build_main_command(self.mission_options()))
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("end", command + "\n")

    def _build_tab_video(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Porta HTTP:").pack(side="left")
        self.video_port = tk.StringVar(value="8080")
        ttk.Entry(controls, textvariable=self.video_port, width=7).pack(
            side="left", padx=6
        )
        ttk.Button(controls, text="Conectar", command=self.start_live_video).pack(
            side="left", padx=5
        )
        ttk.Button(controls, text="Parar", command=self.stop_live_video).pack(
            side="left", padx=5
        )
        self.video_status = ttk.Label(
            parent, text="Execute main.py com a camera habilitada e clique em Conectar."
        )
        self.video_status.pack(anchor="w", padx=10)
        self.live_video_label = tk.Label(
            parent, text="Sem video", bg="black", fg="white"
        )
        self.live_video_label.pack(fill="both", expand=True, padx=10, pady=10)
        self.live_video_label.bind("<Configure>", self._resize_live_video)

    def start_live_video(self):
        targets = self.get_selected_devices()
        if not targets:
            messagebox.showinfo("Video", "Nenhum veiculo detectado.")
            return
        try:
            port = int(self.video_port.get())
            if not 1 <= port <= 65535:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Video", "Porta HTTP invalida.")
            return
        self.stop_live_video()
        self.live_video_running = True
        self.live_video_generation += 1
        generation = self.live_video_generation
        ip = targets[0][1]
        self.video_status.config(text=f"Conectando a {ip}:{port} ...")
        threading.Thread(
            target=self._live_video_worker, args=(ip, port, generation), daemon=True
        ).start()

    def stop_live_video(self):
        self.live_video_running = False
        self.live_video_generation += 1
        self.live_video_frame = None
        if hasattr(self, "video_status"):
            self.video_status.config(text="Video desconectado.")

    def _live_video_worker(self, ip, port, generation):
        url = f"http://{ip}:{port}/frame.png"
        while self.live_video_running and generation == self.live_video_generation:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as response:
                    data = response.read(2_000_000)
                if generation != self.live_video_generation:
                    break
                try:
                    self.live_video_queue.put_nowait((generation, data, None))
                except queue.Full:
                    pass
                time.sleep(0.10)  # ate ~10 FPS; nao bloqueia a GUI
            except Exception as exc:
                try:
                    self.live_video_queue.put_nowait((generation, None, str(exc)))
                except queue.Full:
                    pass
                time.sleep(0.7)

    def _resize_live_video(self, event=None):
        """Ajusta a imagem ao espaço disponível sem distorcer a proporção."""
        if self.live_video_frame is None or not self.live_video_running:
            return
        try:
            width = max(1, self.live_video_label.winfo_width() - 12)
            height = max(1, self.live_video_label.winfo_height() - 12)
            src_w = self.live_video_frame.width()
            src_h = self.live_video_frame.height()
            if src_w < 1 or src_h < 1:
                return
            # Zoom inteiro usando Tk nativo: sem Pillow e sem alterar o stream.
            zoom = max(1, min(width // src_w, height // src_h))
            photo = (
                self.live_video_frame.zoom(zoom, zoom)
                if zoom > 1
                else self.live_video_frame
            )
            # Se a área disponível for menor que a imagem original, reduz proporcionalmente.
            if photo.width() > width or photo.height() > height:
                shrink = max(
                    (photo.width() + width - 1) // width,
                    (photo.height() + height - 1) // height,
                )
                photo = photo.subsample(shrink, shrink)
            self.live_video_photo = photo
            self.live_video_label.config(image=photo, text="")
        except tk.TclError:
            pass

    def _refresh_live_video(self):
        latest = None
        try:
            while True:
                latest = self.live_video_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            generation, data, error = latest
            if self.live_video_running and generation == self.live_video_generation:
                if data:
                    try:
                        self.live_video_frame = tk.PhotoImage(
                            data=base64.b64encode(data).decode("ascii"), format="png"
                        )
                        self._resize_live_video()
                        self.video_status.config(
                            text="Video conectado (imagem publicada pelo main.py)"
                        )
                    except Exception as exc:
                        self.video_status.config(text=f"Erro ao exibir quadro: {exc}")
                elif error:
                    self.video_status.config(text=f"Aguardando video: {error}")
        self.after(100, self._refresh_live_video)

    def _update_camera_mode_visibility(self, event=None):
        """Mostra a escolha de visao apenas para o teste da camera USB."""
        if self.test_module_var.get() == "Câmera USB":
            self.camera_mode_frame.pack(
                side="left", padx=(0, 12), before=self.test_run_button
            )
        else:
            self.camera_mode_frame.pack_forget()

    def _resize_camera_test(self, event=None):
        """Amplia a prévia do teste preservando a proporção, sem Pillow."""
        if self.camera_frame is None or self.camera_label is None:
            return
        try:
            if not self.camera_label.winfo_exists():
                return
            width = max(1, self.camera_label.winfo_width() - 12)
            height = max(1, self.camera_label.winfo_height() - 12)
            src_w = self.camera_frame.width()
            src_h = self.camera_frame.height()
            if src_w < 1 or src_h < 1:
                return
            zoom = max(1, min(width // src_w, height // src_h))
            photo = (
                self.camera_frame.zoom(zoom, zoom) if zoom > 1 else self.camera_frame
            )
            if photo.width() > width or photo.height() > height:
                shrink = max(
                    (photo.width() + width - 1) // width,
                    (photo.height() + height - 1) // height,
                )
                photo = photo.subsample(shrink, shrink)
            self.camera_photo = photo
            self.camera_label.configure(image=photo, text="")
        except tk.TclError:
            pass

    ########################################
    def _build_tab_tests(self, parent):

        ttk.Label(
            parent, text="Testar módulos individuais da biblioteca fva_car."
        ).pack(anchor="w", padx=10, pady=(10, 4))

        self.test_device_label = ttk.Label(
            parent, text="Veículo: aguardando detecção..."
        )
        self.test_device_label.pack(anchor="w", padx=10, pady=(0, 8))

        hint = ttk.Label(
            parent,
            text="O teste é executado na pasta remota definida no topo da janela.",
            foreground="#888",
        )
        hint.pack(anchor="w", padx=10, pady=(0, 8))

        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=10, pady=6)

        ttk.Label(controls, text="Módulo:").pack(side="left")

        # Começamos apenas com o módulo já validado.
        # Novos testes podem ser acrescentados aqui depois.
        self.test_modules = {
            "Encoder / RC / Chaves": "fva_car/encoder.py",
            "Ultrassom": "fva_car/ultrasonic.py",
            "Câmera USB": "__camera__",
            "Carro completo": "fva_car/car.py",
        }

        self.test_module_var = tk.StringVar(value="Encoder / RC / Chaves")

        self.test_module_combo = ttk.Combobox(
            controls,
            textvariable=self.test_module_var,
            values=list(self.test_modules.keys()),
            state="readonly",
            width=28,
        )
        self.test_module_combo.pack(side="left", padx=(8, 16))

        # Exibe o seletor de visao somente quando o modulo Camera USB estiver selecionado.
        self.camera_mode_frame = ttk.Frame(controls)
        ttk.Label(self.camera_mode_frame, text="Visão:").pack(side="left")
        self.camera_mode_var = tk.StringVar(value="Normal")
        self.camera_mode_combo = ttk.Combobox(
            self.camera_mode_frame,
            textvariable=self.camera_mode_var,
            values=["Normal", "ArUco", "Placas"],
            state="readonly",
            width=10,
        )
        self.camera_mode_combo.pack(side="left", padx=(6, 0))
        self.camera_mode_frame.pack(side="left", padx=(0, 12))
        self.test_module_combo.bind(
            "<<ComboboxSelected>>", self._update_camera_mode_visibility
        )
        self._update_camera_mode_visibility()

        self.test_run_button = ttk.Button(
            controls, text="▶ Executar", command=self.run_module_test
        )
        self.test_run_button.pack(side="left", padx=4)

        self.test_stop_button = ttk.Button(
            controls, text="■ Parar", command=self.stop_module_test, state="disabled"
        )
        self.test_stop_button.pack(side="left", padx=4)

        ttk.Button(controls, text="Gravar Arduino", command=self.flash_arduino).pack(
            side="left", padx=4
        )

        ttk.Button(controls, text="Limpar", command=self.clear_test_log).pack(
            side="left", padx=4
        )
        ttk.Button(controls, text="📂 Dados coletados", command=self.open_collected_data).pack(
            side="left", padx=4
        )

        ttk.Label(parent, text="Saída do teste:").pack(
            anchor="w", padx=10, pady=(10, 2)
        )

        self.test_log = scrolledtext.ScrolledText(
            parent,
            bg="black",
            fg="white",
            insertbackground="white",
            font=("Courier", 11),
        )
        self.test_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        # Cores ANSI usadas pelos módulos remotos
        for code, color in {
            "30": "#555555",
            "31": "#ff5555",
            "32": "#55ff55",
            "33": "#ffff55",
            "34": "#5555ff",
            "35": "#ff55ff",
            "36": "#55ffff",
            "37": "#ffffff",
            "90": "#555555",
            "91": "#ff5555",
            "92": "#55ff55",
            "93": "#ffff55",
            "94": "#5555ff",
            "95": "#ff55ff",
            "96": "#55ffff",
            "97": "#ffffff",
        }.items():
            self.test_log.tag_configure(f"ansi_{code}", foreground=color)
        self.test_log.configure(state="disabled")

    ########################################
    def _test_insert_colored(self, text):
        """Insere texto interpretando cores ANSI, sem adicionar quebra de linha."""
        current_tag = None
        pos = 0
        for match in ANSI_ESCAPE_RE.finditer(text):
            part = text[pos : match.start()]
            if part:
                self.test_log.insert("end", part, (current_tag,) if current_tag else ())
            for code in match.group()[2:-1].split(";"):
                if code in ("0", "39"):
                    current_tag = None
                elif code in ("1", "22"):
                    continue
                elif code in (
                    "30",
                    "31",
                    "32",
                    "33",
                    "34",
                    "35",
                    "36",
                    "37",
                    "90",
                    "91",
                    "92",
                    "93",
                    "94",
                    "95",
                    "96",
                    "97",
                ):
                    current_tag = f"ansi_{code}"
            pos = match.end()
        part = text[pos:]
        if part:
            self.test_log.insert("end", part, (current_tag,) if current_tag else ())

    ########################################
    def _queue_test_output(self, text):
        self.test_output_queue.put(text)

    ########################################
    def _flush_test_output(self):
        lines = []
        latest_status = None

        try:
            while len(lines) < 200:
                text = self.test_output_queue.get_nowait()

                plain = ANSI_ESCAPE_RE.sub("", text).lstrip()
                if (
                    plain.startswith("Vel =")
                    or plain.startswith("Distancia =")
                    or plain.startswith("Vel:")
                ):
                    latest_status = text
                else:
                    lines.append(text)
        except queue.Empty:
            pass

        if latest_status is not None:
            self.test_status_pending = latest_status

        if lines or latest_status is not None:
            self.test_log.configure(state="normal")

            # Se já existe uma linha dinâmica, remove-a antes de acrescentar
            # mensagens normais. Assim ela permanece sempre no final.
            if self.test_status_mark is not None:
                try:
                    self.test_log.delete(self.test_status_mark, "end-1c")
                except tk.TclError:
                    pass
                self.test_status_mark = None

            if lines:
                self._test_insert_colored("\n".join(lines) + "\n")

            if self.test_status_pending is not None:
                self.test_status_mark = self.test_log.index("end-1c")
                self._test_insert_colored(self.test_status_pending)

            self.test_log.see("end")
            self.test_log.configure(state="disabled")

        self.after(50, self._flush_test_output)

    ########################################
    def testlog_write(self, text):
        self.test_log.configure(state="normal")
        self._test_insert_colored(text + "\n")
        self.test_log.see("end")
        self.test_log.configure(state="disabled")

    ########################################
    def clear_test_log(self):
        self.test_log.configure(state="normal")
        self.test_log.delete("1.0", "end")
        self.test_log.configure(state="disabled")
        self.test_status_pending = None
        self.test_status_mark = None
        # Descarta mensagens pendentes de execuções anteriores.
        while True:
            try:
                self.test_output_queue.get_nowait()
            except queue.Empty:
                break

    ########################################
    def run_module_test(self):

        if self.test_running:
            messagebox.showinfo(
                "Teste em execução", "Já existe um teste de módulo em execução."
            )
            return

        targets = self.get_selected_devices()

        if not targets:
            messagebox.showinfo(
                "Nenhum alvo",
                "Nenhum veículo foi detectado. Atualize os IPs e tente novamente.",
            )
            return

        module_name = self.test_module_var.get()
        module_path = self.test_modules.get(module_name)

        if not module_path:
            messagebox.showerror(
                "Módulo inválido", "O módulo selecionado não foi encontrado."
            )
            return

        name, ip = targets[0]

        # Cada novo teste começa com o terminal limpo.
        self.clear_test_log()
        self.test_running = True
        self.test_remote_pid = None
        self.test_remote_ip = ip
        self.test_run_id = uuid.uuid4().hex
        self.test_status_pending = None
        self.test_status_mark = None
        self.test_run_button.configure(state="disabled")
        self.test_stop_button.configure(state="normal")
        self.test_device_label.configure(text=f"Veículo: {name.upper()} — {ip}")

        self.testlog_write("")
        self.testlog_write("=" * 60)
        self.testlog_write(f"Teste: {module_name}")
        self.testlog_write(f"Veículo: {name.upper()} ({ip})")
        self.testlog_write("=" * 60)

        if module_path == "__camera__":
            self._open_camera_window()
            camera_mode = self.camera_mode_var.get()
        else:
            camera_mode = None

        threading.Thread(
            target=(
                self._run_camera_test
                if module_path == "__camera__"
                else self._run_module_test
            ),
            args=(
                (name, ip, camera_mode)
                if module_path == "__camera__"
                else (name, ip, module_path)
            ),
            daemon=True,
        ).start()

    ########################################
    def _open_camera_window(self):
        if self.camera_window is not None and self.camera_window.winfo_exists():
            self.camera_window.destroy()
        window = tk.Toplevel(self)
        window.title("FVA — Câmera USB")
        window.geometry("680x560")
        window.protocol("WM_DELETE_WINDOW", self._close_camera_window)
        self.camera_window = window
        self.camera_label = tk.Label(
            window, text="Aguardando imagem...", bg="black", fg="white"
        )
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)
        self.camera_label.bind("<Configure>", self._resize_camera_test)
        ttk.Label(
            window,
            text="Prévia HTTP (até 10 FPS, 320×240). Fechar a janela interrompe o teste.",
        ).pack(pady=(0, 8))

    def _close_camera_window(self):
        self.stop_module_test()
        if self.camera_window is not None:
            self.camera_window.destroy()
        self.camera_window = None
        self.camera_label = None
        self.camera_photo = None
        self.camera_frame = None

    def _show_camera_frame(self, image_bytes):
        if self.camera_label is None or not self.camera_label.winfo_exists():
            return
        try:
            self.camera_frame = tk.PhotoImage(
                data=base64.b64encode(image_bytes).decode("ascii"), format="png"
            )
            self._resize_camera_test()
        except tk.TclError as exc:
            self._queue_test_output(f"Falha ao exibir imagem PNG: {exc}")

    ########################################
    def _run_camera_test(self, name, ip, camera_mode):
        """Inicia camera.py --stream por SSH e exibe /frame.png na janela Tkinter."""
        workdir = (self.dest_entry.get().strip() or DEFAULT_DEST).rstrip("/")
        run_id = self.test_run_id
        log_dir, log_path = remote_log_path(workdir, "teste_camera_" + _slug(camera_mode))
        exit_path = f"/tmp/fva_camera_http_{run_id}.exit"
        port = 8081  # independente da porta 8080 utilizada pelo main.py
        client = None
        pid = None
        log_offset = 0
        last_error = None
        try:
            client = self._connect_ssh(ip)
            mode_flag = {
                "Normal": "--normal",
                "ArUco": "--aruco",
                "Placas": "--placas",
            }[camera_mode]
            command = (
                f"cd {shlex.quote(workdir)} && python3 -u fva_car/camera.py "
                f"--stream --port {port} --fps 10 --scale 0.5 --seconds 60 {mode_flag}"
            )
            inner = (
                f"{command}; rc=$?; printf '%s\\n' \"$rc\" > {shlex.quote(exit_path)}"
            )
            launch = (
                f"mkdir -p {shlex.quote(log_dir)}; "
                f"nohup setsid sh -c {shlex.quote(inner)} "
                f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo FVA_PID:$!"
            )
            _, out, err = client.exec_command(launch, timeout=20)
            response = out.readline().strip()
            if not response.startswith("FVA_PID:"):
                raise RuntimeError(
                    f"Falha ao iniciar câmera: {response} {err.read(300).decode(errors='replace')}"
                )
            pid = int(response.split(":", 1)[1])
            self.test_remote_pid = pid
            self._queue_test_output(f"$ {command}")
            self._queue_test_output(
                f"Visualização HTTP: {ip}:{port} (10 FPS, escala 0.5)"
            )
            if not self.test_running:
                return
            url = f"http://{ip}:{port}/frame.png"
            while self.test_running and self.test_run_id == run_id:
                try:
                    with urllib.request.urlopen(url, timeout=1.5) as response:
                        frame = response.read(2_000_000)
                    if self.test_running and self.test_run_id == run_id:
                        self.ui(self._show_camera_frame, frame)
                    last_error = None
                except Exception as exc:
                    if str(exc) != last_error:
                        self._queue_test_output(f"Aguardando câmera HTTP: {exc}")
                        last_error = str(exc)
                    time.sleep(0.4)
                # Se o processo terminou, encerra o teste (sem SFTP/PNG).
                try:
                    _, stdout, _ = client.exec_command(
                        f"test -f {shlex.quote(exit_path)} && cat {shlex.quote(exit_path)}",
                        timeout=4,
                    )
                    status = stdout.read().decode().strip()
                    if status:
                        self._queue_test_output(
                            f"Teste da câmera encerrado (código {status})."
                        )
                        break
                except Exception:
                    pass
                time.sleep(0.10)
        except Exception as exc:
            self._queue_test_output(f"Erro no teste da câmera: {exc}")
        finally:
            if pid is not None:
                self._stop_remote_test_process(ip, pid)
            if client:
                try:
                    client.close()
                except Exception:
                    pass
            if self.test_run_id == run_id:
                self.test_remote_pid = None
                self.test_remote_ip = None
                self.test_running = False
                self.ui(self.test_run_button.configure, state="normal")
                self.ui(self.test_stop_button.configure, state="disabled")

    ########################################
    def _run_module_test(self, name, ip, module_path):
        """Executa módulo em sessão independente e acompanha log persistente por SFTP."""
        remote_workdir = (self.dest_entry.get().strip() or DEFAULT_DEST).rstrip("/")
        run_id = self.test_run_id
        log_dir, log_path = remote_log_path(remote_workdir, "teste_" + _slug(Path(module_path).stem))
        exit_path = f"/tmp/fva_teste_{run_id}.exit"
        client = None
        sftp = None
        buffer = b""
        offset = 0
        pid = None
        finished = False
        connection_errors = 0
        emit = self._queue_test_output

        try:
            command = f"cd {shlex.quote(remote_workdir)} && python3 -u {shlex.quote(module_path)}"
            emit(f"$ {command}")
            emit("")
            # setsid cria grupo próprio, permitindo que Parar encerre o módulo e filhos.
            inner = (
                f"{command}; rc=$?; printf '%s\\n' \"$rc\" > {shlex.quote(exit_path)}"
            )
            launch = (
                f"mkdir -p {shlex.quote(log_dir)}; "
                f"nohup setsid sh -c {shlex.quote(inner)} "
                f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo FVA_PID:$!"
            )
            client = self._connect_ssh(ip)
            _, out, err = client.exec_command(launch, timeout=20)
            response = out.readline().strip()
            if not response.startswith("FVA_PID:"):
                raise RuntimeError(
                    f"Falha ao iniciar teste: {response} {err.read(500).decode(errors='replace')}"
                )
            pid = int(response.split(":", 1)[1])
            self.test_remote_pid = pid
            emit(f"Log no carro (vem no Coletar Dados): {log_path}")
            # Se Parar foi pressionado enquanto a conexão inicial era estabelecida.
            if not self.test_running:
                self._stop_remote_test_process(ip, pid)
            while self.test_running and not finished:
                try:
                    if (
                        client is None
                        or not client.get_transport()
                        or not client.get_transport().is_active()
                    ):
                        if client:
                            client.close()
                        client = self._connect_ssh(ip)
                    if sftp is None:
                        sftp = client.open_sftp()
                        sftp.get_channel().settimeout(10)
                    with sftp.open(log_path, "rb") as remote_file:
                        remote_file.seek(offset)
                        chunk = remote_file.read(65536)
                    if chunk:
                        offset += len(chunk)
                        buffer += chunk
                        # Retorno de carro também encerra uma atualização dinâmica do sensor.
                        while True:
                            match = re.search(rb"\r\n|\r|\n", buffer)
                            if match is None:
                                break
                            line, buffer = (
                                buffer[: match.start()],
                                buffer[match.end() :],
                            )
                            if line:
                                emit(
                                    line.decode("utf-8", errors="replace")
                                )
                    try:
                        with sftp.open(exit_path, "r") as status_file:
                            rc = int(status_file.read().strip())
                        finished = True
                    except IOError:
                        pass
                    connection_errors = 0
                except (OSError, EOFError, paramiko.SSHException) as exc:
                    connection_errors += 1
                    if sftp:
                        try:
                            sftp.close()
                        except Exception:
                            pass
                    sftp = None
                    if client:
                        try:
                            client.close()
                        except Exception:
                            pass
                    client = None
                    if connection_errors == 1:
                        emit(
                            f"Comunicação interrompida; recuperando log: {exc}"
                        )
                    if connection_errors >= 15:
                        raise RuntimeError(
                            f"Sem SSH após 15 tentativas; log preservado em {log_path}"
                        ) from exc
                time.sleep(0.3 if finished else 0.4)
            if buffer:
                emit(buffer.decode("utf-8", errors="replace"))
            if finished and self.test_running:
                emit(f"\nTeste finalizado (código {rc}).")
            else:
                emit("\nTeste interrompido.")
        except Exception as exc:
            if self.test_running:
                emit(
                    f"Erro no teste de {name.upper()} ({ip}): {exc}"
                )
        finally:
            if sftp:
                try:
                    sftp.close()
                except Exception:
                    pass
            if client:
                try:
                    client.close()
                except Exception:
                    pass
            # Não deixa um teste remoto rodando sem supervisão se a leitura falhar.
            if not finished and pid is not None:
                self._stop_remote_test_process(ip, pid)
            self.test_remote_pid = None
            self.test_remote_ip = None
            self.test_running = False
            self.ui(self.test_run_button.configure, state="normal")
            self.ui(self.test_stop_button.configure, state="disabled")

    ########################################
    def _stop_remote_test_process(self, ip, pid):
        """Encerra apenas o grupo de processos criado para este teste."""
        client = None
        try:
            client = self._connect_ssh(ip)
            # PID de setsid é também o identificador do grupo de processos.
            code = "import os,signal; " f"os.killpg({int(pid)}, signal.SIGTERM)"
            _, out, err = client.exec_command(
                f"python3 -c {shlex.quote(code)}", timeout=10
            )
            if out.channel.recv_exit_status() != 0:
                self._queue_test_output(
                    "Aviso: não foi possível confirmar parada do processo remoto."
                )
        except Exception as exc:
            self._queue_test_output(f"Aviso ao interromper teste remoto: {exc}")
        finally:
            if client:
                client.close()

    ########################################
    def stop_module_test(self):
        if not self.test_running:
            return
        self.test_running = False
        self.test_stop_button.configure(state="disabled")
        # A thread de acompanhamento encerra o processo remoto no finally.
        self._queue_test_output("Solicitada interrupção do teste...")

    ########################################
    def _build_tab_data(self, parent):

        ttk.Label(
            parent, text="Coletar dados dos experimentos armazenados no carrinho."
        ).pack(anchor="w", padx=10, pady=(10, 6))

        # pasta local
        local_frame = ttk.Frame(parent)
        local_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(local_frame, text="Destino no computador:").pack(side="left")

        self.data_dest_entry = ttk.Entry(local_frame)
        self.data_dest_entry.insert(0, os.path.join(os.getcwd(), "experimentos"))
        self.data_dest_entry.pack(side="left", fill="x", expand=True, padx=8)

        ttk.Button(
            local_frame, text="Selecionar...", command=self.select_data_destination
        ).pack(side="left")

        # botao de coleta
        ttk.Button(parent, text="📥 Coletar dados", command=self.collect_data).pack(
            anchor="w", padx=10, pady=6
        )

        # Mensagens da coleta (altura reduzida para liberar espaço)
        ttk.Label(parent, text="Transferências:").pack(anchor="w", padx=10, pady=(4, 2))
        self.data_log = scrolledtext.ScrolledText(parent, height=4)
        self.data_log.pack(fill="x", padx=10, pady=(0, 6))
        self.data_log.configure(state="disabled")

        # Seleção de experimentos já baixados
        selector = ttk.Frame(parent)
        selector.pack(fill="x", padx=10, pady=(2, 5))
        ttk.Label(selector, text="Experimento:").pack(side="left")
        self.log_file_var = tk.StringVar()
        self.log_file_combo = ttk.Combobox(
            selector, textvariable=self.log_file_var, state="readonly", width=38
        )
        self.log_file_combo.pack(side="left", fill="x", expand=True, padx=6)
        self.log_file_combo.bind(
            "<<ComboboxSelected>>", lambda event: self.load_experiment()
        )
        ttk.Button(
            selector, text="Atualizar lista", command=self.refresh_experiments
        ).pack(side="left")

        self.experiment_graphs = [
            ("Velocidade", "Velocidade medida e referência"),
            ("Aceleração", "Aceleração: estimada, modelo e IMU"),
            ("Velocidade angular", "Velocidade angular: estimada, modelo e IMU"),
            ("Orientação", "Orientação × tempo"),
            ("Controle", "Comando de aceleração × tempo"),
            ("Trajetória XY", "Trajetória XY"),
            ("Cones", "Cones: contagem, perda/hold e trecho"),
        ]
        options = ttk.Frame(parent)
        options.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(options, text="Gráficos:").pack(side="left")
        self.graph_vars = {}
        for key, _ in self.experiment_graphs:
            var = tk.BooleanVar(value=True)
            self.graph_vars[key] = var
            ttk.Checkbutton(
                options, text=key, variable=var, command=self.render_experiment_graphs
            ).pack(side="left", padx=4)

        # Área de gráficos com rolagem vertical
        graph_area = ttk.Frame(parent)
        graph_area.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.graph_scroll = tk.Canvas(graph_area, highlightthickness=0)
        graph_bar = ttk.Scrollbar(
            graph_area, orient="vertical", command=self.graph_scroll.yview
        )
        self.graph_scroll.configure(yscrollcommand=graph_bar.set)
        graph_bar.pack(side="right", fill="y")
        self.graph_scroll.pack(side="left", fill="both", expand=True)
        self.graph_inner = ttk.Frame(self.graph_scroll)
        self.graph_window = self.graph_scroll.create_window(
            (0, 0), window=self.graph_inner, anchor="nw"
        )
        self.graph_inner.bind(
            "<Configure>",
            lambda e: self.graph_scroll.configure(
                scrollregion=self.graph_scroll.bbox("all")
            ),
        )
        self.graph_scroll.bind(
            "<Configure>",
            lambda e: self.graph_scroll.itemconfigure(self.graph_window, width=e.width),
        )
        self.experiment_data = None
        self.experiment_paths = {}
        self.experiment_canvases = []
        self.refresh_experiments()

    ########################################
    def refresh_experiments(self):
        """Localiza car.csv dentro das pastas já coletadas, sem usar SSH."""
        base = self.data_dest_entry.get().strip()
        paths = {}
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                dirs.sort()
                if "car.csv" in files:
                    path = os.path.join(root, "car.csv")
                    label = os.path.relpath(root, base)
                    paths[label] = path
        self.experiment_paths = dict(sorted(paths.items(), reverse=True))
        self.log_file_combo.configure(values=list(self.experiment_paths))
        current = self.log_file_var.get()
        if current not in self.experiment_paths:
            self.log_file_var.set(next(iter(self.experiment_paths), ""))
            self.load_experiment()
        elif self.experiment_data is None:
            self.load_experiment()

    ########################################
    def load_experiment(self):
        """Lê o CSV local preservando os nomes das colunas do car.py."""
        path = self.experiment_paths.get(self.log_file_var.get())
        self.experiment_data = None
        if path:
            try:
                with open(path, newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    required = {
                        "t",
                        "x",
                        "y",
                        "v",
                        "vref",
                        "a",
                        "a_model",
                        "a_x",
                        "w",
                        "w_model",
                        "w_imu",
                        "th",
                        "u",
                    }
                    if not reader.fieldnames or not required.issubset(
                        reader.fieldnames
                    ):
                        raise ValueError("Colunas necessárias ausentes no car.csv")
                    # Colunas do modo cones: opcionais, ausentes nos ensaios antigos.
                    optional = [key for key in CONE_COLUMNS if key in reader.fieldnames]
                    data = {key: [] for key in (*required, *optional)}
                    for row in reader:
                        for key in required:
                            data[key].append(float(row[key]))
                        for key in optional:
                            try:
                                data[key].append(float(row[key]))
                            except (TypeError, ValueError):
                                data[key].append(float("nan"))
                if not data["t"]:
                    raise ValueError("Arquivo sem amostras")
                self.experiment_data = data
                self.show_cone_summary(os.path.dirname(path))
            except (OSError, ValueError, TypeError) as exc:
                messagebox.showerror("Erro ao ler experimento", f"{path}\n\n{exc}")
        self.render_experiment_graphs()

    ########################################
    def show_cone_summary(self, folder):
        """Resumo de cones_summary.json (gravado pelo main.py no modo cones)."""
        path = os.path.join(folder, "cones_summary.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, ValueError) as exc:
            self.datalog_write(f"Resumo de cones ilegivel: {exc}")
            return
        frames = summary.get("frames") or 0
        pct = lambda key: (
            f"{100 * (summary.get(key) or 0) / frames:.0f}%" if frames else "-"
        )
        events = (
            ", ".join(f"{k}={v}" for k, v in (summary.get("events") or {}).items())
            or "nenhum"
        )
        fps = summary.get("vision_fps")
        self.datalog_write(
            f"Cones: {frames} quadros | validos {pct('valid')} | estimados {pct('estimated')}"
            f" | hold {pct('hold')} | maior perda {summary.get('longest_invalid_frames')} quadros"
            f" | {fps:.1f} fps"
            if fps
            else f"Cones: {frames} quadros | validos {pct('valid')}"
        )
        self.datalog_write(
            f"   eventos: {events} | trechos: {summary.get('segments')}"
            f" | frames salvos {summary.get('event_frames_saved')}"
            f" | descartes {summary.get('dropped_records')}"
        )

    ########################################
    def plot_experiment_axis(self, ax, key):
        d = self.experiment_data
        if key == "Trajetória XY":
            ax.plot(d["x"], d["y"], label="Trajetória")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal", adjustable="datalim")
        elif key == "Cones":
            if "cone_green_count" not in d:
                ax.text(
                    0.5,
                    0.5,
                    "Sem colunas de cones neste car.csv",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                ax.plot(
                    d["t"], d["cone_green_count"], color="tab:green", label="Verdes"
                )
                ax.plot(d["t"], d["cone_red_count"], color="tab:red", label="Vermelhos")
                ax.set_ylabel("cones")
                twin = ax.twinx()
                for field, label, color in (
                    ("line_valid", "Corredor válido", "tab:blue"),
                    ("line_hold", "Hold (tolerância)", "tab:orange"),
                    ("cone_segment", "Trecho (+1 esq, -1 dir)", "tab:purple"),
                ):
                    if field in d:
                        twin.step(
                            d["t"],
                            d[field],
                            where="post",
                            label=label,
                            color=color,
                            alpha=0.7,
                        )
                twin.set_ylim(-1.2, 1.2)
                lines, labels = ax.get_legend_handles_labels()
                more, more_labels = twin.get_legend_handles_labels()
                ax.legend(
                    lines + more, labels + more_labels, fontsize=7, loc="upper left"
                )
            ax.set_xlabel("Tempo [s]")
        else:
            series = {
                "Velocidade": [("v", "Medida"), ("vref", "Referência")],
                "Aceleração": [
                    ("a", "Estimada"),
                    ("a_model", "Modelo"),
                    ("a_x", "IMU"),
                ],
                "Velocidade angular": [
                    ("w", "Estimada"),
                    ("w_model", "Modelo"),
                    ("w_imu", "IMU"),
                ],
                "Orientação": [("th", "Orientação")],
                "Controle": [("u", "Comando")],
            }
            for field, label in series[key]:
                ax.plot(d["t"], d[field], label=label)
            ax.set_xlabel("Tempo [s]")
            ax.set_ylabel(
                {
                    "Velocidade": "m/s",
                    "Aceleração": "m/s²",
                    "Velocidade angular": "rad/s",
                    "Orientação": "rad",
                    "Controle": "u",
                }[key]
            )
            ax.legend(fontsize=8)
        ax.grid(True)

    ########################################
    def render_experiment_graphs(self):
        for child in self.graph_inner.winfo_children():
            child.destroy()
        self.experiment_canvases = []
        if self.experiment_data is None:
            ttk.Label(
                self.graph_inner,
                text="Selecione um experimento local para visualizar os gráficos.",
            ).pack(pady=25)
            return
        for column in (0, 1):
            self.graph_inner.columnconfigure(column, weight=1, uniform="graphs")
        keys = [key for key, _ in self.experiment_graphs if self.graph_vars[key].get()]
        for i, key in enumerate(keys):
            frame = ttk.LabelFrame(self.graph_inner, text=key)
            frame.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            fig = Figure(figsize=(4.5, 2.7), dpi=90)
            ax = fig.add_subplot(111)
            self.plot_experiment_axis(ax, key)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw_idle()
            self.experiment_canvases.append(canvas)
            ttk.Button(
                frame,
                text="Ampliar",
                command=lambda k=key: self.open_experiment_graph(k),
            ).pack(anchor="e", padx=5, pady=2)

    ########################################
    def open_experiment_graph(self, key):
        if self.experiment_data is None:
            return
        window = tk.Toplevel(self)
        window.title(f"FVA — {key} — {self.log_file_var.get()}")
        window.geometry("1000x650")
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.plot_experiment_axis(ax, key)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()
        window.graph_canvas = canvas

    ########################################
    # Funções utilitárias comuns
    ########################################
    def log_write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    ########################################
    def clear_cmd_log(self):
        self.cmd_log.configure(state="normal")
        self.cmd_log.delete("1.0", "end")
        self.cmd_log.configure(state="disabled")

    ########################################
    def cmdlog_write(self, text):
        self.cmd_log.configure(state="normal")

        ansi_colors = {
            "30": "ansi_black",
            "31": "ansi_red",
            "32": "ansi_green",
            "33": "ansi_yellow",
            "34": "ansi_blue",
            "35": "ansi_magenta",
            "36": "ansi_cyan",
            "37": "ansi_white",
            # cores ANSI brilhantes
            "90": "ansi_black",
            "91": "ansi_red",
            "92": "ansi_green",
            "93": "ansi_yellow",
            "94": "ansi_blue",
            "95": "ansi_magenta",
            "96": "ansi_cyan",
            "97": "ansi_white",
        }

        current_tag = None
        pos = 0

        for match in ANSI_ESCAPE_RE.finditer(text):

            # texto antes do código ANSI
            part = text[pos : match.start()]

            if part:
                if current_tag:
                    self.cmd_log.insert("end", part, current_tag)
                else:
                    self.cmd_log.insert("end", part)

            # interpreta o código ANSI
            codes = match.group()[2:-1].split(";")

            for code in codes:
                if code == "0":
                    current_tag = None
                elif code in ansi_colors:
                    current_tag = ansi_colors[code]

            pos = match.end()

        # restante da linha
        part = text[pos:]

        if part:
            if current_tag:
                self.cmd_log.insert("end", part, current_tag)
            else:
                self.cmd_log.insert("end", part)

        self.cmd_log.insert("end", "\n")
        self.cmd_log.see("end")
        self.cmd_log.configure(state="disabled")

    ########################################
    def select_files(self):
        # raiz do projeto: um nível acima da pasta tools
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        files = filedialog.askopenfilenames(
            title="Selecione arquivos (Ctrl/Shift para múltiplos)",
            initialdir=project_dir,
        )

        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
                self.files_listbox.insert("end", f)

    ########################################
    def add_directory(self):
        # raiz do projeto: um nível acima da pasta tools
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        d = filedialog.askdirectory(title="Selecione uma pasta", initialdir=project_dir)

        if d:
            path = os.path.join(d, "")
            if path not in self.selected_files:
                self.selected_files.append(path)
                self.files_listbox.insert("end", path)

    ########################################
    def remove_selected(self):
        sel = list(self.files_listbox.curselection())
        for idx in reversed(sel):
            val = self.files_listbox.get(idx)
            self.files_listbox.delete(idx)
            try:
                self.selected_files.remove(val)
            except ValueError:
                pass

    ########################################
    def clear_files(self):
        self.files_listbox.delete(0, "end")
        self.selected_files = []

    ########################################
    def refresh_ips(self):
        """Detecta os veículos cadastrados e usa somente o primeiro encontrado."""
        self.ui(self.log_write, "🔍 Procurando veículo...")
        self.ui(
            self.active_device_label.config, text="🔍 Procurando veículo...", fg="black"
        )

        # Gera algum tráfego antes de consultar a tabela ARP.
        possible_hosts = ["raspberrypi.local", "raspberrypi"]
        for info in self.devices.values():
            if info.get("ip"):
                ping_host(info["ip"])
        for host in possible_hosts:
            ping_host(host)

        arp_entries = parse_arp_table()
        arp_by_mac = {mac: ip for ip, mac in arp_entries}

        self.active_device = None

        for name, info in self.devices.items():
            try:
                mac = normalize_mac(info["mac"])
            except ValueError:
                mac = ""

            ip = arp_by_mac.get(mac)
            info["ip"] = ip

            if ip and self.active_device is None:
                self.active_device = (name, ip)

        if self.active_device:
            name, ip = self.active_device
            self.ui(
                self.active_device_label.config,
                text=f"{CAR_ICON}{name.upper()} — {ip}",
                fg=COLORS.get(name, "#00aa00"),
            )
            self.ui(self.log_write, f"🚗 Veículo detectado: {name.upper()} ({ip})")
            self.ui(
                self.test_device_label.config, text=f"Veículo: {name.upper()} — {ip}"
            )
        else:
            self.ui(
                self.active_device_label.config,
                text="⚠️ Nenhum veículo encontrado",
                fg="#cc0000",
            )
            self.ui(self.log_write, "⚠️ Nenhum veículo encontrado.")
            self.ui(
                self.test_device_label.config, text="Veículo: nenhum veículo encontrado"
            )

        self.ui(self.log_write, "✅ Atualização concluída.")

    ########################################
    def get_selected_devices(self):
        """Retorna somente o primeiro veículo detectado (veículo ativo)."""
        return [self.active_device] if self.active_device else []

    ########################################
    # Envio de arquivos (aba 1)
    ########################################
    def send_to_selected(self):
        targets = self.get_selected_devices()
        if not targets:
            messagebox.showinfo(
                "Nenhum alvo",
                "Nenhum veículo foi detectado. Atualize os IPs e tente novamente.",
            )
            return
        threading.Thread(
            target=self._run_sftp_for_targets, args=(targets,), daemon=True
        ).start()

    ########################################
    def send_to_all(self):
        # Mantido apenas por compatibilidade; usa somente o veículo ativo.
        self.send_to_selected()

    ########################################
    def _connect_ssh(self, ip):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            ip,
            username=self.user_entry.get().strip() or SSH_USER,
            password=self.pass_entry.get().strip() or None,
            timeout=20,
            auth_timeout=20,
            banner_timeout=20,
        )
        return client

    ########################################
    def _sftp_mkdir_p(self, sftp, remote_dir):
        remote_dir = posixpath.normpath(remote_dir)
        parts = remote_dir.strip("/").split("/") if remote_dir != "/" else []
        current = "/" if remote_dir.startswith("/") else ""
        for part in parts:
            current = posixpath.join(current, part)
            try:
                sftp.stat(current)
            except IOError:
                sftp.mkdir(current)

    ########################################
    def _sftp_upload_file(self, sftp, local_path, remote_path):
        self._sftp_mkdir_p(sftp, posixpath.dirname(remote_path))
        sftp.put(local_path, remote_path)

    ########################################
    def _sftp_upload_directory_contents(self, sftp, local_dir, remote_dir):
        self._sftp_mkdir_p(sftp, remote_dir)
        for root, dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            remote_root = (
                remote_dir
                if rel == "."
                else posixpath.join(remote_dir, *rel.split(os.sep))
            )
            self._sftp_mkdir_p(sftp, remote_root)
            for dirname in dirs:
                self._sftp_mkdir_p(sftp, posixpath.join(remote_root, dirname))
            for filename in files:
                self._sftp_upload_file(
                    sftp,
                    os.path.join(root, filename),
                    posixpath.join(remote_root, filename),
                )

    ########################################
    def _run_sftp_for_targets(self, targets):
        dest = (self.dest_entry.get().strip() or DEFAULT_DEST).rstrip("/")

        if not self.selected_files:
            self.ui(self.log_write, "⚠️ Nenhum arquivo ou pasta selecionado.")
            return

        missing = [p for p in self.selected_files if not os.path.exists(p)]
        if missing:
            self.ui(self.log_write, "❌ Itens inexistentes:")
            for item in missing:
                self.ui(self.log_write, "   - " + item)
            return

        force = bool(getattr(self, "sync_force_var", None) and self.sync_force_var.get())
        delete_py = bool(getattr(self, "sync_delete_var", None) and self.sync_delete_var.get())
        log = lambda text: self.ui(self.log_write, text)
        for name, ip in targets:
            self.ui(self.log_write, "=" * 60)
            self.ui(self.log_write, f"🚀 Enviando via SFTP para {name.upper()} ({ip})"
                    + (" — tudo" if force else " — só o que mudou"))
            client = None
            sftp = None
            try:
                client = self._connect_ssh(ip)
                sftp = client.open_sftp()
                self._sftp_mkdir_p(sftp, dest)

                for path in self.selected_files:
                    if os.path.isdir(path):
                        # Mesmo comportamento do antigo rsync com barra final:
                        # envia o CONTEÚDO da pasta para o destino.
                        self.ui(self.log_write, f"📁 {path} -> {dest}/")
                        summary = sync_directory(client, sftp, path, dest, log, force, delete_py)
                        self.ui(self.log_write,
                                f"   {summary['sent']} enviado(s) ({summary['bytes']/1024:.0f} kB),"
                                f" {summary['same']} igual(is) pulado(s)"
                                + (f", {summary['removed']} apagado(s)" if summary["removed"] else "")
                                + f" — {summary['seconds']:.1f} s")
                    else:
                        remote_path = posixpath.join(dest, os.path.basename(path))
                        self.ui(self.log_write, f"📄 {path} -> {remote_path}")
                        self._sftp_upload_file(sftp, path, remote_path)

                self.ui(self.log_write, f"✅ Sucesso: {name.upper()} ({ip})")
            except Exception as e:
                self.ui(self.log_write, f"❌ Erro em {name.upper()} ({ip}): {e}")
            finally:
                if sftp:
                    sftp.close()
                if client:
                    client.close()

        self.ui(self.log_write, "🏁 Todas as transferências finalizadas.")

    ########################################
    # Execução remota (aba 2)
    ########################################
    def apply_speed_preset(self):
        self.apply_mission_options()

    def open_camera_stream(self):
        """A missao precisa estar rodando com --stream; abre no navegador local."""
        targets = self.get_selected_devices()
        if not targets:
            messagebox.showinfo(
                "Nenhum alvo",
                "Nenhum veículo foi detectado. Atualize os IPs e tente novamente.",
            )
            return
        webbrowser.open(f"http://{targets[0][1]}:{STREAM_PORT}/")

    def run_cmds_on_selected(self):
        targets = self.get_selected_devices()
        if not targets:
            messagebox.showinfo(
                "Nenhum alvo",
                "Nenhum veículo foi detectado. Atualize os IPs e tente novamente.",
            )
            return
        cmds = [
            c.strip() for c in self.cmd_text.get("1.0", "end").splitlines() if c.strip()
        ]
        if not cmds:
            messagebox.showinfo("Nenhum comando", "Digite ao menos um comando.")
            return

        self.telemetry = {}
        while not self.telemetry_queue.empty():
            try:
                self.telemetry_queue.get_nowait()
            except queue.Empty:
                break
        self.update_plot()

        threading.Thread(
            target=self._run_remote_cmds, args=(targets, cmds), daemon=True
        ).start()

    ########################################
    def flash_arduino(self):
        """Compila e grava o firmware do Nano via Arduino CLI na Raspberry."""
        targets = self.get_selected_devices()
        if not targets:
            messagebox.showinfo(
                "Nenhum veículo", "Nenhum veículo foi detectado. Atualize os IPs."
            )
            return
        if not messagebox.askyesno(
            "Gravar Arduino Nano",
            "Gravar firmware/odometer.ino no Arduino Nano (Old Bootloader)?\n\n"
            "O programa atual do Arduino será substituído. "
            "Verifique se o veículo está imobilizado e seguro.",
        ):
            return

        # Executado integralmente na Raspberry, sem dependências adicionais na GUI.
        # A cópia temporária respeita a exigência da Arduino CLI de pasta/sketch homônimos.
        script = r"""set -eu
if ! command -v arduino-cli >/dev/null 2>&1; then
    echo 'ERRO: arduino-cli não instalado na Raspberry.'
    echo 'Instale o Arduino CLI e o core arduino:avr antes de gravar.'
    exit 1
fi
if [ ! -f firmware/odometer.ino ]; then
    echo 'ERRO: firmware/odometer.ino não encontrado na pasta remota.'
    exit 1
fi
if ! arduino-cli core list | grep -q 'arduino:avr'; then
    echo 'ERRO: core arduino:avr não instalado.'
    echo 'Execute: arduino-cli core install arduino:avr'
    exit 1
fi
# Preferir o caminho persistente por-id; sem ele, usar a única ttyUSB/ttyACM.
set -- /dev/serial/by-id/*
if [ -e "$1" ]; then
    candidates=''
    for port in "$@"; do
        case "$(readlink -f "$port")" in
            /dev/ttyUSB*|/dev/ttyACM*) candidates="$candidates $port" ;;
        esac
    done
    set -- $candidates
else
    set -- /dev/ttyUSB* /dev/ttyACM*
fi
ports=''
for port in "$@"; do
    [ -e "$port" ] && ports="$ports $port"
done
set -- $ports
if [ "$#" -ne 1 ]; then
    echo "ERRO: esperado exatamente um dispositivo serial Arduino; encontrados $# ($ports)."
    exit 1
fi
port="$1"
if command -v fuser >/dev/null 2>&1 && fuser "$port" >/dev/null 2>&1; then
    echo "ERRO: porta $port está sendo usada por outro processo."
    fuser -v "$port" || true
    exit 1
fi
sketch_dir="$(mktemp -d /tmp/fva_arduino_XXXXXX)"
trap 'rm -rf "$sketch_dir"' EXIT
mkdir -p "$sketch_dir/odometer"
cp firmware/odometer.ino "$sketch_dir/odometer/odometer.ino"
echo "Arduino Nano (Old Bootloader) | Porta: $port"
echo 'Compilando firmware...'
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328old "$sketch_dir/odometer"
echo 'Gravando Arduino...'
arduino-cli upload --fqbn arduino:avr:nano:cpu=atmega328old -p "$port" "$sketch_dir/odometer"
echo 'SUCESSO: firmware gravado no Arduino Nano.'"""
        command = "sh -c " + shlex.quote(script)
        # A gravação também utiliza o terminal da aba Testar Módulos.
        self.clear_test_log()
        threading.Thread(
            target=self._run_remote_cmds, args=(targets, [command], True), daemon=True
        ).start()

    ########################################
    def _handle_remote_log_line(self, name, line):
        """Processa a mesma telemetria DATA usada anteriormente."""
        line = line.rstrip("\r\n")
        if not line:
            return
        if ANSI_ESCAPE_RE.sub("", line).lstrip().startswith("DATA,"):
            try:
                self.telemetry_queue.put((name, parse_telemetry_line(line)))
            except ValueError as exc:
                self.ui(
                    self.cmdlog_write,
                    f"[{name.upper()}] Telemetria inválida: {exc} | {line!r}",
                )
        else:
            self.ui(self.cmdlog_write, f"[{name.upper()}] {line}")

    def _run_remote_cmds(self, targets, cmds, arduino_flash=False):
        """Executa no Raspberry e lê log persistente via SFTP com retomada por offset.

        O processo não depende da sessão que lê a saída. O marcador .exit
        indica a conclusão, mesmo se a conexão de leitura cair.
        """
        remote_workdir = (self.dest_entry.get().strip() or DEFAULT_DEST).rstrip("/")

        def report(message):
            if arduino_flash:
                self.ui(self.testlog_write, message)
            else:
                self.ui(self.cmdlog_write, message)

        for name, ip in targets:
            if arduino_flash:
                report(f"🔄 {name.upper()}: compilando e gravando Arduino Nano...")
            else:
                report("\n" + "=" * 60)
                report(
                    f"💻 Executando carro {name.upper()} ({ip}) - workdir: {remote_workdir}"
                )
            for raw_cmd in cmds:
                client = None
                sftp = None
                try:
                    # Nome exclusivo evita misturar execuções antigas e novas.
                    run_id = uuid.uuid4().hex
                    label = "gravar_arduino" if arduino_flash else command_label(raw_cmd)
                    log_dir, log_path = remote_log_path(remote_workdir, label)
                    exit_path = f"/tmp/fva_gui_{run_id}.exit"
                    if not arduino_flash:
                        report(f"$ cd {remote_workdir} && {raw_cmd}")
                    # Processo remoto em segundo plano: grava stdout/stderr na Raspberry.
                    # O status é escrito em arquivo separado APÓS a saída terminar.
                    inner = (
                        f"cd {shlex.quote(remote_workdir)} && {raw_cmd}; "
                        "rc=$?; "
                        f"printf '%s\\n' \"$rc\" > {shlex.quote(exit_path)}"
                    )
                    launch = (
                        f"mkdir -p {shlex.quote(log_dir)}; "
                        f"nohup sh -c {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 "
                        "< /dev/null & echo FVA_PID:$!"
                    )
                    client = self._connect_ssh(ip)
                    _, out, err = client.exec_command(launch, timeout=20)
                    response = out.readline().strip()
                    if not response.startswith("FVA_PID:"):
                        raise RuntimeError(
                            f"Não foi possível iniciar comando remoto: {response} {err.read(500).decode(errors='replace')}"
                        )
                    if not arduino_flash:
                        report(f"📄 Log no carro (vem no Coletar Dados): {log_path}")
                    flash_lines = []
                    # Reabre SFTP se necessário; mantém offset para não duplicar dados.
                    offset = 0
                    buffer = b""
                    finished = False
                    connection_errors = 0
                    while not finished:
                        try:
                            if (
                                client is None
                                or not client.get_transport()
                                or not client.get_transport().is_active()
                            ):
                                if client:
                                    client.close()
                                client = self._connect_ssh(ip)
                            if sftp is None:
                                sftp = client.open_sftp()
                            # Ler somente bytes novos, mesmo depois de reconexão.
                            with sftp.open(log_path, "rb") as remote_file:
                                remote_file.seek(offset)
                                chunk = remote_file.read(65536)
                            if chunk:
                                offset += len(chunk)
                                buffer += chunk
                                while b"\n" in buffer:
                                    line, buffer = buffer.split(b"\n", 1)
                                    decoded = line.decode("utf-8", errors="replace")
                                    if arduino_flash:
                                        flash_lines.append(decoded)
                                    else:
                                        self._handle_remote_log_line(name, decoded)
                            try:
                                with sftp.open(exit_path, "r") as status_file:
                                    rc = int(status_file.read().strip())
                                finished = True
                            except IOError:
                                pass  # comando ainda em execução
                            connection_errors = 0
                        except (OSError, EOFError, paramiko.SSHException) as exc:
                            connection_errors += 1
                            if sftp:
                                try:
                                    sftp.close()
                                except Exception:
                                    pass
                            sftp = None
                            if client:
                                try:
                                    client.close()
                                except Exception:
                                    pass
                            client = None
                            if connection_errors == 1:
                                report(
                                    f"⚠️ Comunicação interrompida; tentando recuperar saída: {exc}"
                                )
                            if connection_errors >= 15:
                                raise RuntimeError(
                                    "Sem comunicação SSH após 15 tentativas. O processo remoto pode continuar; log: "
                                    + log_path
                                ) from exc
                        time.sleep(0.25 if finished else 0.5)
                    if buffer:
                        if arduino_flash:
                            flash_lines.append(buffer.decode("utf-8", errors="replace"))
                        else:
                            self._handle_remote_log_line(
                                name, buffer.decode("utf-8", errors="replace")
                            )

                    if arduino_flash:
                        if rc == 0 and any(
                            "SUCESSO: firmware gravado" in line for line in flash_lines
                        ):
                            report(f"✅ {name.upper()}: firmware gravado com sucesso.")
                        else:
                            errors = [
                                line.strip()
                                for line in flash_lines
                                if "ERRO:" in line
                                or "Error" in line
                                or "error:" in line
                                or "avrdude:" in line
                            ]
                            reason = (
                                errors[-1] if errors else f"Código de retorno: {rc}"
                            )
                            report(
                                f"❌ {name.upper()}: falha na gravação. {reason} (log: {log_path})"
                            )
                    elif rc != 0:
                        report(f"⚠️ Retorno {rc} para comando: {raw_cmd}")
                    else:
                        report(f"✅ Comando finalizado (código 0)")
                except Exception as exc:
                    report(f"❌ Erro em {name.upper()} ({ip}): {exc}")
                    break
                finally:
                    if sftp:
                        try:
                            sftp.close()
                        except Exception:
                            pass
                    if client:
                        try:
                            client.close()
                        except Exception:
                            pass
            if not arduino_flash:
                report(f"🏁 Execução encerrada para {name.upper()}")

    ########################################
    # Execução remota (aba 3)
    ########################################
    def select_data_destination(self):
        directory = filedialog.askdirectory(
            title="Selecione onde salvar os dados dos experimentos"
        )

        if directory:
            self.data_dest_entry.delete(0, "end")
            self.data_dest_entry.insert(0, directory)
            self.refresh_experiments()

    ########################################
    def open_collected_data(self):
        """Abre a pasta onde o Coletar Dados salva (aba Coletar Dados)."""
        open_folder(os.path.abspath(self.data_dest_entry.get().strip() or "experimentos"))

    def collect_data(self):

        targets = self.get_selected_devices()

        if not targets:
            messagebox.showinfo(
                "Nenhum alvo",
                "Nenhum veículo foi detectado. Atualize os IPs e tente novamente.",
            )
            return

        local_base = self.data_dest_entry.get().strip()

        if not local_base:
            messagebox.showinfo(
                "Destino inválido", "Selecione uma pasta para salvar os dados."
            )
            return

        os.makedirs(local_base, exist_ok=True)

        threading.Thread(
            target=self._collect_data, args=(targets, local_base), daemon=True
        ).start()

    ########################################
    def _sftp_download_directory_contents(self, sftp, remote_dir, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = posixpath.join(remote_dir, entry.filename)
            local_path = os.path.join(local_dir, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                self._sftp_download_directory_contents(sftp, remote_path, local_path)
            elif (
                not os.path.isfile(local_path)
                or os.path.getsize(local_path) != entry.st_size
            ):
                # Coleta incremental: ensaios ja baixados (e frames de eventos)
                # nao sao transferidos de novo pelo Wi-Fi.
                sftp.get(remote_path, local_path)

    ########################################
    def _collect_legacy_tmp_logs(self, sftp, local_dest):
        """Saidas gravadas em /tmp pelas versoes antigas da GUI (fva_gui_<id>.log...).

        Nomeadas pela hora do arquivo no carro; baixadas so uma vez.
        """
        try:
            entries = sftp.listdir_attr("/tmp")
        except (OSError, IOError):
            return 0
        folder = os.path.join(local_dest, "terminal", "recuperados_tmp")
        count = 0
        for entry in entries:
            if not LEGACY_TMP_LOGS.match(entry.filename) or not entry.st_size:
                continue
            stamp = time.strftime("%Y-%m-%d_%Hh%Mm%S", time.localtime(entry.st_mtime or 0))
            local_path = os.path.join(folder, f"{stamp}_{entry.filename}")
            if os.path.isfile(local_path) and os.path.getsize(local_path) == entry.st_size:
                continue
            os.makedirs(folder, exist_ok=True)
            sftp.get(posixpath.join("/tmp", entry.filename), local_path)
            count += 1
        return count

    def _collect_data(self, targets, local_base):
        remote_workdir = (self.dest_entry.get().strip() or DEFAULT_DEST).rstrip("/")
        remote_logs = posixpath.join(remote_workdir, "logs")

        for name, ip in targets:
            local_dest = os.path.join(local_base, name)
            os.makedirs(local_dest, exist_ok=True)
            self.ui(
                self.datalog_write, f"📥 Coletando dados de {name.upper()} ({ip})..."
            )

            client = None
            sftp = None
            try:
                client = self._connect_ssh(ip)
                sftp = client.open_sftp()
                self._sftp_download_directory_contents(sftp, remote_logs, local_dest)
                recovered = self._collect_legacy_tmp_logs(sftp, local_dest)
                if recovered:
                    self.ui(self.datalog_write,
                            f"   + {recovered} log(s) antigo(s) de /tmp em terminal/recuperados_tmp/")
                self.ui(self.datalog_write, f"✅ Dados de {name.upper()} coletados em: {os.path.abspath(local_dest)}")
            except FileNotFoundError:
                self.ui(
                    self.datalog_write, f"❌ Pasta remota não encontrada: {remote_logs}"
                )
            except Exception as e:
                self.ui(self.datalog_write, f"❌ Erro em {name.upper()} ({ip}): {e}")
            finally:
                if sftp:
                    sftp.close()
                if client:
                    client.close()

        self.ui(self.datalog_write, "🏁 Coleta finalizada.")
        self.ui(self.refresh_experiments)

    ########################################
    def datalog_write(self, text):
        self.data_log.configure(state="normal")
        self.data_log.insert("end", text + "\n")
        self.data_log.see("end")
        self.data_log.configure(state="disabled")

    ########################################
    def _refresh_live_plot(self):
        """Consome telemetria na thread Tk e atualiza o gráfico no máximo a 5 Hz."""
        changed = False
        try:
            while True:
                name, sample = self.telemetry_queue.get_nowait()
                if name not in self.telemetry:
                    self.telemetry[name] = {key: deque(maxlen=1000) for key in sample}
                for key, value in sample.items():
                    self.telemetry[name][key].append(value)
                changed = True
        except queue.Empty:
            pass
        if changed:
            self.update_plot()
        self.after(200, self._refresh_live_plot)

    ########################################
    def update_plot(self):

        self.ax.clear()

        plot_type = self.plot_var.get()

        # configura os eixos mesmo sem dados
        if plot_type == "Velocidade":
            self.ax.set_xlabel("Tempo [s]")
            self.ax.set_ylabel("Velocidade [m/s]")

        elif plot_type == "Aceleração / Controle":
            self.ax.set_xlabel("Tempo [s]")
            self.ax.set_ylabel("a / u [m/s²]")

        elif plot_type == "Velocidade angular":
            self.ax.set_xlabel("Tempo [s]")
            self.ax.set_ylabel("Velocidade angular [rad/s]")

        elif plot_type == "Orientação":
            self.ax.set_xlabel("Tempo [s]")
            self.ax.set_ylabel("Orientação [rad]")

        elif plot_type == "Trajetória XY":
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            self.ax.set_aspect("equal", adjustable="datalim")

        # plota os dados, caso existam
        for name, data in self.telemetry.items():

            if plot_type == "Velocidade":

                self.ax.plot(data["t"], data["v"], label=f"{name.upper()} - v")

                self.ax.plot(
                    data["t"], data["vref"], "--", label=f"{name.upper()} - vref"
                )

            elif plot_type == "Aceleração / Controle":

                self.ax.plot(data["t"], data["a"], label=f"{name.upper()} - a")

                self.ax.plot(data["t"], data["u"], "--", label=f"{name.upper()} - u")

            elif plot_type == "Velocidade angular":

                self.ax.plot(data["t"], data["w"], label=f"{name.upper()} - w")

            elif plot_type == "Orientação":

                self.ax.plot(data["t"], data["th"], label=f"{name.upper()} - θ")

            elif plot_type == "Trajetória XY":

                self.ax.plot(data["x"], data["y"], label=name.upper())

        if self.telemetry:
            self.ax.legend()

        self.ax.grid(True)
        self.canvas.draw_idle()


########################################
# Execução
########################################
if __name__ == "__main__":
    app = RsyncGUI()
    app.mainloop()
