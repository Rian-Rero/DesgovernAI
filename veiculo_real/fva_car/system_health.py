"""Saude do Raspberry durante a volta: temperatura, clock, limitacao e nucleos.

O detector leva ~7 ms por quadro no notebook e ~89 ms no Pi (logs de 19/11),
12x, mais que a diferenca usual entre os dois. O Pi 4 reduz o clock sozinho
quando esquenta (limite leve a partir de ~80 C) ou com subtensao na bateria, e
isso nao aparecia em lugar nenhum. HealthMonitor le a cada poucos segundos:

    temperatura   /sys/class/thermal/thermal_zone0/temp
    clock         /sys/devices/system/cpu/cpu0/cpufreq (atual, maximo, governor)
    limitacao     vcgencmd get_throttled (bits "agora" e "desde que ligou")

e guarda o pior da volta no resumo (cones_summary.json e saude_raspberry.json).
Fora do Pi (notebook, testes) as leituras que faltam ficam None.

pin_process() fixa um processo (todas as threads) num conjunto de nucleos: o
controle num par, a visao no outro, sem um roubar o nucleo do outro.
"""
import os
from pathlib import Path
import subprocess
import threading
import time

# Bits de vcgencmd get_throttled (documentacao do Raspberry Pi).
THROTTLE_FLAGS = {
    0: "subtensao",
    1: "clock limitado",
    2: "throttling",
    3: "limite de temperatura",
}
SINCE_BOOT_SHIFT = 16
HOT_C = 70.0  # aviso: o limite leve de temperatura comeca em ~80 C
CPU_FOLDER = Path("/sys/devices/system/cpu/cpu0/cpufreq")
THERMAL_FILE = Path("/sys/class/thermal/thermal_zone0/temp")


def decode_throttled(value):
    """(problemas agora, problemas desde que o Pi ligou) a partir do inteiro."""
    if value is None:
        return [], []
    now = [name for bit, name in THROTTLE_FLAGS.items() if value >> bit & 1]
    since = [name for bit, name in THROTTLE_FLAGS.items() if value >> (bit + SINCE_BOOT_SHIFT) & 1]
    return now, since


def read_throttled(run=subprocess.run):
    """Inteiro de `vcgencmd get_throttled`, ou None fora do Pi / com falha."""
    try:
        output = run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1.0).stdout
        return int(output.strip().split("=")[-1], 16)
    except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
        return None


def _read_number(path, scale=1.0):
    try:
        return float(Path(path).read_text().strip()) * scale
    except (OSError, ValueError):
        return None


def read_temperature(path=THERMAL_FILE):
    """Temperatura do processador em C, ou None."""
    return _read_number(path, 1e-3)


def read_cpu(folder=CPU_FOLDER):
    """Clock atual e maximo (MHz) e governor do nucleo 0."""
    folder = Path(folder)
    try:
        governor = (folder / "scaling_governor").read_text().strip()
    except OSError:
        governor = None
    return {"freq_mhz": _read_number(folder / "scaling_cur_freq", 1e-3),
            "max_mhz": _read_number(folder / "cpuinfo_max_freq", 1e-3),
            "governor": governor}


def snapshot(run=subprocess.run, thermal=THERMAL_FILE, cpu=CPU_FOLDER):
    throttled = read_throttled(run)
    now, since = decode_throttled(throttled)
    return {"temp_c": read_temperature(thermal), **read_cpu(cpu), "throttled": throttled,
            "problems_now": now, "problems_since_boot": since}


class HealthMonitor:
    """Amostra a saude do Pi em segundo plano e resume o pior da volta."""
    def __init__(self, period_s=2.0, read=snapshot):
        self.period_s = float(period_s)
        self.read = read
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        values = self.read()
        values["t"] = time.monotonic()
        self.samples.append(values)
        return values

    def start(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True, name="saude-pi")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.period_s):
            try:
                self.sample()
            except Exception:  # diagnostico nunca derruba a missao
                pass

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.sample()  # estado no fim da volta
        except Exception:
            pass
        return self.summary()

    def summary(self):
        values = lambda key: [s[key] for s in self.samples if isinstance(s.get(key), (int, float))]
        temps, freqs, maxes = values("temp_c"), values("freq_mhz"), values("max_mhz")
        # Clock minimo so durante a volta: a 1a e a ultima leitura sao com o carro parado.
        running = [s for s in self.samples[1:-1]] if len(self.samples) >= 3 else self.samples
        freqs = [s["freq_mhz"] for s in running if isinstance(s.get("freq_mhz"), (int, float))]
        problems = sorted({p for s in self.samples for p in s.get("problems_now", [])})
        since = sorted({p for s in self.samples for p in s.get("problems_since_boot", [])})
        limited = sum(bool(s.get("problems_now")) for s in self.samples)
        summary = {
            "samples": len(self.samples),
            "temp_start_c": temps[0] if temps else None,
            "temp_max_c": max(temps) if temps else None,
            "freq_min_mhz": min(freqs) if freqs else None,
            "freq_max_mhz": max(maxes) if maxes else None,
            "governor": next((s["governor"] for s in self.samples if s.get("governor")), None),
            "problems_during_run": problems,
            "limited_samples": limited,
            "problems_since_boot": since,
        }
        summary["warnings"] = health_warnings(summary)
        return summary


def health_warnings(summary):
    """Avisos em portugues para o terminal e o relatorio do ensaio."""
    warnings = []
    if summary.get("problems_during_run"):
        warnings.append("Pi limitado DURANTE a volta (" + ", ".join(summary["problems_during_run"])
                        + f", {summary.get('limited_samples', 0)} leituras): visao mais lenta."
                        + (" Bateria fraca ou fonte do Pi ruim." if "subtensao" in summary["problems_during_run"]
                           else " Coloquem dissipador/ventoinha."))
    elif summary.get("problems_since_boot"):
        warnings.append("Pi ja limitou desde que ligou (" + ", ".join(summary["problems_since_boot"])
                        + "), mas nao nesta volta.")
    temp = summary.get("temp_max_c")
    if temp is not None and temp >= HOT_C:
        warnings.append(f"Pi a {temp:.0f} C (limita em ~80 C): ventilem antes de voltas longas.")
    low, high = summary.get("freq_min_mhz"), summary.get("freq_max_mhz")
    if low is not None and high and low < .9 * high and summary.get("governor") != "performance":
        warnings.append(f"Clock caiu a {low:.0f} de {high:.0f} MHz (governor {summary.get('governor')}):"
                        " 'performance' mantem o maximo.")
    return warnings


def _tasks(pid):
    """Threads do processo (todas precisam da afinidade, nao so a que chamou)."""
    try:
        return [int(task) for task in os.listdir(f"/proc/{pid}/task")]
    except (OSError, ValueError):
        return [pid]


def pin_process(cores, pid=None):
    """Fixa o processo `pid` (padrao: este) e suas threads nos nucleos `cores`.

    Devolve a lista aplicada, ou None se o sistema nao permite ou os nucleos nao
    existem. Threads criadas depois herdam a afinidade."""
    if not hasattr(os, "sched_setaffinity") or not cores:
        return None
    pid = os.getpid() if pid is None else int(pid)
    try:
        available = os.sched_getaffinity(pid)
    except OSError:
        return None
    wanted = {int(core) for core in cores} & set(available)
    if not wanted:
        return None
    applied = False
    for task in _tasks(pid):
        try:
            os.sched_setaffinity(task, wanted)
            applied = True
        except OSError:
            pass  # thread que terminou no meio
    return sorted(wanted) if applied else None


def core_layout(config, count=None):
    """(nucleos do controle, nucleos da visao) pela configuracao, ou (None, None).

    So com 4+ nucleos (Pi 4): com menos, fixar so tiraria opcoes do escalonador."""
    layout = config.get("cpu_cores", True)
    count = os.cpu_count() if count is None else count
    if not layout or not count or count < 4:
        return None, None
    if layout is True:
        layout = {}
    return list(layout.get("control", (0, 1))), list(layout.get("vision", (2, 3)))
