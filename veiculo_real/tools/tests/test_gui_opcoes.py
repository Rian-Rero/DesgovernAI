"""Opcoes da missao da fva_gui_um_carro: cada combinacao vira um comando que o main.py aceita."""
import importlib.util
import itertools
from pathlib import Path
import shlex
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fva_gui_opcoes", ROOT / "tools" / "fva_gui_um_carro.py")
gui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gui)
sys.path.insert(0, str(ROOT))
import main as mission  # noqa: E402


def parsed(command):
    """Configuracao que o main.py montaria com este comando (sem rodar a missao)."""
    argv = shlex.split(command)
    assert argv[:3] == ["python3", "-u", "main.py"], argv
    with patch.object(sys, "argv", argv[2:]), patch.object(mission, "run_mission") as run:
        mission.main()
    return run.call_args[0][0]


class MissionOptionsTests(unittest.TestCase):
    def test_default_command_is_unchanged(self):
        # A rede treinada em 19/11 vem ligada por padrao.
        self.assertEqual(gui.build_main_command({}),
                         "python3 -u main.py --navigation cones --vision-width 320 --steering-filter 3"
                         " --speed 0.7 --tempo 180 --stream 8080 --classifier modelos/cones.onnx")

    def test_map_and_speed_cap_options_reach_main(self):
        self.assertIn("4.0", gui.SPEED_PRESETS)
        command = gui.build_main_command({"cone_map": False, "speed_cap": False, "speed": "4.0"})
        self.assertIn("--sem-mapa", command)
        self.assertIn("--sem-teto-ultrassom", command)
        settings = parsed(command)
        self.assertIs(settings["cone_map"], False)
        self.assertIs(settings["ultrasonic_speed_cap"], False)
        self.assertEqual(settings["velocity_reference"], 4.0)
        self.assertIn("sem-mapa", gui.command_label(command))
        # Sem o estimador o mapa nao existe: nem pede --sem-mapa.
        self.assertNotIn("--sem-mapa", gui.build_main_command({"cone_map": False, "estimator": False}))

    def test_every_checkbox_combination_is_accepted_by_main(self):
        keys = ("stream", "estimator", "pure_pursuit", "preview", "cone_debug", "record_frames", "classifier",
                "curve_echo", "trim_learning")
        for values in itertools.product((False, True), repeat=len(keys)):
            options = dict(zip(keys, values))
            with self.subTest(**options):
                settings = parsed(gui.build_main_command(options))
                self.assertEqual(settings["navigation_mode"], "cones")
                self.assertEqual("stream_port" in settings, options["stream"])
                self.assertEqual(settings.get("cone_estimator", True), options["estimator"])
                if options["estimator"]:
                    self.assertEqual(settings.get("steering_law", "stanley") == "pure_pursuit",
                                     options["pure_pursuit"])
                    self.assertEqual(settings.get("speed_preview_m", 2.5) > 0, options["preview"])
                self.assertEqual(settings.get("cone_debug", True), options["cone_debug"])
                self.assertEqual(settings.get("record_frames_hz", 0) > 0, options["record_frames"])
                self.assertEqual("cone_classifier" in settings, options["classifier"])
                self.assertEqual(settings.get("ultrasonic_curve_filter", True), options["curve_echo"])
                self.assertEqual(settings.get("steering_trim_learning", True), options["trim_learning"])

    def test_values_and_line_mode(self):
        settings = parsed(gui.build_main_command({"speed": "2.0", "vision_width": "480", "steering_filter": "1",
                                                  "record_hz": "5", "record_frames": True, "min_radius": "2.0"}))
        self.assertEqual((settings["velocity_reference"], settings["vision_width"],
                          settings["steering_filter_size"], settings["record_frames_hz"],
                          settings["min_curve_radius_m"]), (2., 480, 1, 5., 2.0))
        self.assertNotIn("min_curve_radius_m", parsed(gui.build_main_command({})))  # 1.2: padrao do main.py
        line = gui.build_main_command({"navigation": "line", "pure_pursuit": False, "record_frames": True,
                                       "classifier": True, "estimator": False})
        settings = parsed(line)
        self.assertEqual(settings["navigation_mode"], "line")
        self.assertNotIn("line_birdseye", settings)  # padrao: camera alta
        classic = parsed(gui.build_main_command({"navigation": "line", "line_highcam": False}))
        self.assertEqual((classic["line_birdseye"], classic["line_estimator"]), (False, False))
        self.assertNotIn("--line-classic", gui.build_main_command({"line_highcam": False}))  # so na linha
        self.assertFalse({"steering_law", "record_frames_hz", "cone_classifier", "cone_estimator"} & set(settings))

    def test_comparison_switches_and_throttle_reach_main(self):
        """Rodada de testes: refino e partida nova liga/desliga, acelerador minimo."""
        self.assertNotIn("cone_refine_bases", parsed(gui.build_main_command({})))
        self.assertFalse(parsed(gui.build_main_command({"refine": False}))["cone_refine_bases"])
        self.assertNotIn("--no-refine", gui.build_main_command({"navigation": "line", "refine": False}))
        self.assertNotIn("throttle_learning", parsed(gui.build_main_command({})))  # auto: aprende no carro
        fixed = parsed(gui.build_main_command({"throttle_offset": "0.28"}))
        self.assertEqual((fixed["throttle_ff_offset"], fixed["throttle_learning"]), (.28, False))
        self.assertEqual(parsed(gui.build_main_command({"throttle_offset": "0,36"}))["throttle_ff_offset"], .36)
        for typing in ("auto", "", ".", "abc", "0.9"):  # auto, digitando ou fora da faixa: aprende
            self.assertNotIn("throttle_ff_offset", parsed(gui.build_main_command({"throttle_offset": typing})))
        old = parsed(gui.build_main_command({"new_start": False, "throttle_offset": "0.28"}))
        self.assertTrue(old["old_start"])
        self.assertNotIn("throttle_ff_offset", old)  # a partida antiga traz o acelerador antigo
        self.assertEqual(mission.corridor_settings(old)["slow_sigma_e"], .1)
        self.assertNotIn("slow_sigma_e", mission.corridor_settings({}))
        label = gui.command_label(gui.build_main_command({"refine": False, "new_start": False}))
        self.assertTrue(label.endswith("_sem-refino_partida-antiga"), label)
        self.assertTrue(gui.command_label(gui.build_main_command({"throttle_offset": "0.28"})).endswith("_acel0.28"))

    def test_stanley_gain_reaches_main(self):
        self.assertNotIn("stanley_gain", parsed(gui.build_main_command({})))  # 2.5: padrao do main.py
        self.assertEqual(parsed(gui.build_main_command({"stanley_gain": "3,0"}))["stanley_gain"], 3.)
        self.assertNotIn("--stanley-gain", gui.build_main_command({"stanley_gain": "3", "pure_pursuit": True}))
        self.assertNotIn("--stanley-gain", gui.build_main_command({"stanley_gain": "abc"}))

    def test_steering_scales_and_curve_opening_reach_main(self):
        self.assertFalse({"steering_left_scale", "steering_right_scale", "racing_line_m"}
                         & set(parsed(gui.build_main_command({}))))
        settings = parsed(gui.build_main_command({"steer_left": "90", "steer_right": "105", "racing_line": "0.20",
                                                  "track_width": "1,3"}))
        self.assertEqual((settings["steering_left_scale"], settings["steering_right_scale"], settings["racing_line_m"]),
                         (.9, 1.05, .2))
        self.assertEqual(settings["track_width_m"], 1.3)
        self.assertEqual(settings["cone_settings"]["track_width_cm"], 130)
        self.assertNotIn("--largura-pista", gui.build_main_command({"track_width": ""}))
        self.assertNotIn("--esterco-esquerda", gui.build_main_command({"steer_left": "abc"}))

    def test_only_main_lines_are_rewritten(self):
        self.assertEqual(gui.with_main_command("ls\npython3 -u main.py --speed 0.7\necho fim", "python3 -u main.py X"),
                         "ls\npython3 -u main.py X\necho fim")
        self.assertEqual(gui.with_main_command("ls", "python3 -u main.py X"), "python3 -u main.py X\nls")
        self.assertEqual(gui.with_main_command("", "python3 -u main.py X"), "python3 -u main.py X")


if __name__ == "__main__":
    unittest.main()


class TerminalLogTests(unittest.TestCase):
    def test_file_names_say_what_ran(self):
        cases = {
            "python3 -u main.py --navigation cones --speed 0.7 --stream 8080": "missao_cones_0.7ms_stanley",
            "python3 -u main.py --navigation cones --speed 0.7 --pure-pursuit": "missao_cones_0.7ms_purepursuit",
            "python3 -u main.py --navigation cones --speed 1.5 --stanley --no-preview":
                "missao_cones_1.5ms_stanley_sem-previa",
            "python3 -u main.py --speed 1.0 --no-estimator --record-frames 3 --classifier modelos/cones.onnx":
                "missao_cones_1.0ms_sem-estimador_gravando_rede",
            "python3 -u main.py --navigation line --speed 0.7": "missao_linha_0.7ms",
            "python3 -m unittest discover -s tests": "testes_unitarios",
            "python3 tools/benchmark_visao.py --cones": "benchmark_visao",
            "python3 tools/ajustar_camera.py": "ajustar_camera",
            "ls -la": "ls",
        }
        for command, label in cases.items():
            with self.subTest(command=command):
                self.assertEqual(gui.command_label(command), label)

    def test_remote_log_goes_inside_logs_folder_collected_by_the_gui(self):
        import time
        now = time.strptime("2026-11-10 19:05:12", "%Y-%m-%d %H:%M:%S")
        folder, path = gui.remote_log_path("/home/alunos/Desktop/fva", "testes_unitarios", now)
        self.assertEqual(folder, "/home/alunos/Desktop/fva/logs/terminal/2026-11-10")
        self.assertEqual(path, folder + "/19h05m12_testes_unitarios.log")
        for name in ("fva_gui_0a1b2c.log", "fva_teste_ff00.log", "fva_camera_http_12ab.log"):
            self.assertTrue(gui.LEGACY_TMP_LOGS.match(name), name)
        for name in ("fva_gui_0a1b2c.exit", "outro.log"):
            self.assertFalse(gui.LEGACY_TMP_LOGS.match(name), name)


class LocalCar:
    """Faz o papel do SSH/SFTP do carro: roda o comando e le arquivos no proprio notebook."""
    def exec_command(self, command, timeout=None):
        import io
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return None, io.StringIO(result.stdout), io.BytesIO(result.stderr.encode())

    def open_sftp(self):
        return self

    def open(self, path, mode="r"):
        return open(path, mode)

    def listdir_attr(self, path):
        import os
        entries = []
        for entry in os.scandir(path):
            info = entry.stat()
            attr = type("A", (), {})()
            attr.filename, attr.st_mode, attr.st_size, attr.st_mtime = entry.name, info.st_mode, info.st_size, info.st_mtime
            entries.append(attr)
        return entries

    def get(self, remote, local):
        import shutil
        shutil.copyfile(remote, local)

    def put(self, local, remote):
        import shutil
        self.puts = getattr(self, "puts", 0) + 1
        shutil.copyfile(local, remote)

    def utime(self, path, times):
        import os
        os.utime(path, times)

    def mkdir(self, path):
        import os
        os.mkdir(path)

    def stat(self, path):
        import os
        return os.stat(path)

    def remove(self, path):
        import os
        os.remove(path)

    def get_transport(self):
        return type("T", (), {"is_active": lambda self: True})()

    def get_channel(self):
        return type("C", (), {"settimeout": lambda self, t: None})()

    def close(self):
        pass


class IncrementalSendTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.folders = [tempfile.TemporaryDirectory() for _ in range(2)]
        self.local, self.car = (Path(f.name) for f in self.folders)
        for rel, text in {"main.py": "print(1)\n", "fva_car/corridor.py": "x = 1\n",
                          "tools/tests/test_a.py": "pass\n", "calibracao/camera.json": "{}",
                          "fva_car/__pycache__/corridor.cpython-312.pyc": "lixo",
                          "logs/20261023/car.csv": "dado local", ".DS_Store": "x"}.items():
            path = self.local / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (self.car / "logs").mkdir()
        (self.car / "logs" / "car.csv").write_text("dado do carro", encoding="utf-8")

    def tearDown(self):
        for folder in self.folders:
            folder.cleanup()

    def sync(self, **options):
        car = LocalCar()
        lines = []
        summary = gui.sync_directory(car, car, str(self.local), str(self.car), lines.append, **options)
        return summary, lines, getattr(car, "puts", 0)

    def test_only_changed_files_are_sent(self):
        import os
        summary, _, puts = self.sync()
        self.assertEqual((summary["sent"], puts), (4, 4))  # sem __pycache__, .pyc, logs/, .DS_Store
        self.assertFalse((self.car / "fva_car" / "__pycache__").exists())
        self.assertEqual((self.car / "logs" / "car.csv").read_text(encoding="utf-8"), "dado do carro")
        summary, _, puts = self.sync()
        self.assertEqual((summary["sent"], summary["same"], puts), (0, 4, 0))
        edited = self.local / "fva_car" / "corridor.py"
        edited.write_text("x = 2\n", encoding="utf-8")  # mesmo tamanho, conteudo novo
        later = os.stat(edited).st_mtime + 30  # edicao real: segundos depois do envio
        os.utime(edited, (later, later))
        summary, lines, puts = self.sync()
        self.assertEqual((summary["sent"], puts), (1, 1))
        self.assertIn("   ↑ fva_car/corridor.py", lines)
        self.assertEqual((self.car / "fva_car" / "corridor.py").read_text(encoding="utf-8"), "x = 2\n")
        # So a data mudou (git checkout, copia): confere o md5 e nao reenvia.
        os.utime(self.local / "main.py", (1e9, 1e9))
        summary, _, puts = self.sync()
        self.assertEqual((summary["sent"], puts), (0, 0))
        self.assertEqual(int(os.stat(self.car / "main.py").st_mtime), int(1e9))
        summary, _, puts = self.sync(force=True)
        self.assertEqual(puts, 4)

    def test_data_folders_and_files_never_go_to_the_car(self):
        data = {
            "experimentos/vermelho/20261023_091339/car.csv": "t,v",            # Coletar Dados dentro da pasta
            "experimentos/vermelho/20261023_091339/cone_frames/f1.jpg": "jpg",
            "logs_terminal/2026-11-10/19h05m12_testes.log": "saida",
            "Logs Antigos/x.txt": "pasta de log com outro nome",
            "teste_curva_lara/cones_debug.jsonl": "{}",                          # ensaio com nome qualquer
            "teste_curva_lara/notas.py": "print('script de analise do ensaio')",
            "tools/2026-10-23_08h45m36/auto.png": "foto do ajuste",             # pasta com nome de data
            "Archive.zip": "zip", "cones_ds.npz": "npz", "tools/video.mp4": "mp4",
        }
        for rel, text in data.items():
            path = self.local / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        skipped = {}
        sent = set(gui.local_tree(str(self.local), skipped))
        self.assertEqual(sent, {"main.py", "fva_car/corridor.py", "tools/tests/test_a.py", "calibracao/camera.json"})
        self.assertEqual(set(skipped["pastas"]), {"experimentos", "logs", "logs_terminal", "Logs Antigos",
                                                  "teste_curva_lara", "tools/2026-10-23_08h45m36"})
        self.assertGreaterEqual(skipped["arquivos"], len(data))
        summary, lines, puts = self.sync()
        self.assertEqual(puts, 4)
        self.assertTrue(any("fora do envio" in line for line in lines), lines)

    def test_first_time_cleanup_wipes_the_car_folder_but_keeps_logs(self):
        # Carro "pesado": experimentos e lixo enviados antes, mais os logs das missoes.
        for rel in ("experimentos/vermelho/20261023/car.csv", "tools/experimentos/x.jpg", "velho.py",
                    "logs/20261110_093000/car.csv", "main.py"):
            path = self.car / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("antigo", encoding="utf-8")
        car = LocalCar()
        removed = dict(gui.clean_remote(car, str(self.car), check=False))
        self.assertEqual(set(removed), {"experimentos", "tools", "velho.py", "main.py"})
        self.assertEqual(sorted(p.name for p in self.car.iterdir()), ["logs"])
        self.assertTrue((self.car / "logs" / "20261110_093000" / "car.csv").exists())
        summary, _, puts = self.sync(force=True)
        self.assertEqual(puts, 4)  # so o codigo volta
        self.assertFalse((self.car / "experimentos").exists())

    def test_cleanup_refuses_folders_that_are_not_a_project(self):
        for path in ("/", "/home", "/home/alunos", "/home/alunos/Desktop", "/tmp/x/y", "/etc/fva/x", "fva"):
            with self.subTest(path=path):
                self.assertFalse(gui.clean_target_is_safe(path))
        self.assertTrue(gui.clean_target_is_safe("/home/alunos/Desktop/fva"))
        with self.assertRaises(RuntimeError):
            gui.clean_remote(LocalCar(), "/home/alunos")

    def test_optional_cleanup_removes_only_stale_python_files(self):
        self.sync()
        for rel in ("tests/test_adas.py", "logs/script.py", "notas.txt"):
            path = self.car / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("antigo", encoding="utf-8")
        summary, _, _ = self.sync()
        self.assertTrue((self.car / "tests" / "test_adas.py").exists())  # padrao: nao apaga nada
        summary, lines, _ = self.sync(delete_py=True)
        self.assertEqual(summary["removed"], 1)
        self.assertFalse((self.car / "tests" / "test_adas.py").exists())
        self.assertTrue((self.car / "logs" / "script.py").exists())
        self.assertTrue((self.car / "notas.txt").exists())


@unittest.skipUnless(__import__("os").environ.get("FVA_TEST_TK") == "1", "Set FVA_TEST_TK=1 for real Tk testing")
class CollectTerminalLogsTests(unittest.TestCase):
    def test_command_output_is_saved_in_logs_and_comes_with_collect_data(self):
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as car_dir, tempfile.TemporaryDirectory() as notebook:
            Path(car_dir, "main.py").write_text(
                "print('Encoder: leitura OK.')\nprint('DATA,0.1,0,0,1,1,0,0,0,0')\n", encoding="utf-8")
            with patch.object(gui.RsyncGUI, "refresh_ips"):
                app = gui.RsyncGUI()
            try:
                app.active_device = ("vermelho", "127.0.0.1")
                app._connect_ssh = lambda *args: LocalCar()
                app.dest_entry.delete(0, "end")
                app.dest_entry.insert(0, car_dir)
                app.data_dest_entry.delete(0, "end")
                app.data_dest_entry.insert(0, notebook)
                app.cmd_text.delete("1.0", "end")
                app.cmd_text.insert("end", "python3 -u main.py --navigation cones --speed 1.5 --stanley")
                started = time.monotonic()
                state = {"phase": "run"}

                def step():
                    text = app.cmd_log.get("1.0", "end")
                    if time.monotonic() - started > 20:
                        app.quit()
                    elif state["phase"] == "run" and "Execução encerrada" in text:
                        state["phase"] = "collect"
                        app.collect_data()
                    elif state["phase"] == "collect" and "Coleta finalizada" in app.data_log.get("1.0", "end"):
                        app.quit()
                    if state["phase"] != "done":
                        app.after(100, step)

                app.after(100, app.run_cmds_on_selected)
                app.after(200, step)
                app.mainloop()
                on_car = list(Path(car_dir, "logs", "terminal").rglob("*.log"))
                collected = [p for p in Path(notebook, "vermelho", "terminal").rglob("*.log")
                             if p.parent.name != "recuperados_tmp"]
            finally:
                app.destroy()
            self.assertEqual(len(on_car), 1)
            self.assertRegex(on_car[0].name, r"^\d\dh\d\dm\d\d_missao_cones_1\.5ms_stanley\.log$")
            self.assertIn("Encoder: leitura OK.", on_car[0].read_text(encoding="utf-8"))
            self.assertEqual([p.name for p in collected], [on_car[0].name])


class MissionControlTests(unittest.TestCase):
    """Botoes Encerrar / Liberar parada: o comando roda no carro (aqui, localmente)."""
    ROOT = Path(__file__).resolve().parents[2]

    def start(self, answers=True):
        import subprocess
        import tempfile
        self.folder = Path(tempfile.mkdtemp())
        self.files = {k: str(self.folder/f"missao.{k}") for k in ("pid", "cmd", "ack")}
        loop = (  # o laco de controle do main.py, so a parte do canal
            "import sys, time\n"
            "while True:\n"
            f"    request = mission.read_mission_command({self.files['cmd']!r})\n"
            "    if request:\n"
            "        action, code = request\n"
            f"        mission.answer_mission_command(code, action + ' recebido pelo carro', {self.files['ack']!r})\n"
            f"        open({str(self.folder/'feito')!r}, 'a').write(action + '\\n')\n"
            "        if action == 'encerrar':\n"
            "            sys.exit(0)\n"
            "    time.sleep(.05)\n"
        ) if answers else "import time\nwhile True:\n    time.sleep(.05)\n"
        (self.folder/"main.py").write_text(
            "import os, signal, sys\n"
            f"sys.path.insert(0, {str(self.ROOT)!r})\n"
            "import main as mission\n"
            f"open({self.files['pid']!r}, 'w').write(str(os.getpid()))\n"
            f"signal.signal(signal.SIGUSR1, lambda *_: open({str(self.folder/'sinal')!r}, 'w').write('1'))\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n" + loop)
        self.process = subprocess.Popen([sys.executable, str(self.folder/"main.py")])
        self.addCleanup(self.process.wait, 5)
        self.addCleanup(self.process.kill)
        deadline = time.monotonic() + 20
        while not Path(self.files["pid"]).exists() and time.monotonic() < deadline:
            time.sleep(.05)

    def run_remote(self, command):
        import subprocess
        return subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=20)

    def command(self, action, wait_s=3.):
        return gui.mission_command(action, self.files["pid"], self.files["cmd"], self.files["ack"], wait_s)

    def test_car_confirms_release_then_stop(self):
        self.start()
        result = self.run_remote(self.command("liberar"))
        self.assertEqual(result.returncode, gui.MISSION_OK, result.stderr)
        self.assertIn("liberar recebido pelo carro", result.stdout)
        self.assertIsNone(self.process.poll())       # liberar nao encerra
        result = self.run_remote(self.command("encerrar"))
        self.assertEqual(result.returncode, gui.MISSION_OK, result.stderr)
        self.assertEqual(int(result.stdout.split()[0]), self.process.pid)
        self.process.wait(5)
        self.assertEqual((self.folder/"feito").read_text().split(), ["liberar", "encerrar"])
        self.assertNotEqual(self.run_remote(gui.mission_alive_command(self.process.pid)).returncode, 0)

    def test_old_code_on_the_car_gets_the_signal(self):
        self.start(answers=False)
        result = self.run_remote(self.command("liberar", wait_s=.5))
        self.assertEqual(result.returncode, gui.MISSION_SIGNAL_ONLY, result.stderr)
        deadline = time.monotonic() + 5
        while not (self.folder/"sinal").exists() and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertTrue((self.folder/"sinal").exists())
        self.assertFalse(Path(self.files["cmd"]).exists())  # pedido sem resposta nao fica para depois

    def test_nothing_running(self):
        self.start()
        self.process.kill()
        self.process.wait(5)
        self.assertEqual(self.run_remote(self.command("encerrar")).returncode, gui.MISSION_NONE)
        self.assertEqual(self.run_remote(gui.mission_signal_command("liberar", str(self.folder/"nada.pid"))).returncode,
                         gui.MISSION_NONE)

    def test_mission_time_reaches_main(self):
        self.assertEqual(parsed(gui.build_main_command({}))["ts"], 180.)
        self.assertEqual(parsed(gui.build_main_command({"mission_time": "60"}))["ts"], 60.)
        self.assertNotIn("--tempo", gui.build_main_command({"mission_time": ""}))
