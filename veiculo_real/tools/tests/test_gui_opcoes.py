"""Opcoes da missao da fva_gui_um_carro: cada combinacao vira um comando que o main.py aceita."""
import importlib.util
import itertools
from pathlib import Path
import shlex
import sys
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
        self.assertEqual(gui.build_main_command({}),
                         "python3 -u main.py --navigation cones --vision-width 0 --steering-filter 3"
                         " --speed 0.7 --stream 8080")

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
                    self.assertEqual(settings.get("steering_law", "pure_pursuit") == "pure_pursuit",
                                     options["pure_pursuit"])
                    self.assertEqual(settings.get("speed_preview_m", 2.5) > 0, options["preview"])
                self.assertEqual(settings.get("cone_debug", True), options["cone_debug"])
                self.assertEqual(settings.get("record_frames_hz", 0) > 0, options["record_frames"])
                self.assertEqual("cone_classifier" in settings, options["classifier"])
                self.assertEqual(settings.get("ultrasonic_curve_filter", True), options["curve_echo"])
                self.assertEqual(settings.get("steering_trim_learning", True), options["trim_learning"])

    def test_values_and_line_mode(self):
        settings = parsed(gui.build_main_command({"speed": "2.0", "vision_width": "320", "steering_filter": "1",
                                                  "record_hz": "5", "record_frames": True, "min_radius": "2.0"}))
        self.assertEqual((settings["velocity_reference"], settings["vision_width"],
                          settings["steering_filter_size"], settings["record_frames_hz"],
                          settings["min_curve_radius_m"]), (2., 320, 1, 5., 2.0))
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
            "python3 -u main.py --navigation cones --speed 0.7 --stream 8080": "missao_cones_0.7ms_purepursuit",
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
