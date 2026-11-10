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
        keys = ("stream", "estimator", "pure_pursuit", "preview", "cone_debug", "record_frames", "classifier")
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

    def test_values_and_line_mode(self):
        settings = parsed(gui.build_main_command({"speed": "1.5", "vision_width": "320", "steering_filter": "1",
                                                  "record_hz": "5", "record_frames": True}))
        self.assertEqual((settings["velocity_reference"], settings["vision_width"],
                          settings["steering_filter_size"], settings["record_frames_hz"]), (1.5, 320, 1, 5.))
        line = gui.build_main_command({"navigation": "line", "pure_pursuit": False, "record_frames": True,
                                       "classifier": True, "estimator": False})
        settings = parsed(line)
        self.assertEqual(settings["navigation_mode"], "line")
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

    def test_log_keeps_text_and_sends_telemetry_to_csv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            log = gui.RunLog("vermelho", "missao_cones_0.7ms_purepursuit", {"comando": "python3 main.py"}, folder)
            log.write("\x1b[32mEncoder: leitura OK.\x1b[0m")
            log.write("DATA,0.1,0,0,1,1,0,0,0,0")
            log.write("DATA,0.2,0,0,1,1,0,0,0,0")
            log.close("codigo de saida 0")
            again = gui.RunLog("vermelho", "missao_cones_0.7ms_purepursuit", {}, folder)
            again.close()
            self.assertRegex(log.path.name, r"^\d\dh\d\dm\d\d_vermelho_missao_cones_0\.7ms_purepursuit\.txt$")
            self.assertNotEqual(log.path, again.path)  # mesmo segundo: nao sobrescreve
            text = log.path.read_text(encoding="utf-8")
            self.assertIn("# comando: python3 main.py", text)
            self.assertIn("Encoder: leitura OK.\n", text)  # sem codigos de cor
            self.assertNotIn("DATA,", text)
            self.assertIn("codigo de saida 0", text)
            self.assertEqual(log.telemetry_path.read_text().count("DATA,"), 2)
