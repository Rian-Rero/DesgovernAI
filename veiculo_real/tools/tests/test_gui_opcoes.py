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
