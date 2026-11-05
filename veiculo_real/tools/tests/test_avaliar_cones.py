"""Rotulagem e avaliacao do detector de cones em um minidataset sintetico."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliar_cones import (RESULT_FILE, WORST_DIR, check_calibration, evaluate, main as avaliar_main,
                           match_points, polyline_distance)
from cone_path import ConePathDetector, ConeSettings
from rotular_cones import (DEFAULT_CALIBRATION, LabelSession, click, load_labels, prelabel,
                           remove_near, save_labels)
from test_cone_path import track


MISSED = [600, 120]  # cone rotulado que o detector nao ve (fundo uniforme)


def make_dataset(folder):
    """3 quadros; rotulos = bases do detector, menos uma verde (vira falso
    positivo) e mais uma vermelha inexistente na imagem (cone perdido)."""
    frames = [track(), track("curve"), track(shift=-20)]
    (folder / "frames").mkdir()
    labels, counts = {}, {"green": 0, "red": 0}
    for index, frame in enumerate(frames):
        name = f"f{index:06d}.jpg"
        cv2.imwrite(str(folder / "frames" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        stored = cv2.imread(str(folder / "frames" / name))
        entry = prelabel(stored, ConePathDetector(), ConeSettings(tracking=False))
        counts["green"] += len(entry["green"])
        counts["red"] += len(entry["red"])
        entry["reviewed"] = True
        labels[name] = entry
    removed = labels["f000000.jpg"]["green"].pop(0)
    labels["f000000.jpg"]["red"].append(list(MISSED))
    save_labels(folder, labels)
    return counts, removed


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.counts, self.removed = make_dataset(self.folder)

    def tearDown(self):
        self.tmp.cleanup()

    def test_precision_and_recall_count_false_positive_and_missed_cone(self):
        report = evaluate(self.folder, settings=ConeSettings(tracking=False))
        green, red = report["detection"]["green"], report["detection"]["red"]
        g, r = self.counts["green"], self.counts["red"]
        self.assertGreaterEqual(g, 10)
        self.assertGreaterEqual(r, 10)
        self.assertEqual((green["tp"], green["fp"], green["fn"]), (g - 1, 1, 0))
        self.assertAlmostEqual(green["precision"], round((g - 1) / g, 4))
        self.assertEqual(green["recall"], 1.0)
        self.assertEqual((red["tp"], red["fp"], red["fn"]), (r, 0, 1))
        self.assertEqual(red["precision"], 1.0)
        self.assertAlmostEqual(red["recall"], round(r / (r + 1), 4))
        self.assertAlmostEqual(report["false_positives_per_frame"], round(1 / 3, 4))
        frame0 = report["frames"]["f000000.jpg"]
        self.assertEqual(frame0["green"]["fp"], 1)
        self.assertEqual(frame0["red"]["fn"], 1)
        # O cone perdido fica na faixa mais distante (y=120 em 80..160).
        band = next(b for b in report["recall_by_row"] if b["y_min"] <= MISSED[1] < b["y_max"])
        self.assertEqual(band["labels"] - band["matched"], 1)

    def test_tracking_cli_writes_report_and_worst_frames(self):
        with redirect_stdout(io.StringIO()) as out:
            code = avaliar_main([str(self.folder), "--save-worst", "2", "--calibration", ""])
        self.assertEqual(code, 0)
        self.assertIn("precisao", out.getvalue())
        report = json.loads((self.folder / RESULT_FILE).read_text())
        for key in ("detection", "false_positives_per_frame", "recall_by_row", "frames_valid",
                    "valid_fraction", "tremor_m", "tremor_px", "wrong_links", "wrong_link_frames",
                    "frames", "config", "worst_frames"):
            self.assertIn(key, report)
        self.assertTrue(report["config"]["tracking"])
        self.assertEqual(report["frames_total"], 3)
        self.assertEqual(report["frames_reviewed"], 3)
        self.assertEqual(set(report["detection"]), {"green", "red", "total"})
        self.assertEqual(len(report["recall_by_row"]), 6)
        self.assertEqual(report["frames_valid"], 3)  # pista sintetica limpa
        self.assertEqual(report["valid_fraction"], 1.0)
        self.assertIsNotNone(report["tremor_px"])
        # O falso positivo e uma base real da borda: o corredor passa por ele.
        self.assertEqual(report["wrong_link_frames"], ["f000000.jpg"])
        self.assertEqual(report["worst_frames"], ["01_f000000.jpg"])
        self.assertTrue((self.folder / WORST_DIR / "01_f000000.jpg").exists())

    def test_unreviewed_frames_are_ignored(self):
        labels = load_labels(self.folder)
        labels["f000000.jpg"]["reviewed"] = False
        save_labels(self.folder, labels)
        report = evaluate(self.folder, settings=ConeSettings(tracking=False))
        self.assertEqual(report["frames_reviewed"], 2)
        self.assertEqual(report["detection"]["total"]["fp"], 0)
        self.assertEqual(report["detection"]["total"]["fn"], 0)

    def test_check_calibration_reports_error_and_short_range(self):
        out = io.StringIO()
        report = check_calibration(DEFAULT_CALIBRATION, out)
        data = json.loads(Path(DEFAULT_CALIBRATION).read_text())
        self.assertEqual(len(report["points"]), len(data["image_points_px"]))
        xs = [p["ground_m"][0] for p in report["points"]]
        self.assertEqual(xs, sorted(xs))
        self.assertLess(report["max_error_m"], .05)
        if max(p[0] for p in data["ground_points_m"]) < 1.5:
            self.assertIn("AVISO", out.getvalue())
            self.assertIsNotNone(report["warning"])


class MatchingTests(unittest.TestCase):
    def test_greedy_matching_prefers_closest_pair(self):
        pairs, fp, fn = match_points([[0, 0], [10, 0]], [[9, 0], [30, 0]], 12)
        self.assertEqual(pairs, [(1, 0)])
        self.assertEqual(fp, [0])
        self.assertEqual(fn, [1])

    def test_polyline_distance_uses_segments(self):
        distance = polyline_distance([[5, 3], [20, 0]], [[0, 0], [10, 0]])
        np.testing.assert_allclose(distance, [3, 10])


class LabelingTests(unittest.TestCase):
    def test_click_adds_and_removes_within_radius(self):
        entry = {"green": [], "red": [], "reviewed": False}
        self.assertEqual(click(entry, "green", 100, 200), "adicionado")
        self.assertEqual(click(entry, "red", 300, 200), "adicionado")
        self.assertEqual(entry, {"green": [[100, 200]], "red": [[300, 200]], "reviewed": False})
        self.assertFalse(remove_near(entry, 100, 209))
        self.assertEqual(click(entry, "red", 105, 205), "removido")  # remove o verde proximo
        self.assertEqual(entry["green"], [])
        self.assertEqual(entry["red"], [[300, 200]])

    def test_session_prelabels_saves_and_preserves_existing_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "frames").mkdir()
            for index in range(2):
                cv2.imwrite(str(folder / "frames" / f"f{index:06d}.jpg"), track())
            save_labels(folder, {"f000000.jpg": {"green": [[1, 2]], "red": [], "reviewed": True},
                                 "outro.jpg": {"green": [], "red": [[5, 6]], "reviewed": True}})
            session = LabelSession(folder, ConePathDetector())
            self.assertEqual(session.name, "f000001.jpg")  # primeiro nao revisado
            entry = session.entry()
            self.assertGreaterEqual(len(entry["green"]), 4)  # pre-rotulo do detector
            self.assertFalse(entry["reviewed"])
            session.click("red", 630, 20)
            session.previous()
            self.assertEqual(session.entry()["green"], [[1, 2]])  # rotulo existente intacto
            session.save()
            saved = json.loads((folder / "labels.json").read_text())
            self.assertEqual(set(saved), {"f000000.jpg", "f000001.jpg", "outro.jpg"})
            self.assertTrue(saved["f000001.jpg"]["reviewed"])
            self.assertIn([630, 20], saved["f000001.jpg"]["red"])
            self.assertEqual(saved["outro.jpg"], {"green": [], "red": [[5, 6]], "reviewed": True})
            self.assertEqual(load_labels(folder), saved)


if __name__ == "__main__":
    unittest.main()
