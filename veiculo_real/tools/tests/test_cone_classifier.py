"""Classificador de candidatos: recorte, regra de aceitacao, integracao e ONNX real."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from cone_classifier import CLASSES, ConeClassifier, LOOSE_COLOR, crop_window, extract_crop  # noqa: E402
from cone_path import ConePathDetector, ConeSettings  # noqa: E402
from extrair_recortes_cones import extract_folder  # noqa: E402
from test_cone_navigation import calibration, metric_scene  # noqa: E402

MODEL = Path(__file__).resolve().parent / 'dados' / 'classificador_teste.onnx'
META = {'classes': list(CLASSES), 'crop': {'size': 32}, 'candidate_settings': {'green_chroma_min': 5},
        'min_confidence': .6}


class FakeNet:
    """Decide pela cor media do recorte: verde -> cone_verde, vermelho -> cone_vermelho."""
    def __init__(self, reject=()):
        self.reject = set(reject)
        self.calls = 0

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        self.calls += 1
        b, g, r = self.blob.mean(axis=(2, 3)).T
        logits = np.zeros((len(b), 3))
        logits[:, 1] = 8*(g - np.maximum(r, b))
        logits[:, 2] = 8*(r - np.maximum(g, b))
        for name in self.reject:
            logits[:, CLASSES.index(name)] = -10
        return logits


class CropTests(unittest.TestCase):
    def test_crop_is_aligned_by_the_base_and_scaled_by_expected_height(self):
        left, top, side = crop_window((100, 200), px_per_m=100., cone_height_m=.2)
        self.assertAlmostEqual(side, 32.)  # 1,6 x 20 cm x 100 px/m
        self.assertAlmostEqual(left + side/2, 100.)
        self.assertAlmostEqual(top + side, 200 + .15*side)  # base um pouco acima do fundo

    def test_crop_outside_the_image_repeats_the_border(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[:, :10] = (0, 255, 0)
        crop = extract_crop(frame, (2, 470), px_per_m=200., cone_height_m=.2)
        self.assertEqual(crop.shape, (32, 32, 3))
        self.assertGreater(crop[..., 1].mean(), 100)  # borda verde repetida a esquerda


class ClassifierTests(unittest.TestCase):
    def test_keeps_only_candidates_confirmed_with_the_same_color(self):
        frame = np.full((480, 640, 3), 120, np.uint8)
        cv2.rectangle(frame, (90, 160), (110, 200), (0, 220, 0), -1)     # verde de verdade
        cv2.rectangle(frame, (290, 160), (310, 200), (0, 0, 220), -1)    # vermelho na mascara verde
        cv2.rectangle(frame, (490, 160), (510, 200), (0, 0, 220), -1)    # vermelho de verdade
        scale = np.full(480, 100.)
        scale[:100] = np.nan
        classifier = ConeClassifier(net=FakeNet(), meta=META)
        verdict = classifier.classify(frame, {'green': [[100, 200], [300, 200], [400, 50]],
                                              'red': [[500, 200]]}, scale, .2)
        np.testing.assert_array_equal(verdict['green'][0], [True, False, True])  # sem escala: mantem
        np.testing.assert_array_equal(verdict['red'][0], [True])
        self.assertTrue(np.isnan(verdict['green'][1][2]))

    def test_detector_drops_rejected_candidates_and_reports_them(self):
        frame = metric_scene()
        plain = ConePathDetector(calibration()).process(frame, ConeSettings(tracking=False), timestamp=1.)
        classifier = ConeClassifier(net=FakeNet(reject=('cone_verde',)), meta=META)
        result = ConePathDetector(calibration(), classifier=classifier).process(
            frame, ConeSettings(tracking=False), timestamp=1.)
        self.assertGreater(len(plain.green_points), 0)
        self.assertEqual(len(result.green_points), 0)
        self.assertEqual(len(result.red_points), len(plain.red_points))
        detection = result.debug['detection']
        self.assertEqual(detection['green']['rejected']['classificador'], len(plain.green_points))
        self.assertEqual(detection['classifier']['kept'], len(plain.red_points))

    def test_real_cone_detector_uses_model_color_settings_unless_overridden(self):
        from fva_car.cones import RealConeDetector
        from fva_car.lateral import GroundCalibration
        cal = GroundCalibration((640, 480), [[0, -.005, 2.4], [-.005, 0, 1.6], [0, 0, 1]])
        classifier = ConeClassifier(net=FakeNet(), meta=META)
        detector = RealConeDetector(cal, classifier=classifier)
        self.assertEqual(detector.settings.green_chroma_min, 5)
        self.assertIs(detector.detector.classifier, classifier)
        explicit = RealConeDetector(cal, settings={'green_chroma_min': 9}, classifier=classifier)
        self.assertEqual(explicit.settings.green_chroma_min, 9)
        self.assertEqual(RealConeDetector(cal).settings.green_chroma_min, ConeSettings().green_chroma_min)

    def test_missing_model_files_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            ConeClassifier('/nao/existe/modelo.onnx')


class OnnxModelTests(unittest.TestCase):
    def test_onnx_model_loads_with_opencv_and_batches_consistently(self):
        classifier = ConeClassifier(MODEL)
        rng = np.random.default_rng(1)
        crops = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
        batch = classifier.predict(crops)
        self.assertEqual(batch.shape, (5, 3))
        np.testing.assert_allclose(batch.sum(axis=1), 1., atol=1e-6)
        np.testing.assert_allclose(classifier.predict(crops[:1])[0], batch[0], atol=1e-5)
        self.assertEqual(classifier.candidate_settings, LOOSE_COLOR)


class ExtractionTests(unittest.TestCase):
    def test_labels_become_positives_and_unlabeled_candidates_negatives(self):
        folder = Path(tempfile.mkdtemp())
        (folder / 'frames').mkdir()
        frame = metric_scene()
        cv2.imwrite(str(folder / 'frames' / 'f000001.jpg'), frame)
        found = ConePathDetector(calibration()).candidates(frame, ConeSettings(tracking=False, **LOOSE_COLOR))
        green = found['green'].round().tolist()
        # Rotula so os verdes: os vermelhos detectados viram negativos.
        (folder / 'labels.json').write_text(json.dumps(
            {'f000001.jpg': {'green': green, 'red': [], 'reviewed': True},
             'f000002.jpg': {'green': [[1, 1]], 'red': [], 'reviewed': False}}))
        X, y, group, counts = extract_folder(folder, calibration())
        self.assertEqual(counts['cone_verde'], len(green))
        self.assertEqual(counts['nao_cone'], len(found['red']))
        self.assertEqual(X.shape[1:], (32, 32, 3))
        self.assertEqual(set(group.tolist()), {0})
        self.assertEqual(int(np.sum(y == CLASSES.index('cone_vermelho'))), 0)


if __name__ == '__main__':
    unittest.main()
