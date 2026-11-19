"""Matriz de confusao contra os rotulos (matriz_confusao.py) e vista de cima (vista_de_cima.py)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
import matriz_confusao as mc  # noqa: E402
import rotular_sim_nao as labeling  # noqa: E402
import vista_de_cima as bev  # noqa: E402
from cone_path import ConeSettings  # noqa: E402
from rotular_cones import load_calibration  # noqa: E402
from test_cone_navigation import H, metric_scene  # noqa: E402
from fva_car.lateral import GroundCalibration  # noqa: E402


class ConfusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.calibration = base/'camera.json'
        self.calibration.write_text(json.dumps({'image_size': [640, 480], 'homography': H.tolist()}))
        run = base/'ensaios'/'20261117_204300'
        (run/'frames').mkdir(parents=True)
        for k in range(3):
            cv2.imwrite(str(run/'frames'/f'f{k:06d}.jpg'), metric_scene(lateral=.05*k))
        self.out = base/'rotulagem'
        labeling.prepare([base/'ensaios'], self.out, self.calibration, step=1, report=lambda m: None)

    def tearDown(self):
        self.temp.cleanup()

    def test_humans_agreeing_with_the_detector_give_a_perfect_matrix(self):
        items, _ = labeling.load(self.out)
        labeling.save_labels(self.out, {item['id']: bool(item['palpite']) for item in items})
        frames, calibrations = mc.ground_truth(self.out, with_calibration=True)
        self.assertTrue(all(calibrations.values()))  # cada item guarda a sua calibracao
        counts = mc.confusion(frames, load_calibration(self.calibration), ConeSettings(tracking=False), 640,
                              calibrations=calibrations)
        total = counts['total']
        self.assertEqual(total.get('fn', 0) + total.get('fp', 0), 0)
        self.assertGreater(total['vp'], 5)
        m = mc.metrics(total)
        self.assertEqual((m['precisao'], m['revocacao']), (1., 1.))

    def test_an_accepted_non_cone_is_a_false_positive_and_the_figure_is_written(self):
        items, _ = labeling.load(self.out)
        labels = {item['id']: bool(item['palpite']) for item in items}
        flipped = next(item for item in items if item['palpite'])  # humano: "nao e cone"
        labels[flipped['id']] = False
        labeling.save_labels(self.out, labels)
        frames = mc.ground_truth(self.out)
        counts = mc.confusion(frames, load_calibration(self.calibration), ConeSettings(tracking=False), 640)
        self.assertEqual(counts['total'].get('fp', 0), 1)
        self.assertLess(mc.metrics(counts['total'])['precisao'], 1.)
        figure = mc.plot({'teste': counts}, Path(self.temp.name)/'m.png')
        self.assertTrue(figure.exists())


class BirdsEyeTests(unittest.TestCase):
    def test_ground_point_lands_where_the_overlay_draws_it(self):
        calibration = GroundCalibration((640, 480), H)
        frame = np.zeros((480, 640, 3), np.uint8)
        pixel = cv2.perspectiveTransform(np.array([[[2., .5]]]), np.linalg.inv(H))[0, 0]
        cv2.circle(frame, tuple(int(round(v)) for v in pixel), 4, (255, 255, 255), -1)
        view = bev.warp(frame, calibration)
        u, v = bev.to_view([[2., .5]])[0]
        self.assertGreater(int(view[int(round(v)), int(round(u))].max()), 200)  # o mesmo lugar em metros

    def test_panel_with_a_debug_record(self):
        record = {'frame': 7, 'control': {'v': .5, 'vref': .6, 'steer_cmd_rad': .1},
                  'vision': {'detection': {'green': {'m': [[1.5, .6]]}, 'red': {'m': [[1.5, -.6]]}},
                             'corridor': {'state': 'AMBAS', 'errors': [.05, .02, 0.], 'dkappa': 0.,
                                          'width_m': 1.2, 'sides': {'green_left': True}}}}
        panel = bev.render(metric_scene(), GroundCalibration((640, 480), H), record)
        width, height = bev.view_size()
        self.assertEqual(panel.shape[0], height)
        self.assertGreater(panel.shape[1], width)


if __name__ == '__main__':
    unittest.main()
