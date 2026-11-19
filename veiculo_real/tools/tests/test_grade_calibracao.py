"""Grade sugerida e marcadores da calibracao (tools/grade_calibracao.py)."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import calibrar_camera_lateral as calibrate  # noqa: E402
import grade_calibracao as grid  # noqa: E402
from fva_car.lateral import GroundCalibration  # noqa: E402

CALIBRATION = GroundCalibration.from_file(ROOT / 'calibracao' / 'camera_lateral.json')


class SuggestedGridTests(unittest.TestCase):
    def setUp(self):
        self.points = grid.suggest_grid(CALIBRATION.homography, CALIBRATION.image_size)
        self.pixels = grid.predict_pixels(CALIBRATION.homography, self.points)

    def test_only_visible_points_in_click_order(self):
        self.assertGreaterEqual(len(self.points), 15)
        width, height = CALIBRATION.image_size
        self.assertTrue(np.all((self.pixels[:, 0] >= 0) & (self.pixels[:, 0] < width)
                               & (self.pixels[:, 1] >= 0) & (self.pixels[:, 1] < height)))
        xs = [x for x, _ in self.points]
        self.assertEqual(xs, sorted(xs))  # perto -> longe
        for x in set(xs):  # na mesma distancia: esquerda -> direita na imagem
            row = self.pixels[[k for k, p in enumerate(self.points) if p[0] == x], 0]
            self.assertTrue(np.all(np.diff(row) > 0))

    def test_covers_where_the_cones_are_used(self):
        xs = np.array([x for x, _ in self.points])
        ys = np.array([y for _, y in self.points])
        self.assertGreaterEqual(xs.max(), 2.9)
        self.assertGreaterEqual(np.abs(ys).max(), .9)
        self.assertLessEqual(xs.min(), 1.)
        far_rows = {x for x in xs if x >= 1.5}
        for x in far_rows:  # a linha do centro em toda distancia longe
            self.assertIn((x, 0.), self.points)

    def test_pdf_and_saved_grid_feed_the_click_window(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path, pages = grid.markers_pdf(self.points, folder / 'm.pdf', hidden=[(.6, 1.)])
            near = sum(1 for x, _ in self.points if x < 2.)
            far = len(self.points) - near
            self.assertEqual(pages, 1 + -(-near // 4) + -(-far // 2))
            self.assertTrue(path.read_bytes().startswith(b'%PDF'))
            saved = grid.save_grid(self.points, folder / 'grade.json')
            self.assertEqual(grid.load_grid(saved), self.points)
            args = argparse.Namespace(output=ROOT / 'calibracao' / 'camera_lateral.json', grade_fixa=False,
                                      grade_x=None, grade_y=None)
            with patch.object(grid, 'GRID_FILE', saved):  # a grade do PDF manda na ordem dos cliques
                targets, hints = calibrate.grid_targets(args, CALIBRATION.image_size)
            self.assertEqual(targets, self.points)
            self.assertEqual(len(hints), len(targets))
            fixed, _ = calibrate.grid_targets(argparse.Namespace(**{**vars(args), 'grade_fixa': True}),
                                              CALIBRATION.image_size)
            self.assertEqual(len(fixed), 30)

    def test_labels(self):
        self.assertEqual(grid.coordinates_text(2., .5), 'x = 2,00 m   y = 0,50 m ESQ')
        self.assertEqual(grid.coordinates_text(1., 0.), 'x = 1,00 m   y = centro')
        self.assertIn('DIR', grid.coordinates_text(1., -.25))


if __name__ == '__main__':
    unittest.main()


class CameraDiagnosisTests(unittest.TestCase):
    """camera_geometry / diagnose: camera girada e zero lateral deslocado."""
    @staticmethod
    def synthetic(lateral=.1, yaw_right_deg=8., height=.15, pitch_down_deg=4.):
        import math
        f = 320/math.tan(math.radians(30.))
        K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1.]])
        yaw, pitch = math.radians(-yaw_right_deg), math.radians(pitch_down_deg)
        forward = np.array([math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), -math.sin(pitch)])
        right = np.cross(forward, [0, 0, 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.vstack([right, down, forward])  # chao (x frente, y esq, z cima) -> camera
        center = np.array([0., lateral, height])
        ground = np.array([(x, y) for x in (.8, 1., 1.5, 2., 2.5) for y in (-.5, -.25, 0., .25, .5)])
        cam = np.c_[ground, np.zeros(len(ground))] - center
        q = (K @ (R @ cam.T)).T
        return q[:, :2]/q[:, 2:3], ground

    def test_lateral_position_and_yaw_are_recovered(self):
        pixels, ground = self.synthetic()
        geometry = calibrate.camera_geometry(calibrate.fit_homography(pixels, ground))
        self.assertAlmostEqual(geometry['lateral_m'], .1, delta=.01)
        self.assertAlmostEqual(geometry['yaw_deg'], 8., delta=.3)  # + = virada para a direita
        self.assertAlmostEqual(geometry['roll_deg'], 0., delta=.3)

    def test_offset_origin_and_shifted_row_are_reported(self):
        pixels, ground = self.synthetic(lateral=.18)
        _, warnings = calibrate.diagnose(pixels, ground)
        self.assertTrue(any('18 cm a esquerda' in w for w in warnings), warnings)
        pixels, ground = self.synthetic(lateral=0.)
        shifted = ground.copy()
        shifted[np.isclose(ground[:, 0], 1.5), 1] += .19
        _, warnings = calibrate.diagnose(pixels, shifted)
        self.assertTrue(any('fileira de 1.50 m' in w for w in warnings), warnings)
