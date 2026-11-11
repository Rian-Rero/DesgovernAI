"""Linha amarela com a camera alta: vista de cima + estimador (fva_car/lateral.py)."""
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fva_car.corridor import sample_centerline  # noqa: E402
from fva_car.lateral import BirdsEyeView, GroundCalibration, RealLineDetector, line_samples  # noqa: E402
from fva_car.odometry import PoseHistory  # noqa: E402
import main  # noqa: E402

CALIBRATION = GroundCalibration.from_file(ROOT/'calibracao'/'camera_lateral.json')


def floor_with_line(state, width=.05):
    """Piso cinza com uma fita amarela de `width` m ao longo de `state` (clotoide)."""
    image = np.full((480, 640, 3), (150, 156, 158), np.uint8)
    inverse = np.linalg.inv(np.asarray(CALIBRATION.homography, float))
    for s in np.arange(-.3, 5., .012):
        quad = np.vstack([sample_centerline(state, [s, s+.014], width/2),
                          sample_centerline(state, [s+.014, s], -width/2)])
        if np.any(quad[:, 0] < .3):
            continue
        q = np.c_[quad, np.ones(4)] @ inverse.T
        pixels = q[:, :2]/q[:, 2:3]
        if np.all(np.abs(pixels) < 4000):
            cv2.fillConvexPoly(image, pixels.round().astype(np.int32), (30, 200, 235))
    return image


class BirdsEyeTests(unittest.TestCase):
    def test_view_pixels_map_back_to_the_same_ground_point(self):
        view = BirdsEyeView(CALIBRATION)
        ground = np.array([[1., 0.], [2., .5], [.8, -.3]])
        q = np.c_[ground, np.ones(3)] @ view.ground_to_view.T
        pixels = q[:, :2]/q[:, 2:3]
        np.testing.assert_allclose(view.to_ground(pixels), ground, atol=1e-9)
        self.assertEqual(view.size, (280, 250))
        # Linha reta a frente: coluna do meio da vista (y = 0).
        self.assertAlmostEqual(pixels[0, 0], view.size[0]/2, places=6)

    def test_line_samples_are_spaced_along_the_line(self):
        points = np.c_[np.linspace(.2, 3.5, 300), np.zeros(300)]
        samples = line_samples(points)
        self.assertTrue(np.all(np.diff(samples[:, 0]) >= .15 - 1e-9))
        self.assertTrue(np.all((samples[:, 0] > .3) & (samples[:, 0] <= 3.)))


class HighCameraLineTests(unittest.TestCase):
    def detector(self):
        detector = RealLineDetector(CALIBRATION, path_settings=main.parameters['path_settings'],
                                    render_preview=False, birdseye=True,
                                    estimator=main.line_estimator_settings(main.parameters))
        detector.odometry = PoseHistory()
        return detector

    def test_offset_and_heading_on_a_straight(self):
        for offset, heading in ((0., 0.), (.15, 0.), (-.2, 0.), (0., .15)):
            with self.subTest(offset=offset, heading=heading):
                detector = self.detector()
                image = floor_with_line(np.array([offset, heading, 0., 0., 1.]))
                for k in range(4):
                    detector.odometry.append(.1*k, -.36, 0., 0.)
                    observation = detector.observe_frame(image, .1*k)
                self.assertTrue(observation.valid, observation.reason)
                self.assertAlmostEqual(observation.errors[0], offset, delta=.03)
                self.assertAlmostEqual(observation.errors[1], heading, delta=.05)
                self.assertTrue(math.isfinite(observation.path_dkappa))  # pure pursuit e previa usam

    def test_tight_curve_is_detected_in_the_top_view(self):
        # R = 1,5 m a direita: na imagem da camera alta a linha deita e o
        # detector antigo (coluna por linha da imagem) nao a encontrava.
        image = floor_with_line(np.array([0., -.1, -1/1.5, 0., 1.]))
        classic = RealLineDetector(CALIBRATION, path_settings=main.parameters['path_settings'], render_preview=False)
        self.assertFalse(classic.observe_frame(image, 1.).valid)
        detector = self.detector()
        for k in range(4):
            detector.odometry.append(.1*k, -.36, 0., 0.)
            observation = detector.observe_frame(image, .1*k)
        self.assertTrue(observation.valid, observation.reason)
        self.assertLess(observation.errors[2], -.2)  # curva a direita

    def test_mission_uses_the_estimator_path_for_the_line(self):
        config = dict(main.parameters)
        derived = main.follower_config(config, 'line', main.line_estimator_settings(config))
        self.assertEqual(derived['steering_law'], config['steering_law'])
        self.assertGreater(derived['speed_sight_curvature'], 0.)
        classic = main.follower_config(dict(config, line_estimator=False), 'line',
                                       main.line_estimator_settings(dict(config, line_estimator=False)))
        self.assertEqual(classic['steering_law'], 'stanley')


class VirtualLineTests(unittest.TestCase):
    def test_closed_loop_line_following(self):
        from tools import pista_virtual as pv
        track = pv.build_track([('reta', 3.), ('curva', -1.5, 90), ('reta', 3.)], 1.3, rng=np.random.default_rng(2))
        result = pv.run(track, pv.base_config({'steering_law': 'pure_pursuit'}),
                        corridor=main.line_estimator_settings(main.parameters), seed=2, line=True)
        self.assertTrue(result['finished'])
        self.assertLess(result['rms_m'], .06)


if __name__ == '__main__':
    unittest.main()
