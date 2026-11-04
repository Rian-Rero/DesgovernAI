"""Percepcao de cones e integracao, sem acionar hardware."""
import math
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import RealConeDetector, ConeObservation
from fva_car.lateral import GroundCalibration, RealStanleyFollower, LineObservation
from test_lateral import fake_car
from test_mission import mission


def scene(bend=0., offset=0., reverse=False):
    calibration = GroundCalibration((640, 480), [[0, -.005, 2.4], [-.005, 0, 1.6], [0, 0, 1]])
    frame = np.full((480, 640, 3), 35, np.uint8)
    inverse = np.linalg.inv(calibration.homography)
    for sign in (-1, 1):
        for x in (.4, .8, 1.2, 1.6):
            y = offset+bend*x*x+sign*.5
            pixel = cv2.perspectiveTransform(np.array([[[x, y]]]), inverse)[0, 0]
            u, v = np.rint(pixel).astype(int)
            red = (sign == 1) != reverse
            color = (0, 0, 255) if red else (0, 220, 0)
            cv2.fillPoly(frame, [np.array([[u, v-20], [u-7, v], [u+7, v]])], color)
    return calibration, frame


class ConeTests(unittest.TestCase):
    def test_camera_metric_center_both_directions_and_curves(self):
        for bend in (0., -.15, .15):
            for reverse in (False, True):
                calibration, frame = scene(bend=bend, offset=.08, reverse=reverse)
                detector = RealConeDetector(calibration)
                observation = detector.observe_frame(frame, 10.)
                self.assertTrue(observation.valid, observation.reason)
                self.assertAlmostEqual(observation.errors[0], .08, delta=.02)
                self.assertAlmostEqual(observation.errors[1], 0., delta=.04)
                self.assertAlmostEqual(observation.errors[2], 2*bend, delta=.04)
                self.assertEqual(len(observation.red_pixels), 4)
                self.assertEqual(len(observation.green_pixels), 4)
                self.assertIsNone(observation.frame)
                self.assertEqual(detector.with_preview(observation).frame.shape, frame.shape)

    def test_reduced_processing_preserves_coordinates(self):
        calibration, frame = scene(bend=.1)
        full = RealConeDetector(calibration).observe_frame(frame, 10.)
        reduced = RealConeDetector(calibration, processing_width=320).observe_frame(frame, 10.)
        self.assertTrue(reduced.valid, reduced.reason)
        np.testing.assert_allclose(reduced.errors, full.errors, atol=.035)

    def test_one_border_and_camera_fault_do_not_authorize_motion(self):
        calibration, frame = scene()
        frame[:, :320] = 35  # elimina uma borda
        detector = RealConeDetector(calibration)
        for image in (frame, None, np.zeros((240, 320, 3), np.uint8)):
            observation = detector.observe_frame(image, 10.)
            self.assertFalse(observation.valid)
            car = fake_car()
            follower = RealStanleyFollower({'line_loss_frames': 0, 'line_loss_tolerance': 0})
            self.assertFalse(follower.apply(car, observation, 10.))
            self.assertEqual(car.vref, 0.)
            self.assertEqual(car.st, 0.)

    def test_adapter_uses_shared_hsv_detector_and_metric_geometry(self):
        from tools.cone_path import ConePathDetector, ConeSettings
        from fva_car.lateral import fit_line_errors
        calibration, frame = scene(bend=.1, reverse=True)
        detector = RealConeDetector(calibration)
        observation = detector.observe_frame(frame, 10.)
        expected = ConePathDetector(calibration).process(frame, ConeSettings(roi_top=0), timestamp=10.)
        self.assertTrue(expected.valid, expected.status)
        np.testing.assert_array_equal(observation.cone_result.green_mask, expected.green_mask)
        np.testing.assert_allclose(observation.cone_result.ground_center, expected.ground_center)
        np.testing.assert_allclose(observation.errors,
            fit_line_errors(expected.ground_center, [0, 0], [1, 0], with_curvature=True))
        self.assertEqual(detector.settings.roi_top, 0)
        self.assertEqual(mission.parameters['path_settings']['roi_top'], 0)

    def test_invalid_settings_fail_before_hardware(self):
        calibration, _ = scene()
        for settings in ({'red_hsv_ranges': [[[180, 0, 0], [255, 255, 255]]]},
                         {'roi_top': -1}, {'min_width_cm': 200, 'max_width_cm': 100}):
            with self.assertRaises(ValueError):
                RealConeDetector(calibration, settings)

    def test_cli_selects_cones(self):
        with patch.object(sys, 'argv', ['main.py', '--navigation', 'cones']), patch.object(mission, 'run_mission') as run:
            mission.main()
        self.assertEqual(run.call_args.args[0]['navigation_mode'], 'cones')


if __name__ == '__main__':
    unittest.main()
