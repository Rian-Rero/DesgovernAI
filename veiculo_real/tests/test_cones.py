"""Percepcao de cones e integracao, sem acionar hardware."""
import math
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import RealConeDetector, corridor_center, PriorityNavigationDetector, ConeObservation
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
                self.assertAlmostEqual(observation.errors[0], .08, delta=.015)
                self.assertAlmostEqual(observation.errors[1], 0., delta=.03)
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

    def test_width_overlap_and_outlier_validation(self):
        x = np.array([.3, .8, 1.3, 1.8])
        left = np.column_stack((x, np.ones(4)))
        right = np.column_stack((x+.1, -np.ones(4)))
        result = corridor_center(left, right)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[2], 2.)
        self.assertIsNone(corridor_center(left, right+np.array([3., 0.])))
        self.assertIsNone(corridor_center(left, right, max_width=1.))
        left[1, 1] += 1.
        self.assertIsNone(corridor_center(left, right))

    def test_invalid_settings_fail_before_hardware(self):
        calibration, _ = scene()
        for settings in ({'red_hsv_ranges': [[[180, 0, 0], [255, 255, 255]]]},
                         {'min_width_m': -1}, {'min_width_m': 2, 'max_width_m': 1}):
            with self.assertRaises(ValueError):
                RealConeDetector(calibration, settings)

    def test_priority_and_fallback_use_same_frame(self):
        cones, line = Mock(), Mock()
        detector = PriorityNavigationDetector(cones, line)
        frame = np.zeros((10, 10, 3), np.uint8)
        valid_cones = ConeObservation(10., errors=(0., 0., 0.))
        cones.observe_frame.return_value = valid_cones
        self.assertIs(detector.observe_frame(frame, 10.), valid_cones)
        line.observe_frame.assert_not_called()
        cones.observe_frame.return_value = ConeObservation(11., reason='sem cones')
        line.observe_frame.return_value = LineObservation(11., errors=(.1, 0., 0.))
        fallback = detector.observe_frame(frame, 11.)
        self.assertTrue(fallback.valid)
        self.assertEqual(fallback.reference_source, 'line')
        self.assertIs(line.observe_frame.call_args.args[0], frame)
        self.assertEqual(line.observe_frame.call_args.args[1], 11.)
        detector.with_preview(valid_cones)
        cones.with_preview.assert_called_with(valid_cones)
        detector.preview_bgr(fallback)
        line.preview_bgr.assert_called_with(fallback)
        line.observe_frame.return_value = LineObservation(12., reason='sem linha')
        invalid = detector.observe_frame(frame, 12.)
        self.assertFalse(invalid.valid)
        self.assertIn('Sem referencia', invalid.reason)

    def test_cli_selects_cones(self):
        with patch.object(sys, 'argv', ['main.py', '--navigation', 'cones']), patch.object(mission, 'run_mission') as run:
            mission.main()
        self.assertEqual(run.call_args.args[0]['navigation_mode'], 'cones')


if __name__ == '__main__':
    unittest.main()
