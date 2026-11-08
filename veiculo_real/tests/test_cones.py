"""Percepcao de cones e integracao, sem acionar hardware."""
import math
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import CONE_RECOVERY_SPEED, RealConeDetector, ConeObservation
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

    def test_distant_center_uses_local_direction_instead_of_extrapolated_curve(self):
        from fva_car.lateral import fit_line_errors
        x = np.linspace(1.0, 1.8, 17)
        points = np.column_stack((x, .05 + .2*(x-1.4)**2))
        lateral, heading, curvature = fit_line_errors(
            points, [0, 0], [1, 0], with_curvature=True, limit_curve_extrapolation=True)
        self.assertAlmostEqual(lateral, .05, delta=.02)
        self.assertAlmostEqual(heading, 0.)
        self.assertEqual(curvature, 0.)

    def test_far_straight_center_starts_slowly_but_far_curve_does_not(self):
        # Ensaio parado (logs 20261023_080354): o primeiro portao ficava a 1,9-2,3 m.
        from tools.cone_path import ConePathResult
        calibration, frame = scene()
        detector = RealConeDetector(calibration, settings={'single_border_recovery': True})
        x = np.linspace(2.0, 2.7, 15)
        mask = np.zeros((480, 640), np.uint8)
        for curvature, expected in ((.05, True), (.6, False)):
            result = ConePathResult(mask, mask, mask, center=np.array([[320, 300]], np.int32),
                                    source='solo', ground_center=np.column_stack((x, .03 + 0*x)),
                                    curvature_m_inv=curvature, status='Centro pelos portoes de cones (m)')
            with patch.object(detector.detector, 'process', return_value=result):
                observation = detector.observe_frame(frame, 10.)
            self.assertEqual(observation.valid, expected, observation.reason)
            if expected:
                self.assertAlmostEqual(observation.errors[0], .03, delta=.01)
                self.assertEqual(observation.speed_limit, CONE_RECOVERY_SPEED)

    def test_gate_midpoints_skip_diagonal_pairs_at_end_of_border(self):
        from tools.cone_track_path import gate_midpoints
        left = [[1.0, .5], [1.5, .5]]
        right = [[.4, -.5], [1.0, -.5], [1.5, -.5]]
        midpoints, widths = gate_midpoints(left, right)
        np.testing.assert_allclose(midpoints, [[1., 0.], [1.5, 0.]])
        np.testing.assert_allclose(widths, [1., 1.])

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

    def test_recovery_steers_away_from_near_border_regardless_of_color(self):
        for side in (-1, 1):
            for reverse in (False, True):
                with self.subTest(side=side, reverse=reverse):
                    calibration, frame = scene(offset=-side*.25, reverse=reverse)
                    if side > 0:
                        frame[:, 320:] = 35
                    else:
                        frame[:, :320] = 35
                    detector = RealConeDetector(calibration, mission.parameters['cone_settings'])
                    observation = detector.observe_frame(frame, 10.)
                    self.assertTrue(observation.valid, observation.reason)
                    self.assertTrue(observation.debug['recovery'])
                    self.assertLess(side*observation.errors[0], 0.)
                    car = fake_car()
                    follower = RealStanleyFollower({'velocity_reference': .7})
                    self.assertTrue(follower.apply(car, observation, 10.))
                    self.assertLess(side*car.st, 0.)
                    self.assertLessEqual(car.vref, CONE_RECOVERY_SPEED)
                    self.assertTrue(follower.apply(car, LineObservation(10.05), 10.05))
                    self.assertLessEqual(car.vref, CONE_RECOVERY_SPEED)
                    self.assertFalse(follower.apply(car, LineObservation(11.1), 11.1))

    def test_recovery_prefers_the_measured_center_with_both_borders(self):
        calibration, frame = scene(offset=.08)
        detector = RealConeDetector(calibration, mission.parameters['cone_settings'])
        observation = detector.observe_frame(frame, 10.)
        self.assertTrue(observation.valid, observation.reason)
        self.assertFalse(observation.debug['recovery'])
        self.assertAlmostEqual(observation.errors[0], .08, delta=.02)
        self.assertTrue(math.isinf(observation.speed_limit))

    def test_recovery_requires_persistence_for_an_isolated_cone(self):
        calibration, frame = scene(offset=-.25)
        frame[:, 320:] = 35
        frame[:290] = 35
        frame[335:] = 35
        detector = RealConeDetector(calibration, mission.parameters['cone_settings'])
        self.assertFalse(detector.observe_frame(frame, 10.).valid)
        observation = detector.observe_frame(frame, 10.1)
        self.assertTrue(observation.valid, observation.reason)
        self.assertTrue(observation.debug['gates']['recovery_single_cone'])
        self.assertLess(observation.errors[0], 0.)
        self.assertEqual(observation.errors[2], 0.)

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
            fit_line_errors(expected.ground_center, [0, 0], [1, 0], with_curvature=True,
                            limit_curve_extrapolation=True))
        self.assertEqual(detector.settings.roi_top, 0)
        self.assertEqual(mission.parameters['cone_settings']['roi_top'], 0)

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


    def test_invalid_frame_is_processed_once(self):
        from tools.cone_path import ConePathDetector
        calibration, frame = scene()
        frame[:, :320] = 35
        detector = RealConeDetector(calibration)
        with patch.object(ConePathDetector, 'process', autospec=True,
                          side_effect=ConePathDetector.process) as process:
            observation = detector.observe_frame(frame, 10.)
        self.assertFalse(observation.valid)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(observation.debug['valid'], False)

    def test_swapped_colors_valid_on_first_frame_with_debug(self):
        calibration, frame = scene()  # vermelho a esquerda (+y)
        detector = RealConeDetector(calibration, {'green_left': True})
        observation = detector.observe_frame(frame, 10.)
        self.assertTrue(observation.valid, observation.reason)
        self.assertFalse(detector.green_left)
        self.assertFalse(observation.cone_settings.green_left)
        self.assertEqual(len(observation.debug['stanley_errors']), 3)
        self.assertIn('trajectory', observation.debug)

    def test_cone_loss_keeps_trajectory_with_speed_ramp_then_stops(self):
        settings = {'line_loss_frames': 4, 'line_loss_tolerance': 1., 'loss_speed_ramp': True,
                    'line_search_rate_deg_s': 0., 'curve_min_speed': .15, 'velocity_reference': .7,
                    'speed_filter_s': 0}
        car = fake_car()
        follower = RealStanleyFollower(settings)
        valid = lambda t: LineObservation(timestamp=t, errors=(.05, 0., 0.))
        for step in range(10):
            follower.apply(car, valid(10.+.01*step), 10.+.01*step)
        cruise = car.vref
        self.assertGreater(cruise, .5)
        speeds = []
        for frame in range(1, 5):
            now = 10.1+.05*frame
            self.assertTrue(follower.apply(car, LineObservation(timestamp=now, reason='sem cones'), now))
            self.assertTrue(follower.last_state['hold'])
            speeds.append(car.vref)
        self.assertTrue(all(a > b for a, b in zip([cruise]+speeds, speeds)), speeds)
        self.assertAlmostEqual(speeds[-1], .15, delta=1e-6)
        self.assertIn('vref x', follower.status)
        self.assertEqual(car.st, follower.last_state['steer_rad'])
        # Esgotada a tolerancia: neutro e vref zerada.
        self.assertFalse(follower.apply(car, LineObservation(timestamp=10.4, reason='sem cones'), 10.4))
        self.assertEqual(car.vref, 0.)
        self.assertFalse(follower.last_state['active'])

    def test_line_mode_keeps_full_speed_during_hold(self):
        car = fake_car()
        follower = RealStanleyFollower({'line_loss_frames': 4, 'speed_filter_s': 0, 'velocity_reference': .7})
        for step in range(10):
            follower.apply(car, LineObservation(timestamp=10.+.01*step, errors=(.05, 0., 0.)), 10.+.01*step)
        cruise = car.vref
        follower.apply(car, LineObservation(timestamp=10.2, reason='sem faixa'), 10.2)
        self.assertAlmostEqual(car.vref, cruise, delta=.02)


if __name__ == '__main__':
    unittest.main()
