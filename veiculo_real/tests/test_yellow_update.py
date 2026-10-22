"""Regressoes da integracao Stanley com a faixa pontilhada de fix:yellow.

Imagens sinteticas e carro falso; nenhum sensor ou atuador e inicializado.
"""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

REAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL))
SPEC = importlib.util.spec_from_file_location(
    'real_yellow_update_under_test', REAL / 'fva_car/lateral.py')
lateral = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lateral
SPEC.loader.exec_module(lateral)


def dashed_scene(curved=False, sparse=False):
    """Mesmos tracos da regressao original em tools/tests/test_yellow_path.py."""
    frame = np.full((480, 640, 3), 35, np.uint8)
    intervals = (((180, 204), (294, 318), (408, 432)) if sparse
                 else ((155, 205), (275, 325), (395, 479)))
    for start, end in intervals:
        rows = np.arange(start, end + 1)
        columns = 320 + (.0012 * (rows - 460) ** 2 if curved else np.zeros(len(rows)))
        cv2.polylines(frame, [np.c_[columns, rows].astype(np.int32)],
                      False, (0, 220, 240), 16)
    return frame


def calibration():
    # Homografia sintetica afim, em metros no eixo dianteiro.
    # x=.005*(480-v), y=.003*(320-u): frente positiva e esquerda positiva.
    matrix = np.array([[0., -.005, 2.4], [-.003, 0., .96], [0., 0., 1.]])
    return lateral.GroundCalibration((640, 480), matrix)


def fake_car():
    car = SimpleNamespace(v=.4, dt=.02, t=1., vref=0., u=0., st=0.,
                          emergencia=False, gear='forward', vel_error_integral=3.)
    car._set_ref = Mock(side_effect=lambda value: setattr(car, 'vref', value))
    car.set_vel = Mock(side_effect=lambda value: setattr(car, 'vref', value))
    car.set_u = Mock(side_effect=lambda value: setattr(car, 'u', value))
    car.set_steer = Mock(side_effect=lambda value: setattr(car, 'st', value))
    car.log = Mock()
    return car


class RealDashedPathTests(unittest.TestCase):
    def detector(self, **overrides):
        settings = {'clahe': False, 'smoothing': 0}
        settings.update(overrides)
        return lateral.RealLineDetector(calibration(), path_settings=settings)

    def test_sparse_dashes_accepted_by_backend_reach_stanley(self):
        frame = dashed_scene(sparse=True)
        settings = lateral.PathSettings(clahe=False, smoothing=0)
        backend = lateral.YellowPathDetector().process(frame, settings, timestamp=10.)
        self.assertTrue(backend.valid)
        self.assertGreater(backend.support, .28)
        self.assertLess(backend.support, .5)
        observation = self.detector().observe_frame(frame, 10.)
        self.assertTrue(observation.valid, observation.reason)
        np.testing.assert_allclose(observation.errors[:2], [0., 0.], atol=.01)
        car = fake_car()
        self.assertTrue(lateral.RealStanleyFollower({}).apply(car, observation, 10.))
        car.set_vel.assert_called_once()
        self.assertGreater(car.vref, 0.)

    def test_sparse_curved_dashes_preserve_the_metric_curve(self):
        observation = self.detector().observe_frame(dashed_scene(curved=True, sparse=True), 10.)
        self.assertTrue(observation.valid, observation.reason)
        cross, heading, curvature = observation.errors
        self.assertLess(abs(cross), .03)
        self.assertAlmostEqual(heading, math.atan(.0288), delta=.09)
        self.assertLess(curvature, -.1)
        self.assertTrue(lateral.RealStanleyFollower({}).apply(fake_car(), observation, 10.))

    def test_disabled_dashes_or_short_gap_do_not_authorize_motion(self):
        for settings in ({'dashed': False}, {'dash_gap': 40}):
            with self.subTest(settings=settings):
                observation = self.detector(**settings).observe_frame(dashed_scene(), 10.)
                self.assertFalse(observation.valid)
                car = fake_car()
                self.assertFalse(lateral.RealStanleyFollower({}).apply(car, observation, 10.))
                car.set_vel.assert_not_called()
                car.set_u.assert_called_with(0.)
                self.assertEqual(car.vel_error_integral, 0.)

    def test_empty_frame_after_dashes_is_invalid_and_cuts_throttle(self):
        detector = self.detector()
        first = detector.observe_frame(dashed_scene(sparse=True), 10.)
        self.assertTrue(first.valid, first.reason)
        empty = np.full((480, 640, 3), 35, np.uint8)
        lost = detector.observe_frame(empty, 10.05)
        self.assertFalse(lost.valid)
        car = fake_car()
        self.assertFalse(lateral.RealStanleyFollower({}).apply(car, lost, 10.05))
        car.set_vel.assert_not_called()
        car.set_u.assert_called_with(0.)
        car.set_steer.assert_called_with(0.)

    def test_color_support_and_reconstructed_fraction_are_logged_separately(self):
        detector = self.detector()
        frame = dashed_scene(sparse=True)
        backend = lateral.YellowPathDetector().process(frame, detector.settings, timestamp=10.)
        observation = detector.observe_frame(frame, 10.)
        expected_fraction = cv2.countNonZero(backend.dash_mask) / cv2.countNonZero(backend.path_mask)
        self.assertAlmostEqual(observation.color_support, backend.support)
        self.assertAlmostEqual(observation.dash_fraction, expected_fraction)
        self.assertLess(observation.color_support, .5)
        self.assertGreater(observation.dash_fraction, .5)
        car = fake_car()
        lateral.RealStanleyFollower({}).apply(car, observation, 10.)
        values = car.log.call_args.kwargs
        self.assertAlmostEqual(values['line_color_support'], observation.color_support)
        self.assertAlmostEqual(values['line_dash_fraction'], observation.dash_fraction)
        stale_car = fake_car()
        self.assertFalse(lateral.RealStanleyFollower({}).apply(stale_car, observation, 10.5))
        stale_values = stale_car.log.call_args.kwargs
        self.assertTrue(math.isnan(stale_values['line_color_support']))
        self.assertTrue(math.isnan(stale_values['line_dash_fraction']))


class DashedSettingsValidationTests(unittest.TestCase):
    def test_dashed_requires_a_boolean(self):
        for value in (1, 0, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lateral.RealLineDetector(calibration(), path_settings={'dashed': value})
        for value in (True, False):
            lateral.RealLineDetector(calibration(), path_settings={'dashed': value})

    def test_dash_gap_requires_an_integer_within_supported_pixel_range(self):
        for value in (-1, 481, 12.5, float('nan'), float('inf'), '120', True, False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lateral.RealLineDetector(calibration(), path_settings={'dash_gap': value})
        for value in (0, 40, 120, 480):
            lateral.RealLineDetector(calibration(), path_settings={'dash_gap': value})


if __name__ == '__main__':
    unittest.main()
