"""Resposta do filtro de direcao, com servos e threads substituidos."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fva_car.filter import MovingAverage


def load_servos():
    package = ModuleType('_steering_response')
    package.__path__ = [str(ROOT / 'fva_car')]
    driver = ModuleType('adafruit_servokit')
    driver.ServoKit = Mock(return_value=MagicMock())
    spec = importlib.util.spec_from_file_location(package.__name__ + '.servos', ROOT / 'fva_car/servos.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package.__name__: package, 'adafruit_servokit': driver}):
        spec.loader.exec_module(module)
    return module, driver.ServoKit


class SteeringResponseTests(unittest.TestCase):
    def test_short_filter_reaches_requested_angle_with_less_delay(self):
        module, kit = load_servos()
        with patch.object(module.threading, 'Thread'):
            short = module.Servos(steering_filter_size=3)
            previous = module.Servos(steering_filter_size=10)
        short_response = [short.set_steer(.1) for _ in range(3)]
        previous_response = [previous.set_steer(.1) for _ in range(3)]
        self.assertAlmostEqual(short_response[-1], .1)
        self.assertAlmostEqual(previous_response[-1], .03)
        self.assertGreater(short_response[0], previous_response[0])

    def test_one_sample_filter_preserves_saturation_and_has_no_average_delay(self):
        module, kit = load_servos()
        with patch.object(module.threading, 'Thread'):
            servo = module.Servos(steering_filter_size=1)
        self.assertAlmostEqual(servo.set_steer(1.), module.MAX_STERRING_ANGLE)
        self.assertAlmostEqual(servo.set_steer(-1.), -module.MAX_STERRING_ANGLE)

    def test_invalid_filter_size_is_rejected_before_i2c_is_opened(self):
        module, kit = load_servos()
        for size in (0, -1, True, 3., '3'):
            with self.subTest(size=size), self.assertRaises(ValueError):
                module.Servos(steering_filter_size=size)
        kit.assert_not_called()


class MovingAverageRegressionTests(unittest.TestCase):
    def test_running_sum_matches_window_mean_and_reset(self):
        rng = np.random.default_rng(42)
        for size in (1, 3, 5, 10, 30):
            filt = MovingAverage(size, initial=.2)
            history = [.2] * size
            for sample in rng.normal(size=1000):
                history = (history + [sample])[-size:]
                self.assertAlmostEqual(filt.filter(sample), np.mean(history), places=12)
            filt.reset(.7)
            self.assertAlmostEqual(filt.filter(.7), .7)

    def test_nonfinite_value_expires_instead_of_poisoning_running_sum(self):
        for sample in (float('nan'), float('inf')):
            filt = MovingAverage(3)
            self.assertFalse(np.isfinite(filt.filter(sample)))
            for _ in range(3):
                value = filt.filter(1.)
            self.assertAlmostEqual(value, 1.)
