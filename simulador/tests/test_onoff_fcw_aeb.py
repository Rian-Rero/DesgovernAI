import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcw_aeb import TTC_FCW, FcwAeb, stopping_distance
from onoff import OnOffController


def mock_car(v, dist=4.0, t=10.0):
    car = Mock()
    car.v, car.t = v, t
    car.get_distance.return_value = (dist, True)
    return car


class OnOffTests(unittest.TestCase):
    def test_hysteresis_keeps_state_inside_band(self):
        control = OnOffController(vref=1.0, hysteresis=0.05, u_max=1.0)
        commands = []
        for v in [0.0, 0.97, 1.04, 1.06, 1.03, 0.96, 0.94]:
            car = mock_car(v)
            control(car)
            commands.append(car.set_u.call_args.args[0])
        self.assertEqual(commands, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0])

    def test_reference_must_exceed_hysteresis(self):
        with self.assertRaises(ValueError):
            OnOffController(vref=0.05, hysteresis=0.05)


class FcwAebTests(unittest.TestCase):
    def setUp(self):
        self.control = FcwAeb(vref=1.0, obstacle='fixo', distance=8.0, spawn_time=math.inf)

    def test_stopping_distance(self):
        self.assertAlmostEqual(stopping_distance(1.0), 0.20 + 0.30 + 0.5)
        self.assertAlmostEqual(stopping_distance(-1.5), stopping_distance(1.5))

    def test_warning_comes_before_braking(self):
        car = mock_car(1.0, dist=TTC_FCW * 1.0 - 0.01)
        self.control(car)
        self.assertTrue(self.control.fcw)
        self.assertFalse(self.control.aeb)
        car.set_vel.assert_called_once_with(1.0)

    def test_braking_latches_until_stop_then_coasts(self):
        self.control(mock_car(1.0, dist=0.9))
        self.assertTrue(self.control.aeb)
        car = mock_car(0.5, dist=4.0)  # obstaculo saiu do campo do sensor
        self.control(car)
        car.set_u.assert_called_once_with(-1.0)
        car.set_vel.assert_not_called()
        car = mock_car(0.01, dist=4.0)
        self.control(car)
        car.set_coast.assert_called_once()
        self.assertTrue(self.control.aeb)

    def test_no_reading_means_no_warning(self):
        car = mock_car(1.0, dist=4.0)
        self.control(car)
        self.assertFalse(self.control.fcw)
        self.assertTrue(math.isnan(car.log.call_args.kwargs['ttc']))


if __name__ == '__main__':
    unittest.main()
