import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import StanleyController, YellowLineFollower


class StanleyTests(unittest.TestCase):
    def test_invalid_configuration_cannot_reach_the_actuator(self):
        for parameter in ('gain', 'softening', 'max_steer', 'max_rate'):
            for value in (0., -1., math.nan, math.inf, -math.inf):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        StanleyController(**{parameter: value})
        for value in (math.pi / 2, math.pi):
            with self.subTest(max_steer=value), self.assertRaises(ValueError):
                StanleyController(max_steer=value)

    def test_each_error_contributes_to_steering(self):
        # A linha a esquerda exige correcao positiva; um erro angular
        # contrario reduz essa correcao antes da saturacao do atuador.
        controller = StanleyController(gain=2., softening=.5,
                                       max_steer=1., max_rate=100.)
        steer = controller.update(.3, -.1, 1., .05)
        distance_term = math.atan(.4)
        self.assertAlmostEqual(controller.heading_term, -.1)
        self.assertAlmostEqual(controller.cross_track_term, distance_term)
        self.assertAlmostEqual(controller.raw_steer, distance_term - .1)
        self.assertAlmostEqual(controller.target_steer, distance_term - .1)
        self.assertAlmostEqual(steer, distance_term - .1)

    def test_telemetry_distinguishes_saturation_and_rate_limit(self):
        controller = StanleyController(max_steer=.2, max_rate=.5)
        steer = controller.update(1., .3, 1., .02)
        self.assertGreater(controller.raw_steer, .2)
        self.assertAlmostEqual(controller.target_steer, .2)
        self.assertAlmostEqual(steer, .01)

    def test_heading_wrap_and_standstill_have_finite_command(self):
        controller = StanleyController(gain=2., softening=.4,
                                       max_steer=1.2, max_rate=100.)
        steer = controller.update(.1, 4 * math.pi + .2, 0., .05)
        self.assertAlmostEqual(controller.heading_term, .2)
        self.assertAlmostEqual(steer, .2 + math.atan(.5))
        self.assertTrue(math.isfinite(steer))
        self.assertAlmostEqual(controller.update(0., -4 * math.pi, 0., .05), 0.)

    def test_invalid_sample_does_not_change_last_command(self):
        controller = StanleyController()
        previous = controller.update(.1, .1, 1., .05)
        for args in ((math.nan, 0., 1., .05), (0., math.inf, 1., .05),
                     (0., 0., math.nan, .05), (0., 0., 1., 0.),
                     (0., 0., 1., -.01)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                controller.update(*args)
            self.assertEqual(controller.steer, previous)


def simulated_car(parameters=None):
    car = Mock()
    car.parameters = {} if parameters is None else parameters
    car.t, car.dt, car.v = .05, .05, 1.
    car.vref, car.u, car.st, car.motor_pwm_percent = 1., 0., 0., 0.
    car.get_image.return_value = np.zeros((40, 60, 3), dtype=np.uint8)
    car.sim.getObjectMatrix.return_value = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
    car.sim.getObjectPosition.return_value = [0, 0, 0]
    car.sim.getObjectFloatParam.return_value = math.pi / 3
    car.set_steer.side_effect = lambda command: setattr(car, 'st', command)
    return car


class StanleyIntegrationTests(unittest.TestCase):
    def test_follower_uses_configured_gains_limits_and_scene_sign(self):
        car = simulated_car({'stanley_gain': 2., 'stanley_softening': .7,
                             'stanley_max_steer_deg': 10.,
                             'stanley_max_rate_deg_s': 40.})
        follower = YellowLineFollower(car)
        with patch('fva_car.lateral.fit_line_errors', return_value=(.3, .2, 0.)):
            follower(car)
        self.assertAlmostEqual(follower.controller.cross_track_term, math.atan(.6 / 1.7))
        self.assertAlmostEqual(follower.controller.target_steer, math.radians(10.))
        self.assertAlmostEqual(car.st, -math.radians(2.))
        logged = car.log.call_args.kwargs
        self.assertAlmostEqual(logged['stanley_heading_rad'], .2)
        self.assertAlmostEqual(logged['stanley_cross_track_rad'], math.atan(.6 / 1.7))
        self.assertAlmostEqual(logged['stanley_raw_rad'], .2 + math.atan(.6 / 1.7))
        self.assertAlmostEqual(logged['stanley_target_rad'], math.radians(10.))

    def test_follower_respects_the_car_hardware_limit(self):
        follower = YellowLineFollower(simulated_car({'stanley_max_steer_deg': 45.}))
        self.assertAlmostEqual(follower.controller.max_steer, math.radians(20.))
        for value in (math.nan, math.inf, -math.inf, 0., -1.):
            with self.subTest(limit=value), self.assertRaises(ValueError):
                YellowLineFollower(simulated_car({'stanley_max_steer_deg': value}))

    def test_lost_line_does_not_log_old_error_terms_as_current(self):
        car = simulated_car()
        follower = YellowLineFollower(car)
        with patch('fva_car.lateral.fit_line_errors', side_effect=[(.1, .1, 0.), None]):
            follower(car)
            car.t += car.dt
            follower(car)
        logged = car.log.call_args.kwargs
        self.assertEqual(logged['line_valid'], 0)
        car.set_vel.assert_called_with(0.)
        for key in ('stanley_heading_rad', 'stanley_cross_track_rad',
                    'stanley_raw_rad', 'stanley_target_rad'):
            self.assertTrue(math.isnan(logged[key]), key)


if __name__ == '__main__':
    unittest.main()
