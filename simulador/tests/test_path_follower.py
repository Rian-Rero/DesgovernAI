"""Hierarquia cones/faixa e Stanley compartilhado, sem camera ou simulador real."""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.path_follower import HierarchicalPathFollower, PATH_SOURCE_CODES, _image_points
from fva_car.car import Car


def observation(errors=None):
    return SimpleNamespace(
        errors=errors, path=np.empty((0, 3)),
        red_pixels=np.array([[10., 25.], [20., 20.]]),
        green_pixels=np.array([[45., 25.], [40., 20.]]),
        red_points=np.empty((0, 3)), green_points=np.empty((0, 3)),
        red_lateral_m=.8, green_lateral_m=-.7,
        reason='Corredor valido' if errors is not None else 'Cones insuficientes')


def fake_car(parameters=None):
    car = Mock()
    car.parameters = {} if parameters is None else parameters
    car.t, car.dt, car.v = .05, .05, 1.
    car.vref, car.u, car.st, car.motor_pwm_percent = 1., 0., 0., 0.
    car.get_image.return_value = np.zeros((40, 60, 3), dtype=np.uint8)
    car.sim.getObjectMatrix.return_value = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
    car.sim.getObjectPosition.return_value = [0, 0, 0]
    car.sim.getObjectFloatParam.return_value = math.pi/3
    car.sim.getJointPosition.return_value = 0.
    car.set_steer.side_effect = lambda value: setattr(car, 'st', value)
    car.set_vel.side_effect = lambda value: setattr(car, 'vref', value)
    return car


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        self.car = fake_car()
        self.follower = HierarchicalPathFollower(self.car)
        self.follower.cones = Mock()

    def test_cones_take_priority_without_even_segmenting_yellow(self):
        self.follower.cones.observe.return_value = observation((.2, .1, .3))
        original = self.car.get_image.return_value.copy()
        with patch('fva_car.path_follower.yellow_centers') as yellow:
            self.follower(self.car)
        yellow.assert_not_called()
        self.car.get_image.assert_called_once_with(gray=False)
        self.assertEqual(self.car.path_source, 'cones')
        logged = self.car.log.call_args.kwargs
        self.assertEqual(logged['path_source'], PATH_SOURCE_CODES['cones'])
        self.assertEqual(logged['path_valid'], 1)
        self.assertEqual(logged['cones_valid'], 1)
        self.assertEqual(logged['yellow_evaluated'], 0)
        self.assertEqual(logged['line_valid'], 0)
        self.assertEqual(logged['red_lateral_m'], .8)
        self.assertEqual(logged['green_lateral_m'], -.7)
        self.assertEqual(logged['red_cones'], 2)
        # Overlay nunca altera a captura usada para segmentacao.
        np.testing.assert_array_equal(self.car.get_image.return_value, original)
        self.assertTrue(np.any(self.car.line_image != original))
        self.assertLess(self.car.st, 0.)  # eixo da cena usa sinal invertido
        self.assertGreater(self.car.vref, 0.)

    def test_yellow_is_used_only_when_cones_do_not_define_a_corridor(self):
        self.follower.cones.observe.return_value = observation()
        with patch('fva_car.path_follower.fit_line_errors', return_value=(-.1, -.1, 0.)):
            self.follower(self.car)
        self.assertEqual(self.car.path_source, 'yellow')
        self.car.get_image.assert_called_once()
        logged = self.car.log.call_args.kwargs
        self.assertEqual(logged['path_source'], 1)
        self.assertEqual(logged['cones_valid'], 0)
        self.assertEqual(logged['yellow_evaluated'], 1)
        self.assertEqual(logged['line_valid'], 1)
        self.assertGreater(self.car.st, 0.)

    def test_source_change_keeps_one_stanley_and_its_steering_rate_limit(self):
        controller = self.follower.controller
        self.follower.cones.observe.return_value = observation((.5, .2, .4))
        self.follower(self.car)
        previous = self.car.st
        self.car.t += self.car.dt
        self.follower.cones.observe.return_value = observation()
        with patch('fva_car.path_follower.fit_line_errors', return_value=(-.5, -.2, -.4)):
            self.follower(self.car)
        self.assertIs(self.follower.controller, controller)
        self.assertEqual(self.car.path_source, 'yellow')
        self.assertLessEqual(abs(self.car.st-previous), controller.max_rate*self.car.dt + 1e-12)
        self.follower.cones.observe.return_value = observation((.1, 0., 0.))
        self.car.t += self.car.dt
        self.follower(self.car)
        self.assertEqual(self.car.path_source, 'cones')
        self.assertEqual(self.car.get_image.call_count, 3)
        self.follower.cones.observe.return_value = observation()
        self.car.t += self.car.dt
        self.follower(self.car)
        self.assertEqual(self.car.path_source, 'none')
        self.assertEqual(self.car.vref, 0.)

    def test_loss_stops_and_briefly_holds_steering_then_clears_current_errors(self):
        self.follower.cones.observe.return_value = observation((.3, .2, .4))
        self.follower(self.car)
        previous = self.car.st
        self.follower.cones.observe.return_value = observation()
        self.car.t += .1
        self.follower(self.car)
        self.assertEqual(self.car.path_source, 'none')
        self.assertEqual(self.car.vref, 0.)
        self.assertEqual(self.car.st, previous)
        logged = self.car.log.call_args.kwargs
        self.assertEqual(logged['path_valid'], 0)
        for name in ('cross_track_m', 'heading_error_rad', 'curvature_m_inv',
                     'stanley_heading_rad', 'stanley_cross_track_rad',
                     'stanley_raw_rad', 'stanley_target_rad'):
            self.assertTrue(math.isnan(logged[name]), name)
        self.car.t += .4
        self.follower(self.car)
        self.assertEqual(self.car.st, 0.)
        self.assertEqual(self.follower.controller.steer, 0.)

    def test_nonfinite_cone_geometry_cannot_reach_actuators(self):
        self.follower.cones.observe.return_value = observation((math.nan, 0., 0.))
        self.follower(self.car)
        self.assertEqual(self.car.path_source, 'none')
        self.assertEqual(self.car.vref, 0.)
        self.assertEqual(self.car.st, 0.)

    def test_missing_camera_does_not_use_old_perception(self):
        self.car.get_image.return_value = None
        self.follower(self.car)
        self.follower.cones.observe.assert_not_called()
        self.car.get_image.assert_called_once()
        self.assertEqual(self.car.path_source, 'none')
        self.assertEqual(self.car.vref, 0.)
        self.assertEqual(self.car.line_image.shape, (240, 320, 3))

    def test_curve_in_either_direction_uses_stanley_and_reduces_speed(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                car = fake_car({'velocity_reference': 1.5})
                follower = HierarchicalPathFollower(car)
                follower.cones = Mock()
                follower.cones.observe.return_value = observation((0., direction*.2, direction*.9))
                follower(car)
                self.assertEqual(math.copysign(1., car.st), -direction)
                self.assertLess(car.vref, 1.)
                self.assertGreaterEqual(car.vref, .5)

    def test_expired_yellow_memory_does_not_override_missing_cones(self):
        points = np.column_stack((np.linspace(0., 1., 20), np.zeros(20), np.ones(20)))
        self.follower.line_memory.update(0., points, [0., 0., 1.])
        self.follower.cones.observe.return_value = observation()
        self.car.t = .31
        self.follower(self.car)
        self.assertEqual(self.car.path_source, 'none')
        self.assertEqual(self.car.vref, 0.)

    def test_log_remains_numeric_for_existing_csv_writer(self):
        self.follower.cones.observe.return_value = observation((.1, .2, .3))
        self.follower(self.car)
        logged = self.car.log.call_args.kwargs
        numeric = np.array(list(logged.values()), dtype=float)
        self.assertEqual(len(numeric), len(logged))
        self.assertEqual(logged['path_source'], 2)
        with tempfile.TemporaryDirectory() as folder:
            saved_car = Car.__new__(Car)
            saved_car.logfile = folder
            saved_car.traj = [{'t': 0.}, dict(t=.05, **logged)]
            saved_car.save()
            rows = np.genfromtxt(Path(folder)/'car.csv', delimiter=',', names=True)
            self.assertEqual(rows['path_source'][-1], 2.)
            self.assertTrue(math.isnan(rows['path_source'][0]))

    def test_cone_floor_height_can_be_calibrated_independently_of_paint(self):
        car = fake_car({'cone_ground_z': .002, 'line_ground_z': .007})
        follower = HierarchicalPathFollower(car)
        self.assertEqual(follower.cone_ground_z, .002)
        self.assertEqual(follower.ground_z, .007)


class OverlayProjectionTests(unittest.TestCase):
    def test_world_path_reprojects_to_the_same_pixels_as_ground_projection(self):
        s = math.sqrt(.5)
        rotation = np.array([[0, s, s], [1, 0, 0], [0, s, -s]])
        origin = np.array([0., 0., 1.])
        transform = np.column_stack((rotation, origin))
        points = np.array([[.5, .1, 0.], [1., -.1, 0.]])
        camera = (points-origin) @ rotation
        focal = 320/(2*math.tan(math.pi/6))
        expected = np.column_stack((160-.5-focal*camera[:, 0]/camera[:, 2],
                                    120-.5-focal*camera[:, 1]/camera[:, 2]))
        pixels = _image_points(points, (240, 320, 3), transform, math.pi/3)
        np.testing.assert_array_equal(pixels, np.rint(expected).astype(int))


if __name__ == '__main__':
    unittest.main()
