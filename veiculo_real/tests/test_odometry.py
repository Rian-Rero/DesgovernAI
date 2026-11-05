"""Historico de poses: delta do carro entre capturas, no referencial do carro."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.odometry import PoseHistory
from fva_car.cones import RealConeDetector
from test_cones import scene


class PoseHistoryTests(unittest.TestCase):
    def test_delta_is_expressed_in_the_car_frame_at_t0(self):
        poses = PoseHistory()
        # Carro apontando para +y do mundo (th = 90 graus) andando 1 m/s.
        for k in range(11):
            poses.append(k*.1, 2., k*.1, math.pi/2)
        forward, left, yaw = poses.delta(.2, .7)
        self.assertAlmostEqual(forward, .5)
        self.assertAlmostEqual(left, 0.)
        self.assertAlmostEqual(yaw, 0.)

    def test_interpolates_turn_and_wraps_angle(self):
        poses = PoseHistory()
        poses.append(0., 0., 0., 2*math.pi - .1)
        poses.append(1., 1., 0., .1)  # cruza 0 rad: giro de +0,2
        self.assertAlmostEqual(poses.delta(0., 1.)[2], .2)
        self.assertAlmostEqual(poses.delta(0., .5)[2], .1)

    def test_outside_recorded_interval_returns_none(self):
        poses = PoseHistory()
        self.assertIsNone(poses.delta(0., 1.))
        poses.append(1., 0., 0., 0.)
        poses.append(2., 1., 0., 0.)
        self.assertIsNone(poses.delta(.5, 1.5))
        self.assertIsNone(poses.delta(1., 2.5))
        self.assertIsNotNone(poses.delta(1., 2.05))  # pequena extrapolacao
        poses.append(1.5, 9., 9., 0.)  # fora de ordem: ignorado
        poses.append(3., float('nan'), 0., 0.)
        self.assertEqual(len(poses.poses), 2)

    def test_cone_detector_passes_odometry_delta_to_the_map(self):
        calibration, frame = scene()
        odometry = Mock()
        odometry.delta.return_value = (.05, 0., 0.)
        detector = RealConeDetector(calibration, odometry=odometry)
        first = detector.observe_frame(frame, 10.)
        second = detector.observe_frame(frame, 10.05)
        odometry.delta.assert_called_once_with(10., 10.05)
        self.assertEqual(first.cone_result.motion_source, 'sem odometria')
        self.assertEqual(second.cone_result.motion_source, 'odometria')
        self.assertTrue(second.valid, second.reason)


if __name__ == '__main__':
    unittest.main()
