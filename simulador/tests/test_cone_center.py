import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import ConeCenterPath
from fva_car.lateral import BicycleModel, StanleyController, curve_speed


class ConeCenterTests(unittest.TestCase):
    def test_straight_middle_and_orientation_with_offset(self):
        x = np.arange(-10., 40., 2.)
        red = np.column_stack((x, np.full(len(x), 2.)))
        green = np.column_stack((x+.3, np.full(len(x), -2.)))
        path = ConeCenterPath(red, green, [0, 0], [1, 0])
        lateral, heading, width, curvature = path.errors([0, .3], [1, 0])
        self.assertAlmostEqual(lateral, -.3)
        self.assertAlmostEqual(heading, 0.)
        self.assertAlmostEqual(curvature, 0.)
        self.assertGreater(width, 3.9)
        self.assertLess(width, 4.5)

    def test_curve_tracking_in_both_directions_without_stopping(self):
        theta = np.linspace(0, 2*math.pi, 40, endpoint=False)
        inner = 4*np.column_stack((np.cos(theta), np.sin(theta)))
        outer = 8*np.column_stack((np.cos(theta+.04), np.sin(theta+.04)))
        for sign in (-1, 1):
            red, green = (inner, outer) if sign == 1 else (outer, inner)
            path = ConeCenterPath(red, green, [6., 0.], [0, sign])
            self.assertTrue(path.closed)
            state = np.array([6.3, 0., sign*math.pi/2])
            model = BicycleModel()
            control = StanleyController(gain=1.5, softening=.5, max_rate=math.radians(40))
            tail = []
            for i in range(1000):
                forward = np.array([math.cos(state[2]), math.sin(state[2])])
                errors = path.errors(state[:2]+.302*forward, forward)
                self.assertIsNotNone(errors)
                lateral, heading, width, curvature = errors
                speed = curve_speed(1.5, curvature, lateral, heading)
                steer = control.update(lateral, heading+math.atan(.302*curvature), speed, .05)
                state += .05*model.derivative(state, speed, steer)
                if i > 300:
                    tail.append(lateral)
                    self.assertEqual(math.copysign(1., curvature), sign)
            self.assertLess(max(abs(np.asarray(tail))), .12)

    def test_neighbor_lane_does_not_move_straight_center(self):
        x = np.arange(-10., 40., 2.)
        red = np.vstack((np.column_stack((x, np.full(len(x), 2.))),
                         np.column_stack((x, np.full(len(x), 3.7)))))
        green = np.column_stack((x+.3, np.full(len(x), -2.)))
        path = ConeCenterPath(red, green, [0, 0], [1, 0])
        lateral, heading, _, _ = path.errors([0, 0], [1, 0])
        self.assertAlmostEqual(lateral, 0., places=8)
        self.assertAlmostEqual(heading, 0., places=8)


if __name__ == '__main__':
    unittest.main()
