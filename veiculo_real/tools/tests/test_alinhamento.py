"""Giro da percepcao medido pelos cones parados com o carro andando (alinhamento_camera.py)."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alinhamento_camera as tool  # noqa: E402


class AlignmentTests(unittest.TestCase):
    def synthetic_rows(self, rotation_deg, turn_per_m, seed=0):
        """Linhas (d, x, residuo) como sairiam de cone_pairs com uma percepcao girada."""
        rng = np.random.default_rng(seed)
        d = rng.uniform(.04, .1, 800)
        x = rng.uniform(.7, 2.1, 800)
        residual = d*math.sin(math.radians(rotation_deg)) + (x + tool.WHEELBASE)*d*turn_per_m
        return np.c_[d, x, residual + rng.normal(0., .003, 800)]

    def test_rotation_and_unseen_turn_are_separated(self):
        angle, turn = tool.fit(self.synthetic_rows(-11., -.17))
        self.assertAlmostEqual(angle, -11., delta=1.)
        self.assertAlmostEqual(turn, -.17, delta=.02)
        angle, turn = tool.fit(self.synthetic_rows(0., 0., seed=1))
        self.assertAlmostEqual(angle, 0., delta=1.)

    def test_left_and_right_steering_response(self):
        rng = np.random.default_rng(3)
        d = rng.uniform(.04, .1, 1500)
        x = rng.uniform(.7, 2.1, 1500)
        curvature = rng.uniform(-1., 1., 1500)
        real = np.where(curvature > 0, 1.2, .9)*curvature - .1  # esquerda vira 20% a mais
        residual = (x + tool.WHEELBASE)*d*(real - curvature) + rng.normal(0., .002, 1500)
        angle, bias, left, right = tool.fit_sides(np.c_[d, x, residual, curvature])
        self.assertAlmostEqual(left, 1.2, delta=.03)
        self.assertAlmostEqual(right, .9, delta=.03)
        self.assertAlmostEqual(bias, -.1, delta=.02)
        self.assertAlmostEqual(angle, 0., delta=1.)

    def test_pairs_from_a_rotated_perception(self):
        from fva_car.lateral import GroundCalibration
        H = np.array([[0., 0., 100.], [-1., 0., 320.], [0., 1., -90.]])  # pixel -> chao (testes)
        calibration = GroundCalibration((640, 480), H)
        inverse = np.linalg.inv(H)
        cones = np.array([[x, y] for x in np.arange(1., 6., .5) for y in (-.6, .6)])
        rotation = math.radians(-8.)
        R = np.array([[math.cos(rotation), -math.sin(rotation)], [math.sin(rotation), math.cos(rotation)]])
        records, position = [], 0.
        for frame in range(25):
            local = (cones - [position, 0.]) @ R.T  # o que a percepcao girada ve
            q = np.c_[local, np.ones(len(local))] @ inverse.T
            pixels = q[:, :2]/q[:, 2:3]
            reduced = (pixels + .5)/2 - .5
            records.append({'vision': {'tracking': {'motion': 'odometria', 'motion_m': [.06, 0., 0.]},
                                       'detection': {'green': {'px': reduced.tolist()}, 'red': {'px': []}}}})
            position += .06
        rows = tool.cone_pairs(calibration, records, wheelbase=0.)
        self.assertGreater(len(rows), 50)
        self.assertAlmostEqual(tool.fit(rows, wheelbase=0.)[0], -8., delta=1.)


if __name__ == '__main__':
    unittest.main()
