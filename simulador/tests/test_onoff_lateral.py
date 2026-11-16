import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import OnOffSteering, orientation_reference
from fva_car.trajetorias import circle, figure_eight, straight


class OrientationReferenceTests(unittest.TestCase):
    def test_matches_angle_to_lookahead_point(self):
        # Ponto a L_d ao longo da tangente, partindo do ponto mais proximo.
        for e_y, e_psi, lookahead in ((.3, .2, .8), (-.5, -.4, 1.2), (0., .1, .5)):
            tangent = np.array([math.cos(e_psi), math.sin(e_psi)])
            normal = np.array([-math.sin(e_psi), math.cos(e_psi)])
            point = e_y * normal + lookahead * tangent
            self.assertAlmostEqual(orientation_reference(e_y, e_psi, lookahead),
                                   math.atan2(point[1], point[0]))

    def test_rejects_invalid_lookahead(self):
        for value in (0., -1., math.nan):
            with self.assertRaises(ValueError):
                orientation_reference(0., 0., value)


class OnOffSteeringTests(unittest.TestCase):
    def setUp(self):
        self.c = OnOffSteering(delta_on=math.radians(15), h_on=math.radians(3),
                               h_off=math.radians(1), max_rate=1e3)

    def test_three_levels_with_hysteresis(self):
        on = math.radians(15)
        self.assertAlmostEqual(self.c.update(math.radians(4), .05), on)
        # Entre h_off e h_on mantem o nivel.
        self.assertAlmostEqual(self.c.update(math.radians(2), .05), on)
        self.assertAlmostEqual(self.c.update(math.radians(.5), .05), 0.)
        self.assertAlmostEqual(self.c.update(math.radians(-4), .05), -on)
        self.assertEqual(self.c.switches, 3)

    def test_rate_limit(self):
        c = OnOffSteering(delta_on=math.radians(20), max_rate=math.radians(60))
        self.assertAlmostEqual(c.update(1., .05), math.radians(3))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            OnOffSteering(h_on=math.radians(1), h_off=math.radians(2))
        with self.assertRaises(ValueError):
            self.c.update(math.nan, .05)


class PathTests(unittest.TestCase):
    def test_straight_offset_sign(self):
        path = straight((0., 0.), 0., offset=.5)
        e_y, e_psi, _ = path.errors((0., 0.), 0.)
        self.assertAlmostEqual(e_y, .5, places=3)  # referencia a esquerda
        self.assertAlmostEqual(e_psi, 0.)

    def test_circle_tangent_and_lap(self):
        path = circle((0., 0.), 0., radius=2.)
        e_y, e_psi, _ = path.errors((0., 0.), 0.)
        self.assertAlmostEqual(e_y, 0., places=3)
        # Percorre a volta inteira sobre a referencia.
        for i in range(0, len(path.points) + 20, 10):
            p = path.points[i % len(path.points)]
            _, _, s = path.errors(p, path.heading[i % len(path.points)])
        self.assertGreater(s, path.length)

    def test_eight_changes_curvature_sign(self):
        path = figure_eight((0., 0.), 0., radius=2.)
        half = len(path.points) // 2
        dpsi = np.diff(np.unwrap(path.heading))
        self.assertTrue(np.all(dpsi[:half - 2] > 0))
        self.assertTrue(np.all(dpsi[half + 1:-2] < 0))


if __name__ == '__main__':
    unittest.main()
