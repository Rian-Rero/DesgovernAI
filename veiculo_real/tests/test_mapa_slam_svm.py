"""Mapa de cones (EKF-SLAM), tracado de margem maxima (SVM) e o MapPath que junta os dois."""
import math
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cone_slam import ConeSLAM  # noqa: E402
from fva_car.corridor import CorridorEstimate  # noqa: E402
from fva_car.map_path import MapPath  # noqa: E402
from fva_car.svm_path import KernelSVM, centerline, densify, path_errors, virtual_border  # noqa: E402


def straight_cones(x0, width=1.3, spacing=1., view=(.8, 3.)):
    """Cones de uma reta no referencial de um carro em x0 (mundo), dentro da visao."""
    xs = np.arange(0., 30., spacing) - x0
    xs = xs[(xs >= view[0]) & (xs <= view[1])]
    return np.c_[xs, np.full(len(xs), width/2)], np.c_[xs, np.full(len(xs), -width/2)]


def arc_cones(radius, width, spacing=.5, angle=math.pi/2):
    """Curva a esquerda de raio `radius` (centro), comecando no eixo do carro."""
    out = []
    for side in (+1, -1):
        r = radius - side*width/2
        t = np.arange(0., angle, spacing/r)
        out.append(np.c_[r*np.sin(t), radius - r*np.cos(t)])
    return out


class KernelSvmTests(unittest.TestCase):
    def test_separates_two_borders(self):
        left, right = straight_cones(0., view=(0., 3.))
        svm = KernelSVM(gamma=1.).fit(np.vstack([left, right]), np.r_[np.ones(len(left)), -np.ones(len(right))])
        self.assertTrue(np.all(svm.decision(left) > 0))
        self.assertTrue(np.all(svm.decision(right) < 0))
        self.assertAlmostEqual(float(svm.decision([[1.5, 0.]])[0]), 0., delta=.1)


class CenterlineTests(unittest.TestCase):
    def test_straight_corridor_is_centered(self):
        left, right = straight_cones(0., view=(-.5, 3.))
        points, margin = centerline(left, right, 1.3)
        self.assertLess(np.max(np.abs(points[:, 1])), .03)
        self.assertAlmostEqual(margin, .65, delta=.05)
        e, psi, kappa, _ = path_errors(points)
        self.assertAlmostEqual(e, 0., delta=.02)
        self.assertAlmostEqual(psi, 0., delta=.03)
        self.assertAlmostEqual(kappa, 0., delta=.05)

    def test_offset_and_heading(self):
        left, right = straight_cones(0., view=(-.5, 3.))
        angle = .2
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        shift = np.array([0., .25])
        points, _ = centerline(left @ rotation.T + shift, right @ rotation.T + shift, 1.3)
        e, psi, _, _ = path_errors(points)
        self.assertAlmostEqual(e, .25/math.cos(angle), delta=.04)
        self.assertAlmostEqual(psi, angle, delta=.05)

    def test_curve_bends_the_right_way(self):
        left, right = arc_cones(2., 1.3)
        points, _ = centerline(left, right, 1.3)
        e, psi, kappa, _ = path_errors(points)
        # R = 2 m: ~6 cm de vies para dentro (a margem maxima entre as bordas nao e o
        # arco central; e por isso que o mapa e so reserva do estimador).
        self.assertAlmostEqual(e, 0., delta=.08)
        self.assertGreater(kappa, .2)  # curva a esquerda: curvatura positiva

    def test_one_border_uses_virtual_cones(self):
        left, _ = straight_cones(0., view=(-.5, 3.))
        virtual = virtual_border(left, 1.3, -1.)
        self.assertTrue(np.allclose(virtual[:, 1], -.65, atol=1e-6))
        points, _ = centerline(left, [], 1.3)
        self.assertLess(np.max(np.abs(points[:, 1])), .03)

    def test_densify_fills_gaps_but_not_jumps(self):
        border = np.array([[0., 0.], [1., 0.], [5., 0.]])
        filled = densify(border, step=.25, max_gap=2.)
        self.assertEqual(len(filled), 6)  # 0, .25, .5, .75, 1 e o 5 sem preencher
        self.assertTrue(np.allclose(filled[-1], [5., 0.]))

    def test_path_must_reach_the_axle(self):
        """Extrapolar um polinomio de 1,5 m ate o eixo da rumos absurdos: sem tracado."""
        left, right = straight_cones(0., view=(1.5, 3.))
        points, _ = centerline(left, right, 1.3)
        self.assertIsNone(path_errors(points))

    def test_empty_input(self):
        self.assertIsNone(centerline([], [], 1.3)[0])


class ConeSlamTests(unittest.TestCase):
    def test_keeps_cones_that_left_the_image(self):
        slam = ConeSLAM()
        step = .1
        for k in range(30):  # 3 m em linha reta, a camera so ve de 0,8 a 3 m
            green, red = straight_cones(k*step)
            slam.step(k*.1, {'green': green, 'red': red}, (step, 0., 0.) if k else None)
        landmarks = slam.landmarks()
        beside = landmarks['green'][landmarks['green'][:, 0] < .8]
        self.assertGreater(len(beside), 0)  # ja sairam da imagem, continuam no mapa
        truth = np.arange(0., 30.) - 29*step
        for point in landmarks['green']:
            self.assertLess(np.min(np.abs(truth - point[0])), .05)
            self.assertAlmostEqual(point[1], .65, delta=.05)

    def test_corrects_odometry_drift(self):
        """Odometria com 10% a mais de avanco: os cones revistos puxam a pose de volta."""
        slam = ConeSLAM()
        step = .1
        for k in range(30):
            green, red = straight_cones(k*step)
            slam.step(k*.1, {'green': green, 'red': red}, (1.1*step, 0., 0.) if k else None)
        self.assertLess(abs(slam.mu[0] - 29*step), .1)  # so odometria: 3,19 m

    def test_one_off_false_positive_is_dropped(self):
        slam = ConeSLAM()
        slam.step(0., {'green': [[2., .65], [1.5, -1.]], 'red': [[2., -.65]]}, None)
        for k in range(1, 15):
            slam.step(k*.1, {'green': [[2., .65]], 'red': [[2., -.65]]}, (0., 0., 0.))
        landmarks = slam.landmarks(min_hits=1)
        self.assertEqual(len(landmarks['green']), 1)

    def test_time_going_back_resets(self):
        slam = ConeSLAM()
        slam.step(1., {'green': [[2., .65]]}, None)
        slam.step(.5, {}, None)
        self.assertEqual(slam.count, 0)


class MapPathTests(unittest.TestCase):
    def run_straight(self, estimate_for, frames=25, replace=False):
        path = MapPath(replace=replace)
        out = None
        for k in range(frames):
            green, red = straight_cones(k*.1)
            out = path.step(k*.1, green, red, True, (.1, 0., 0.) if k else None, False, estimate_for(k))
        return path, out

    def test_keeps_the_estimator_while_it_is_valid(self):
        estimate = CorridorEstimate(True, 'AMBAS', (.01, 0., 0.), 1.3, state_vector=(.01, 0., 0., 0., 1.3))
        path, out = self.run_straight(lambda k: estimate)
        self.assertIs(out, estimate)
        self.assertFalse(path.last['usado'])
        self.assertGreater(path.last['marcos'], 4)

    def test_takes_over_when_the_estimator_is_lost(self):
        valid = CorridorEstimate(True, 'AMBAS', (0., 0., 0.), 1.3, state_vector=(0., 0., 0., 0., 1.3))
        lost = CorridorEstimate(False, 'PERDIDO')
        path, out = self.run_straight(lambda k: valid if k < 20 else lost)
        self.assertTrue(out.valid)
        self.assertEqual(out.state, 'MAPA')
        self.assertTrue(out.slow)
        self.assertAlmostEqual(out.errors[0], 0., delta=.03)
        self.assertTrue(path.last['usado'])

    def test_replace_mode_uses_the_map_always(self):
        estimate = CorridorEstimate(True, 'AMBAS', (.2, 0., 0.), 1.3, state_vector=(.2, 0., 0., 0., 1.3))
        _, out = self.run_straight(lambda k: estimate, replace=True)
        self.assertAlmostEqual(out.errors[0], 0., delta=.03)
        self.assertIn('SVM', out.reason)


class DetectorOptionTests(unittest.TestCase):
    def test_detector_builds_the_map_only_with_the_estimator(self):
        from fva_car.cones import RealConeDetector
        from fva_car.lateral import GroundCalibration
        calibration = GroundCalibration.from_file(Path(__file__).resolve().parent/'dados'/'camera_pista_virtual.json')
        self.assertIsNotNone(RealConeDetector(calibration, corridor=True, cone_map=True).map)
        self.assertTrue(RealConeDetector(calibration, corridor=True, cone_map='substituir').map.replace)
        self.assertIsNone(RealConeDetector(calibration, corridor=True).map)
        self.assertIsNone(RealConeDetector(calibration, corridor=None, cone_map=True).map)

    def test_map_is_on_by_default_in_main(self):
        import main
        self.assertIs(main.parameters['cone_map'], True)


if __name__ == '__main__':
    unittest.main()
