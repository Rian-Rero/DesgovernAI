"""Pure pursuit, previa de curva e rampa da vref (fva_car/lateral.py) e config do main."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import (LineObservation, RealStanleyFollower, path_points, preview_speed,  # noqa: E402
                             pure_pursuit_steer)
import main  # noqa: E402
from test_lateral import fake_car  # noqa: E402

L = .36


class PathGeometryTests(unittest.TestCase):
    def test_path_points_on_a_circle(self):
        kappa = .5
        points = path_points((0., 0., kappa), 0., [1., 2.])
        for s, (x, y) in zip((1., 2.), points):
            self.assertAlmostEqual(x, math.sin(kappa*s)/kappa, places=3)
            self.assertAlmostEqual(y, (1 - math.cos(kappa*s))/kappa, places=3)

    def test_pure_pursuit_steady_state_on_a_circle(self):
        # Regime: traseiro num circulo de raio R, dianteiro no centro do corredor
        # (raio sqrt(R^2+L^2), rumo atan(L/R) em relacao ao carro).
        for radius in (1.5, -2.5, 4.):
            for lookahead in (.4, 1.):
                with self.subTest(radius=radius, lookahead=lookahead):
                    sign = math.copysign(1., radius)
                    front = math.hypot(radius, L)
                    errors = (0., sign*math.atan(L/abs(radius)), sign/front)
                    steer, _, _ = pure_pursuit_steer(errors, 0., lookahead, L)
                    self.assertAlmostEqual(steer, math.atan(L/radius), delta=.002)

    def test_pure_pursuit_turns_toward_the_path(self):
        self.assertGreater(pure_pursuit_steer((.2, 0., 0.), 0., .6, L)[0], 0.)   # centro a esquerda
        self.assertLess(pure_pursuit_steer((0., -.2, 0.), 0., .6, L)[0], 0.)     # caminho virando a direita

    def test_preview_slows_before_a_curve_it_can_only_coast_into(self):
        reference = 1.5
        straight, _, _ = preview_speed((0., 0., 0.), 0., reference, 2.5)
        self.assertEqual(straight, reference)
        # Curva de raio 2 m comecando ~1 m a frente (clotoide).
        ahead, kappa, where = preview_speed((0., 0., 0.), .5, reference, 2.5, lateral_accel=.4,
                                            deceleration=.5, speed=1.5)
        bend = math.sqrt(.4/1.25)
        self.assertLess(ahead, reference)
        self.assertGreater(ahead, bend)        # ainda pode rolar ate la
        self.assertAlmostEqual(kappa, 1.25, places=2)
        self.assertAlmostEqual(where, 2.5, places=2)
        inside, _, _ = preview_speed((0., 0., .5), 0., reference, 2.5, lateral_accel=.4)
        self.assertAlmostEqual(inside, math.sqrt(.4/.5), places=3)


class SightLimitTests(unittest.TestCase):
    def test_straight_top_speed_is_what_still_lets_it_coast_into_the_tightest_curve(self):
        # Reta vista ate 2,5 m; depois pode vir curva de R = 1,5 m. Carro so rola (0,5 m/s^2).
        speed = 2.
        for _ in range(30):  # ponto fixo: a reacao depende da velocidade atual
            speed, kappa, where = preview_speed((0., 0., 0.), 0., 2., 2.5, lateral_accel=.4,
                                                deceleration=.5, speed=speed, sight_curvature=1/1.5)
        bend = math.sqrt(.4*1.5)
        self.assertAlmostEqual(where, 2.5)
        self.assertGreater(speed, 1.4)    # corre bem mais que a curva...
        self.assertLess(speed, 1.75)      # ...mas so o que ainda da para rolar ate ela
        self.assertAlmostEqual(speed**2, bend**2 + 2*.5*(2.5 - .3*speed), places=2)
        # Sem o limite, so curvas vistas reduzem: na reta fica na referencia.
        self.assertEqual(preview_speed((0., 0., 0.), 0., 2., 2.5, lateral_accel=.4)[0], 2.)

    def test_mission_turns_min_radius_into_sight_limit_only_with_estimator(self):
        config = dict(main.parameters)
        self.assertAlmostEqual(main.follower_config(config, 'cones', {})['speed_sight_curvature'], 1/1.5)
        self.assertEqual(main.follower_config(dict(config, min_curve_radius_m=0), 'cones', {})
                         ['speed_sight_curvature'], 0.)
        self.assertEqual(main.follower_config(config, 'cones', None)['speed_sight_curvature'], 0.)
        self.assertNotIn('speed_sight_curvature', main.follower_config(config, 'line', None))


class FollowerTests(unittest.TestCase):
    settings = dict(velocity_reference=1., curve_min_speed=.2, vision_latency_compensation=False,
                    speed_filter_s=0.)

    def apply(self, follower, observation, car=None, now=10.):
        car = car or fake_car()
        follower.apply(car, observation, now)
        return car

    def test_pure_pursuit_needs_the_estimator_path(self):
        follower = RealStanleyFollower(dict(self.settings, steering_law='pure_pursuit'))
        self.apply(follower, LineObservation(10., errors=(.1, 0., 0.), path_dkappa=0.))
        self.assertEqual(follower.last_state['steering_law'], 'pure_pursuit')
        self.assertIsNotNone(follower.last_state['pursuit'])
        follower = RealStanleyFollower(dict(self.settings, steering_law='pure_pursuit'))
        self.apply(follower, LineObservation(10., errors=(.1, 0., 0.)))  # linha amarela: sem dkappa
        self.assertEqual(follower.last_state['steering_law'], 'stanley')

    def test_invalid_steering_law_is_rejected(self):
        with self.assertRaises(ValueError):
            RealStanleyFollower({'steering_law': 'mpc'})

    def test_preview_reduces_vref_before_the_curve(self):
        # Reta agora, curva fechada 1 m a frente: a 2 m/s nao da para rolar ate ela.
        settings = dict(self.settings, velocity_reference=2.)
        plain = self.apply(RealStanleyFollower(settings),
                           LineObservation(10., errors=(0., 0., 0.), path_dkappa=1.))
        preview = self.apply(RealStanleyFollower(dict(settings, speed_preview_m=2.5)),
                             LineObservation(10., errors=(0., 0., 0.), path_dkappa=1.))
        self.assertAlmostEqual(plain.vref, 2., places=3)
        self.assertLess(preview.vref, 1.3)

    def test_vref_rises_in_a_ramp_and_falls_at_once(self):
        follower = RealStanleyFollower(dict(self.settings, speed_rise_m_s2=1., curvature_filter_s=0.))
        car = fake_car()
        car.dt = .02
        self.apply(follower, LineObservation(10., errors=(0., 0., 0.)), car, 10.)
        self.assertAlmostEqual(car.vref, 1., places=3)  # primeiro passo: sem historico
        follower.speed_target = .3
        self.apply(follower, LineObservation(10.02, errors=(0., 0., 0.)), car, 10.02)
        self.assertAlmostEqual(car.vref, .32, places=3)  # sobe 1 m/s^2 * 0,02 s
        follower.speed_target = 1.
        self.apply(follower, LineObservation(10.04, errors=(0., 0., 1.)), car, 10.04)
        self.assertAlmostEqual(car.vref, math.sqrt(.4/1.), places=2)  # curva: desce na hora


class MissionConfigTests(unittest.TestCase):
    def test_line_mode_and_legacy_cones_keep_stanley(self):
        config = dict(main.parameters)
        self.assertEqual(main.follower_config(config, 'cones', {})['steering_law'], 'pure_pursuit')
        for mode, corridor in (('line', None), ('cones', None)):
            derived = main.follower_config(config, mode, corridor)
            self.assertEqual(derived['steering_law'], 'stanley')
            self.assertEqual(derived['speed_preview_m'], 0.)
            self.assertEqual(derived['speed_rise_m_s2'], 0.)


if __name__ == '__main__':
    unittest.main()
