"""Throttle de regime aprendido (fva_car/throttle.py) e salvo entre corridas (main.py)."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402
from fva_car.throttle import MIN_SAMPLES, ThrottleMap  # noqa: E402


def steady(throttle_map, v, u, seconds=3.):
    for _ in range(int(seconds/.02)):
        throttle_map.update(v, u, .02)


class ThrottleMapTests(unittest.TestCase):
    def test_empty_map_is_the_old_line(self):
        throttle_map = ThrottleMap(.32, .22)
        self.assertAlmostEqual(throttle_map.feedforward(.5), .32 + .22*.5)
        self.assertEqual(throttle_map.feedforward(0.), 0.)

    def test_learns_the_measured_regime_and_shifts_the_line_outside(self):
        throttle_map = ThrottleMap(.32, .22)
        steady(throttle_map, .5, .40)   # o carro de 10/11: ~0,40 a 0,5 m/s
        steady(throttle_map, .9, .59)   # e 0,59 a 0,9 m/s
        self.assertAlmostEqual(throttle_map.feedforward(.5), .40, places=3)
        self.assertAlmostEqual(throttle_map.feedforward(.7), .495, places=3)   # interpolado
        self.assertAlmostEqual(throttle_map.feedforward(.3), .40 - .22*.2, places=3)  # reta encostada em 0,5
        self.assertAlmostEqual(throttle_map.feedforward(1.2), .59 + .22*.3, places=3)
        self.assertGreater(throttle_map.learned_s, 5.)

    def test_a_short_visit_does_not_count_and_bad_samples_are_ignored(self):
        throttle_map = ThrottleMap(.32, .22)
        steady(throttle_map, .5, .9, seconds=(MIN_SAMPLES - 1)*.02)
        self.assertAlmostEqual(throttle_map.feedforward(.5), .32 + .11)
        throttle_map.update(.5, 1., .02)    # saturado
        throttle_map.update(5., .5, .02)    # fora das faixas
        self.assertEqual(int(throttle_map.n.sum()), MIN_SAMPLES - 1)

    def test_round_trip_and_corrupt_file(self):
        throttle_map = ThrottleMap(.32, .22)
        steady(throttle_map, .6, .44)
        again = ThrottleMap.from_dict(json.loads(json.dumps(throttle_map.to_dict())), .32, .22)
        self.assertAlmostEqual(again.feedforward(.6), .44, places=3)
        for bad in (None, {}, {"v": [1, 2], "u": [.4, .4], "n": [30, 30]}, {"v": "x", "u": 1, "n": 2}):
            self.assertTrue(np.all(ThrottleMap.from_dict(bad, .32, .22).n == 0))


class MissionFileTests(unittest.TestCase):
    def test_saved_only_after_enough_regime_and_loaded_next_run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'acelerador.json'
            config = {"throttle_learning": True}
            short = ThrottleMap(.32, .22)
            steady(short, .6, .44, seconds=1.)
            self.assertIsNone(main.save_throttle_map(SimpleNamespace(throttle_map=short), config, path))
            learned = ThrottleMap(.32, .22)
            steady(learned, .6, .44, seconds=4.)
            self.assertIsNotNone(main.save_throttle_map(SimpleNamespace(throttle_map=learned), config, path))
            self.assertIsNone(main.save_throttle_map(SimpleNamespace(throttle_map=learned),
                                                     {"throttle_learning": False}, path))
            loaded = ThrottleMap.from_dict(main.load_throttle_map(path), .32, .22)
            self.assertAlmostEqual(loaded.feedforward(.6), .44, places=3)
            self.assertIsNone(main.load_throttle_map(Path(folder)/'nao_existe.json'))



class CarControllerTests(unittest.TestCase):
    """set_vel do Car com o mapa: feedforward dele e aprendizado so em regime e no automatico."""
    def car(self, auto=True):
        from unittest.mock import Mock
        from test_rc_input import Gear, load_car
        module = load_car()
        car = module.Car.__new__(module.Car)
        car._encoder_ready = lambda: True
        car.emergencia, car.gear = False, Gear.FORWARD
        car.sel_tracao = 1 if auto else 0
        car.rc_acelerador = 0.
        car.v = car.vref = car.u = car.a = 0.
        car.dt = car.sample_rate = .02
        car.vel_error_integral = 0.
        car.atuador = Mock()
        car._set_ref = lambda value: setattr(car, 'vref', value)
        car.throttle_map = ThrottleMap(.32, .22)
        return car

    def test_steady_speed_teaches_the_map_and_feeds_forward(self):
        car = self.car()
        car.v = .6
        for _ in range(200):  # 4 s parado na referencia
            car.set_vel(.6)
        self.assertGreater(car.throttle_map.learned_s, 3.)
        self.assertAlmostEqual(car.throttle_map.feedforward(.6), car.u, delta=.01)
        car.v, car.a = .2, 1.  # acelerando: nao e regime
        before = car.throttle_map.learned_s
        car.set_vel(.6)
        self.assertEqual(car.throttle_map.learned_s, before)

    def test_radio_mode_does_not_learn(self):
        car = self.car(auto=False)
        car.rc_acelerador, car.v = .3, .6
        for _ in range(100):
            car.set_vel(0.)
        self.assertEqual(car.throttle_map.learned_s, 0.)


if __name__ == '__main__':
    unittest.main()


class SteeringScaleTests(unittest.TestCase):
    """Escala do servo por lado: so o comando do servo muda; a odometria usa o desejado."""
    def car(self, left=1., right=1., auto=True):
        from unittest.mock import Mock
        from test_rc_input import load_car
        module = load_car()
        car = module.Car.__new__(module.Car)
        car._encoder_ready = lambda: True
        car.emergencia = False
        car.sel_direcao = 1 if auto else 0
        car.rc_direcao = 0.
        car.atuador = Mock()
        car.atuador.set_steer = Mock(side_effect=lambda value: value)
        car.steer_scale_left, car.steer_scale_right = left, right
        return car

    def test_each_side_scales_only_the_servo_command(self):
        car = self.car(left=.9, right=1.1)
        car.set_steer(.2)
        self.assertAlmostEqual(car.st, .2)                              # odometria: o desejado
        self.assertAlmostEqual(car.atuador.set_steer.call_args[0][0], .18)
        car.set_steer(-.2)
        self.assertAlmostEqual(car.atuador.set_steer.call_args[0][0], -.22)
        car.set_steer(.34)  # perto do batente: nunca passa dele
        self.assertLessEqual(abs(car.atuador.set_steer.call_args[0][0]), .3491)

    def test_radio_mode_is_not_scaled(self):
        car = self.car(left=.8, auto=False)
        car.rc_direcao = .5
        car.set_steer(0.)
        self.assertAlmostEqual(car.atuador.set_steer.call_args[0][0], car.st)
