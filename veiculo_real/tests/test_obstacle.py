"""Parada pelo ultrassom em curva (fva_car/obstacle.py) e a ligacao no main.py."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.corridor import CorridorEstimate  # noqa: E402
from fva_car.obstacle import border_echo, curve_echo  # noqa: E402
import test_mission  # noqa: E402  (modulo, nao a classe: senao o unittest roda os testes dela de novo)
mission = test_mission.mission

WIDTH = 1.4


def observation(vector, state='SO_EXTERNA', valid=True, timestamp=10.):
    estimate = CorridorEstimate(valid, state, tuple(vector[:3]), WIDTH, state_vector=tuple(vector))
    return SimpleNamespace(corridor=estimate, timestamp=timestamp)


def first_echo(vector, side):
    """Menor distancia em que a borda `side` entra no feixe do ultrassom."""
    for distance in np.arange(.15, 1.5, .01):
        if border_echo(distance, vector) == side:
            return float(distance)
    return None


class CurveEchoTests(unittest.TestCase):
    # Curva a esquerda de raio 2 m, carro no centro, virando 12 graus para a esquerda.
    LEFT_CURVE = (0., .15, .5, 0., WIDTH)

    def test_outer_cone_ahead_in_a_curve_is_ignored(self):
        distance = first_echo(self.LEFT_CURVE, 'direita')
        self.assertIsNotNone(distance)
        ignore, reason = curve_echo(distance, observation(self.LEFT_CURVE), math.radians(12), .8, 10.1)
        self.assertTrue(ignore, reason)
        self.assertIn('direita', reason)

    def test_obstacle_in_the_middle_of_the_lane_still_stops(self):
        # Eco a 0,3 m: nenhuma borda passa no feixe tao perto.
        self.assertIsNone(border_echo(.3, self.LEFT_CURVE))
        ignore, reason = curve_echo(.3, observation(self.LEFT_CURVE), math.radians(12), .8, 10.1)
        self.assertFalse(ignore)
        self.assertIn('meio da faixa', reason)

    def test_still_stops_on_straights_without_steering_or_without_vision(self):
        distance = first_echo(self.LEFT_CURVE, 'direita')
        cases = {
            'reta': (observation((0., 0., 0., 0., WIDTH)), math.radians(12), 10.1),
            'nao virou': (observation(self.LEFT_CURVE), math.radians(3), 10.1),
            'virando para fora': (observation(self.LEFT_CURVE), math.radians(-12), 10.1),
            'sem corredor': (observation(self.LEFT_CURVE, state='MEMORIA'), math.radians(12), 10.1),
            'invalido': (observation(self.LEFT_CURVE, valid=False), math.radians(12), 10.1),
            'visao velha': (observation(self.LEFT_CURVE), math.radians(12), 11.),
            'perto da borda': (observation((.5, .15, .5, 0., WIDTH)), math.radians(12), 10.1),
            'sem visao': (SimpleNamespace(timestamp=10.), math.radians(12), 10.1),
        }
        for name, (obs, steer, now) in cases.items():
            with self.subTest(name):
                self.assertFalse(curve_echo(distance, obs, steer, .8, now)[0])


class MissionWiringTests(unittest.TestCase):
    # So os auxiliares do teste da missao (sem herdar e repetir os testes dela).
    setUp = test_mission.MissionTests.setUp
    fake_thread = test_mission.MissionTests.fake_thread
    run_fake = test_mission.MissionTests.run_fake

    def run_cones(self, ignore):
        self.car.distances = iter([(.1, True), (.1, True)])
        follower = Mock()
        with patch.object(mission, 'RealConeDetector', return_value=Mock(classifier=None)), \
                patch.object(mission, 'cone_calibration_warnings', return_value=[]), \
                patch.object(mission, 'RealStanleyFollower', return_value=follower), \
                patch.object(mission, 'curve_echo', return_value=(ignore, 'teste')) as check:
            self.run_fake(dict(self.settings, navigation_mode='cones', save=False, vision_process=False))
        logs = [call.kwargs for call in self.car.log.call_args_list if 'obstacle_stop' in call.kwargs]
        return follower, logs, check

    def test_echo_explained_by_the_curve_does_not_latch_the_stop(self):
        follower, logs, check = self.run_cones(True)
        self.assertTrue(check.called)
        self.assertFalse(any(call.kwargs['obstacle_detected'] for call in follower.apply.call_args_list))
        self.assertEqual([(log['obstacle_ignored'], log['obstacle_stop']) for log in logs], [(1, 0), (1, 0)])

    def test_unexplained_echo_still_latches(self):
        follower, logs, _ = self.run_cones(False)
        # Confirmado na 2a leitura seguida (obstacle_confirm_s).
        self.assertEqual([call.kwargs['obstacle_detected'] for call in follower.apply.call_args_list],
                         [False, True])
        self.assertEqual([log['obstacle_stop'] for log in logs], [0, 1])


if __name__ == '__main__':
    unittest.main()
