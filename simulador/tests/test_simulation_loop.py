import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.car import Car
from fva_car.filter import AlphaFilter
from fva_car.speed_controller import SpeedPI
from main import parameters, run_mission


class SimulationLoopTests(unittest.TestCase):
    def test_interrupted_mission_saves_log_and_closes_car(self):
        for interrupted in [True, False]:
            car = Mock()
            car.t = 0.0
            car.traj = [{'t': 0.0}]
            if interrupted:
                car.step.side_effect = KeyboardInterrupt
            else:
                car.step.return_value = False
            with patch('main.Car', return_value=car):
                run_mission(dict(parameters, real_time=False), display=False)
            car.save.assert_called_once()
            car.close.assert_called_once()

    def test_mission_closes_car_if_saving_fails(self):
        car = Mock()
        car.t = parameters['ts']
        car.traj = [{'t': 0.0}]
        car.save.side_effect = OSError('disk unavailable')
        with patch('main.Car', return_value=car), self.assertRaises(OSError):
            run_mission(parameters, display=False)
        car.close.assert_called_once()

    def test_states_use_one_consistent_remote_snapshot(self):
        car = Car.__new__(Car)
        car.remote = Mock()
        car.robot = 1
        car.t, car.tinit, car.v = 0.0, 0.0, 0.0
        car.v_filt = AlphaFilter(alpha=1)
        car.w_filt = AlphaFilter(alpha=1)
        car.a_filt = AlphaFilter(alpha=1)
        car.sim = Mock()
        car.remote.get_state.return_value = {
            'time': 0.05, 'position': [1, 2, 3],
            'quaternion': [0, 0, 0, 1],
            'matrix': [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0],
            'linear': [1.5, 0, -9], 'angular': [0, 0, 0.1],
        }
        car.get_states()
        car.remote.get_state.assert_called_once_with(car.robot)
        self.assertEqual(car.sim.mock_calls, [])
        self.assertEqual(car.v_raw, 1.5)
        self.assertEqual(car.v, 1.5)
        self.assertAlmostEqual(car.a, 30)
        np.testing.assert_array_equal(car.p, [1, 2])

    def test_first_step_initializes_script_before_reading_state(self):
        car = Car.__new__(Car)
        car.speed_controller = SpeedPI()
        car.motorL, car.motorR = 'L', 'R'
        car.steerL, car.steerR = 'steerL', 'steerR'
        car.client = Mock()
        car.sim = Mock()
        car.get_time = Mock(return_value=0.05)
        car.get_states = Mock()
        car.set_forward = Mock()
        car.set_u = Mock()
        car.set_steer = Mock()
        car.beep = Mock()
        car.save_traj = Mock()
        calls = Mock()
        calls.attach_mock(car.sim.startSimulation, 'start')
        calls.attach_mock(car.client.step, 'step')
        calls.attach_mock(car.get_states, 'state')
        car.start_mission()
        self.assertEqual([entry[0] for entry in calls.mock_calls], ['start', 'step', 'state'])


if __name__ == '__main__':
    unittest.main()
