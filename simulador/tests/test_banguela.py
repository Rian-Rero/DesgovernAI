import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.car import CAR, Car
from fva_car.filter import AlphaFilter
from fva_car.speed_controller import SpeedPI
from identificar_banguela import dynamic_data, integral_equations, nnls_small, simulate


class Motors:
    def __init__(self):
        self.velocity = {}
        self.force = {}

    def setJointTargetVelocity(self, h, value):
        self.velocity[h] = value

    def setJointForce(self, h, value):
        self.force[h] = value


class CoastTests(unittest.TestCase):
    def make_car(self, gear=1):
        car = Car.__new__(Car)
        car.v, car.gear = gear * 1.5, gear
        car.sim = Motors()
        car.motorL, car.motorR = 'L', 'R'
        car.speed_controller = SpeedPI()
        return car

    def test_coast_is_zero_torque_even_with_friction_and_reverse(self):
        for gear in [1, -1]:
            car = self.make_car(gear)
            car.speed_controller.integral = 1.0
            car.set_u(0.0)
            self.assertGreater(car.sim.force['L'], 0.0)
            car.set_coast()
            self.assertEqual(car.sim.force, {'L': 0.0, 'R': 0.0})
            self.assertEqual(car.sim.velocity, {'L': 0.0, 'R': 0.0})
            self.assertEqual(car.motor_torque, 0.0)
            self.assertEqual(car.speed_controller.integral, 0.0)
            self.assertTrue(car.coasting)

    def test_motor_units_allow_2_5_m_s(self):
        car = self.make_car()
        car.set_u(0.3)
        self.assertAlmostEqual(car.sim.velocity['L'] * CAR['RW'], 2.5)
        self.assertFalse(car.coasting)

    def test_velocity_excludes_vertical_motion_and_preserves_actual_sign(self):
        car = self.make_car()
        car.robot = 1
        car.v_filt = AlphaFilter(alpha=1.0)
        car.w_filt = AlphaFilter(alpha=1.0)
        car.sim = SimpleNamespace(
            getObjectVelocity=lambda h: ([-0.5, 0.0, -9.0], [0.0, 0.0, 0.0]),
            getObjectMatrix=lambda h, world: [0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0],
        )
        v, _ = car.get_vel()
        self.assertEqual(v, -0.5)
        self.assertEqual(car.v_raw, -0.5)

    def test_acceleration_uses_current_sampling_interval(self):
        car = self.make_car()
        car.t, car.tinit, car.dt, car.v = 0.1, 0.0, 0.5, 1.0
        car.get_time = lambda: 0.15
        car.get_vel = lambda: (1.1, 0.0)
        car.get_pos = lambda: np.zeros(2)
        car.get_yaw = lambda: 0.0
        car.a_filt = AlphaFilter(alpha=1.0)
        car.get_states()
        self.assertAlmostEqual(car.a, 2.0)

    def test_integral_identification_recovers_known_coast_losses(self):
        trials = []
        for initial in [0.5, 1.0, 1.5, 2.0, 2.5]:
            t = np.arange(0.0, 4.0, 0.05)
            # Solucao analitica independente: v_dot=-0.6-0.2*v.
            v = (initial + 3.0) * np.exp(-0.2 * t) - 3.0
            valid = v >= 0.15
            trials.append({'data': {'t': t[valid], 'v_raw': v[valid],
                                    'coast': np.ones(np.sum(valid), dtype=bool)}})
        matrix, response = integral_equations(trials)
        coefficients = nnls_small(matrix, response)
        np.testing.assert_allclose(coefficients[:2], [0.6, 0.2], atol=0.001)
        self.assertLess(coefficients[2], 0.001)

    def test_zero_input_prediction_matches_known_constant_deceleration(self):
        t = np.arange(0.0, 1.01, 0.05)
        predicted = simulate(t, np.zeros(len(t)), 1.0, [0.6, 0, 0], [1], 0)[:, 0]
        np.testing.assert_allclose(predicted, 1.0 - 0.6 * t, atol=1e-12)

    def test_actuator_lag_matches_analytic_step_response(self):
        t = np.arange(0.0, 1.01, 0.05)
        inputs = np.ones(len(t))
        inputs[0] = 0.0
        prediction = simulate(t, inputs, 0.0, [0, 0, 0], [1], 0.2)[:, 0]
        elapsed = np.maximum(0.0, t - 0.05)
        exact = elapsed - 0.2 * (1.0 - np.exp(-elapsed / 0.2))
        np.testing.assert_allclose(prediction, exact, atol=1e-12)

    def test_band_selection_preserves_samples_near_noisy_boundary(self):
        data = {'t': np.arange(8) * 0.1 + 0.3,
                'v_raw': np.array([0.0, 0.46, 0.44, 0.49, 0.51, 0.46, 0.47, 0.42]),
                'u_motor': np.ones(8)}
        t, _, v = dynamic_data({'data': data})
        np.testing.assert_allclose(np.diff(t), 0.1)
        np.testing.assert_allclose(v, [0.46, 0.44, 0.49, 0.51, 0.46, 0.47])


if __name__ == '__main__':
    unittest.main()
