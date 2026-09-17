import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.car import CAR, Car
from fva_car.speed_controller import (
    COAST_DECELERATION, MOTOR_INPUT_GAIN, MOTOR_TORQUE_FACTOR, SpeedPI,
)


class SpeedControllerTests(unittest.TestCase):
    def test_integral_uses_elapsed_simulation_time(self):
        controller = SpeedPI(kp=2.0, ki=3.0)
        controller.update(0.1, 0.02, -1, 1)
        output = controller.update(0.1, 0.07, -1, 1)
        self.assertAlmostEqual(output, 0.206)
        self.assertAlmostEqual(controller.integral, 0.009)

    def test_saturation_does_not_accumulate_integral(self):
        for direction in [-1, 1]:
            controller = SpeedPI()
            for _ in range(1000):
                output = controller.update(direction * 10, 0.01, -1, 1)
                self.assertEqual(output, direction)
            self.assertEqual(controller.integral, 0.0)
            self.assertEqual(controller.update(0, 0.01, -1, 1), 0.0)

    def test_integral_can_unwind_while_output_is_saturated(self):
        controller = SpeedPI(kp=1, ki=1)
        controller.integral = 2
        for _ in range(20):
            controller.update(-0.2, 0.1, -1, 1)
        self.assertAlmostEqual(controller.integral, 1.6)

    def test_zero_time_does_not_integrate_and_negative_time_is_rejected(self):
        controller = SpeedPI()
        controller.update(0.01, 0, -1, 1)
        self.assertEqual(controller.integral, 0)
        with self.assertRaises(ValueError):
            controller.update(0.01, -0.01, -1, 1)

    def make_car(self, gear=1):
        car = Car.__new__(Car)
        car.v_raw = 0.0
        car.v = gear * 1.5
        car.dt = 0.01
        car.gear = gear
        car.speed_controller = SpeedPI()
        car.set_u = Mock()
        return car

    def test_set_vel_uses_filtered_velocity_and_converts_normalized_input(self):
        motor_gain = 2 * MOTOR_TORQUE_FACTOR * MOTOR_INPUT_GAIN
        expected = COAST_DECELERATION / motor_gain - CAR['MI'] * CAR['GRAV']
        for gear in [-1, 1]:
            car = self.make_car(gear)
            car.set_vel(gear * 1.5)
            self.assertAlmostEqual(car.set_u.call_args.args[0], expected)
            self.assertEqual(car.vref, gear * 1.5)

    def test_requested_stop_does_not_apply_feedforward_at_rest(self):
        car = self.make_car()
        car.v_raw = 0.0
        car.speed_controller.integral = 1.0
        car.set_vel(0)
        car.set_u.assert_called_once_with(0.0, compensate_friction=False)
        self.assertEqual(car.speed_controller.integral, 0)

    def test_rejects_constant_load_with_bounded_actuation(self):
        controller = SpeedPI()
        velocity, acceleration = 0.5, 0.0
        peak = velocity
        for _ in range(1000):
            output = controller.update(1.5 - velocity, 0.01, -1.0824, 0.8580)
            self.assertLessEqual(output, 0.8580)
            self.assertGreaterEqual(output, -1.0824)
            # Integra a planta independentemente em subpassos de 1 ms.
            for _ in range(10):
                acceleration += 0.001 * (output - 0.1 - acceleration) / 0.009302
                velocity += 0.001 * acceleration
            peak = max(peak, velocity)
        self.assertLess(abs(velocity - 1.5), 0.001)
        self.assertLess(peak, 1.55)


if __name__ == '__main__':
    unittest.main()
