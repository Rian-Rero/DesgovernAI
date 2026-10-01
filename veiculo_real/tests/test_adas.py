"""Decisao ADAS, controlador PI e PWM sem acesso a GPIO/I2C/serial."""

import contextlib
import importlib
import io
import math
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


PACKAGE = "_fva_braking_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "fva_car")]
servokit = types.ModuleType("adafruit_servokit")
servokit.ServoKit = Mock(side_effect=AssertionError("Hardware nao permitido nos testes"))
with patch.dict(sys.modules, {
    PACKAGE: package,
    f"{PACKAGE}.encoder": types.ModuleType("encoder"),
    f"{PACKAGE}.imu": types.ModuleType("imu"),
    "adafruit_servokit": servokit,
}):
    adas = importlib.import_module(f"{PACKAGE}.adas")
    car_module = importlib.import_module(f"{PACKAGE}.car")
    servos = importlib.import_module(f"{PACKAGE}.servos")

Mode = adas.BrakingMode


class DecisionTests(unittest.TestCase):
    def make_adas(self):
        return adas.BrakingADAS(adas.BrakingConfig(coast_deceleration=0.2))

    def test_velocity_and_distance_select_cruise_coast_or_brake(self):
        for speed, distance, mode in (
            (1.0, 4.0, Mode.CRUISE),
            (0.2, 0.42, Mode.COAST),
            (1.0, 1.0, Mode.BRAKE),
        ):
            with self.subTest(speed=speed, distance=distance):
                result = self.make_adas().update(speed, distance, True, 1.0, 0)
                self.assertEqual(result.mode, mode)

    def test_stopping_distance_includes_latency_and_clearance(self):
        system = self.make_adas()
        self.assertAlmostEqual(system.stopping_distance(1.0), 0.2 + 0.4 + 2.5)
        self.assertAlmostEqual(system.stopping_distance(2.0), 0.2 + 0.8 + 10.0)

    def test_coast_escalates_and_brake_does_not_chatter(self):
        system = self.make_adas()
        self.assertEqual(system.update(1, 3.3, True, 1, 0).mode, Mode.COAST)
        self.assertEqual(system.update(1, 2, True, 1, 0.02).mode, Mode.BRAKE)
        self.assertEqual(system.update(0.8, 4, True, 1, 0.04).mode, Mode.BRAKE)
        self.assertEqual(system.update(0, 0.2, True, 1, 0.06).mode, Mode.HOLD)

    def test_resume_requires_stopped_clear_path_and_continuous_confirmation(self):
        system = self.make_adas()
        system.update(1, 1, True, 1, 0)
        self.assertEqual(system.update(0, 4, True, 1, 1).mode, Mode.HOLD)
        self.assertEqual(system.update(0, 0.3, True, 1, 1.4).mode, Mode.HOLD)
        self.assertEqual(system.update(0, 4, True, 1, 2).mode, Mode.HOLD)
        self.assertEqual(system.update(0, 4, True, 1, 2.49).mode, Mode.HOLD)
        self.assertEqual(system.update(0, 4, True, 1, 2.51).mode, Mode.CRUISE)

    def test_invalid_sensors_block_propulsion(self):
        system = self.make_adas()
        self.assertEqual(system.update(1, 4, False, 1, 0).mode, Mode.BRAKE)
        self.assertEqual(system.update(0, 4, False, 1, 1).mode, Mode.FAULT)
        self.assertEqual(system.update(1, 4, True, 1, 2, speed_valid=False).mode, Mode.FAULT)
        self.assertEqual(system.update(math.nan, 4, True, 1, 3).mode, Mode.FAULT)
        self.assertEqual(system.update(1, math.nan, True, 1, 4).mode, Mode.BRAKE)

    def test_unintended_backward_motion_is_neutral(self):
        result = self.make_adas().update(-0.2, 0.2, True, 1, 0)
        self.assertEqual(result.mode, Mode.HOLD)

    def test_zero_reference_and_close_obstacle_do_not_start_vehicle(self):
        self.assertEqual(self.make_adas().update(0, 4, True, 0, 0).mode, Mode.COAST)
        self.assertEqual(self.make_adas().update(0, 0.2, True, 1, 0).mode, Mode.HOLD)

    def test_invalid_configuration_is_rejected(self):
        for options in ({"coast_deceleration": 0}, {"max_brake": 2}, {"reaction_time": math.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                adas.BrakingConfig(**options)


class FakeActuator:
    def __init__(self):
        self.commands = []

    def get_gear(self):
        return servos.Gear.FORWARD

    def set_throttle(self, command):
        self.commands.append(("drive", command))

    def set_neutral(self):
        self.commands.append(("neutral", 0))

    def set_brake(self, command):
        self.commands.append(("brake", command))

    def set_reverse(self):
        raise AssertionError("ADAS nao deve armar marcha re")


class ControllerTests(unittest.TestCase):
    def make_car(self):
        def init_sensors(car):
            car.atuador = FakeActuator()
            car.bz = Mock()
            car.bz.beep.return_value = True
            car.odometer = Mock()
            car.imu = Mock()
            car.imu.get_gyro.return_value = (0, 0, 0)

        with patch.object(car_module.Car, "init_sensors", init_sensors), patch.object(
            car_module.Car, "get_car_color", return_value=None,
        ):
            car = car_module.Car({
                "initial_position": [0, 0, 0], "save": False,
                "adas": {"coast_deceleration": 0.2},
            })
        car.emergencia = False
        car.velocity_valid = True
        car.now = 0
        car.get_time = lambda: car.now
        return car

    def apply(self, car, speed, distance):
        car.v = car.v_raw = speed
        car.t = car.now
        with contextlib.redirect_stdout(io.StringIO()):
            return car.set_adas_vel(1.0, distance, True)

    def test_coast_is_neutral_even_with_accumulated_integral(self):
        car = self.make_car()
        car.vel_error_integral = 10
        decision = self.apply(car, 0.2, 0.42)
        self.assertEqual(decision.mode, Mode.COAST)
        self.assertEqual(car.u, 0)
        self.assertEqual(car.vel_error_integral, 0)
        self.assertEqual(car.atuador.commands[-1], ("neutral", 0))
        car.bz.beep.assert_called_once()

    def test_braking_reuses_pi_with_negative_output_and_zero_reference(self):
        car = self.make_car()
        decision = self.apply(car, 1, 1)
        self.assertEqual(decision.mode, Mode.BRAKE)
        self.assertEqual(car.vref, 0)
        self.assertLess(car.u, 0)
        self.assertGreater(car.atuador.commands[-1][1], 0)
        self.assertEqual(car.gear, servos.Gear.FORWARD)

    def test_raw_encoder_stops_reverse_command_before_filtered_speed(self):
        car = self.make_car()
        self.apply(car, 1, 1)
        car.v_raw = 0
        car.v = 0.2
        car.now = 0.02
        with contextlib.redirect_stdout(io.StringIO()):
            car.set_adas_vel(1, 0.3, True)
        self.assertEqual(car.atuador.commands[-1], ("neutral", 0))
        self.assertEqual(car.u, 0)

    def test_speed_failure_and_emergency_never_apply_reverse(self):
        car = self.make_car()
        car.velocity_valid = False
        self.apply(car, 1, 1)
        self.assertEqual(car.atuador.commands[-1], ("neutral", 0))
        car.velocity_valid = True
        car.emergencia = True
        self.apply(car, 1, 1)
        self.assertEqual(car.atuador.commands[-1], ("neutral", 0))

    def test_coast_and_brake_do_not_alter_normal_pi(self):
        car = self.make_car()
        car.v = 0.4
        car.set_vel(1)
        expected = min(1, 0.17 + 0.6 + 0.8 * (0.6 * car.dt))
        self.assertAlmostEqual(car.u, expected)

    def test_csv_records_decision_and_command_in_current_sample(self):
        car = self.make_car()
        car.save_traj()
        self.apply(car, 1, 1)
        self.assertEqual(len(car.traj), 1)
        self.assertEqual(car.traj[0]["adas_mode"], int(Mode.BRAKE))
        self.assertEqual(car.traj[0]["u"], car.u)
        self.assertEqual(car.traj[0]["obstacle_distance"], 1)

    def test_brake_limit_and_integrator_are_reset_before_resuming(self):
        car = self.make_car()
        car.adas = adas.BrakingADAS(adas.BrakingConfig(max_brake=0.4))
        for _ in range(20):
            self.apply(car, 1, 1)
            car.now += car.dt
            self.assertGreaterEqual(car.u, -0.4)
            self.assertLessEqual(car.u, 0)
        self.assertEqual(car.vel_error_integral, 0)
        car.now = 1
        self.assertEqual(self.apply(car, 0, 4).mode, Mode.HOLD)
        car.now = 1.6
        self.assertEqual(self.apply(car, 0, 4).mode, Mode.CRUISE)
        self.assertGreater(car.u, 0)
        self.assertEqual(car.atuador.commands[-1][0], "drive")

    def test_sensor_reads_preserve_signed_raw_velocity_and_validity(self):
        car = self.make_car()
        car.odometer.get_vel.return_value = (-0.2, True)
        car.get_vel()
        self.assertEqual(car.v_raw, -0.2)
        car.odometer.get_vel.return_value = (math.nan, True)
        car.get_vel()
        self.assertFalse(car.velocity_valid)

    def test_stop_against_stationary_obstacle_in_model(self):
        for distance, expected_mode in ((3.3, Mode.COAST), (1.0, Mode.BRAKE)):
            with self.subTest(distance=distance):
                car = self.make_car()
                speed = 1.0
                first_mode = None
                for _ in range(1000):
                    decision = self.apply(car, speed, distance)
                    if first_mode is None:
                        first_mode = decision.mode
                    brake = max(0, -car.u)
                    # Modelo de teste explicito, nao calibracao do carrinho.
                    deceleration = 0.2 + 2 * brake
                    next_speed = max(0, speed - deceleration * car.dt)
                    distance -= (speed + next_speed) / 2 * car.dt
                    speed = next_speed
                    car.now += car.dt
                    if decision.mode == Mode.HOLD:
                        break
                self.assertEqual(first_mode, expected_mode)
                self.assertLessEqual(speed, car.adas.config.stop_speed)
                self.assertGreaterEqual(distance, car.adas.config.clearance)
                self.assertEqual(car.u, 0)


class ActuatorTests(unittest.TestCase):
    def make_actuator(self):
        actuator = servos.Servos.__new__(servos.Servos)
        actuator.lock = threading.Lock()
        actuator.throttle_lock = threading.Lock()
        actuator.gear = servos.Gear.FORWARD
        actuator.max_throttle = np.deg2rad(20)
        actuator.min_pwm_throttle = servos.ZERO_THROTTLE_ANGLE - actuator.max_throttle
        actuator.max_pwm_throttle = servos.ZERO_THROTTLE_ANGLE + actuator.max_throttle
        actuator.trim_throttle = 0
        actuator.dt = 0.03
        actuator.st_pwm = actuator.pan_pwm = 0
        actuator.stop = threading.Event()
        actuator._set_servo = Mock()
        return actuator

    def test_brake_pwm_is_immediate_without_wait_or_reverse_arming(self):
        actuator = self.make_actuator()
        with patch.object(servos.time, "sleep", side_effect=AssertionError("Freio nao pode esperar")):
            actuator.set_brake(0.5)
        self.assertLess(actuator.pwm, servos.ZERO_THROTTLE_ANGLE)
        self.assertEqual(actuator.gear, servos.Gear.FORWARD)
        self.assertTrue(actuator.braking)
        actuator._set_servo.assert_called_once()
        actuator.set_neutral()
        self.assertEqual(actuator.pwm, servos.ZERO_THROTTLE_ANGLE)
        self.assertFalse(actuator.braking)
        self.assertEqual(actuator.dth_pwm, 0)

    def test_periodic_actuator_preserves_brake_and_then_neutral(self):
        actuator = self.make_actuator()
        actuator.set_brake(0.5)
        # Um ciclo real da thread, sem atrasar o teste.
        with patch.object(actuator.stop, "wait", side_effect=lambda delay: actuator.stop.set()):
            actuator.actuator()
        self.assertLess(actuator.pwm, servos.ZERO_THROTTLE_ANGLE)
        actuator.set_neutral()
        actuator.stop.clear()
        with patch.object(actuator.stop, "wait", side_effect=lambda delay: actuator.stop.set()):
            actuator.actuator()
        self.assertEqual(actuator.pwm, servos.ZERO_THROTTLE_ANGLE)


if __name__ == "__main__":
    unittest.main()
