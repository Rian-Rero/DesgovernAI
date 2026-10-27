"""Integracao RC/AUTO preservando o PI local, sem acessar hardware."""
import importlib.util
from enum import Enum
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np


class Gear(Enum):
    FORWARD = 'forward'
    REVERSE = 'reverse'


def load_car():
    root = Path(__file__).resolve().parents[1] / 'fva_car'
    package = ModuleType('_rc_merge_test')
    package.__path__ = [str(root)]
    modules = {package.__name__: package}
    for driver in ('encoder', 'servos', 'buzzer', 'ultrasonic', 'imu'):
        modules[package.__name__ + '.' + driver] = ModuleType(driver)
    modules[package.__name__ + '.servos'].Gear = Gear
    name = package.__name__ + '.car'
    spec = importlib.util.spec_from_file_location(name, root / 'car.py')
    module = importlib.util.module_from_spec(spec)
    modules[name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class RCInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_car()

    def setUp(self):
        self.car = self.module.Car.__new__(self.module.Car)
        self.car.emergencia = False
        self.car.encoder_valid = True
        self.car.sel_direcao = self.car.sel_tracao = 1
        self.car.rc_direcao = self.car.rc_acelerador = .25
        self.car.gear = Gear.FORWARD
        self.car.v = self.car.vref = self.car.u = self.car.st = 0.
        self.car.dt = self.car.sample_rate = .02
        self.car.vel_error_integral = 0.
        self.car.atuador = Mock()

    def test_rc_traction_replaces_reference_but_still_uses_local_pi(self):
        self.car.sel_tracao = 0
        self.car.set_vel(1.8)
        expected = .25 * self.module.CAR['VELMAX']
        self.assertAlmostEqual(self.car.vref, expected)
        self.assertGreater(self.car.u, 0.)
        self.car.atuador.set_throttle.assert_called_once_with(self.car.u)
        self.car.atuador.set_torque.assert_not_called()

    def test_auto_traction_keeps_received_reference(self):
        self.car.set_vel(.3)
        self.assertAlmostEqual(self.car.vref, .3)

    def test_filtered_steering_is_logged_separately_from_requested_steering(self):
        self.car.atuador.set_steer.return_value = .03
        self.car.set_steer(.1)
        self.assertAlmostEqual(self.car.st, .1)
        self.assertAlmostEqual(self.car.st_filtered, .03)

    def test_optional_imu_telemetry_skips_i2c_without_changing_model_states(self):
        self.car.parameters = {'imu_telemetry': False}
        self.car.odometer = Mock()
        self.car.odometer.get_vel.return_value = (.4, True)
        self.car.imu = Mock()
        self.car.v_filt = Mock()
        self.car.v_filt.filter.side_effect = lambda value: value
        self.car.w_filt = Mock()
        self.car.w_filt.filter.side_effect = lambda value: value
        self.car.a_filt = Mock()
        self.car.a_filt.filter.side_effect = lambda value: value
        self.car.st = .1
        self.car.v_ant, self.car.v = .2, .4
        velocity, yaw_rate = self.car.get_vel()
        acceleration = self.car.get_accel()
        self.car.imu.get_gyro.assert_not_called()
        self.car.imu.get_accel.assert_not_called()
        self.assertAlmostEqual(velocity, .4)
        self.assertAlmostEqual(yaw_rate, .4/self.module.CAR['L'] * np.tan(.1))
        self.assertAlmostEqual(acceleration, (.4-.2)/.02)
        self.assertTrue(np.isnan(self.car.w_imu))
        self.assertTrue(np.isnan(self.car.a_x))

    def test_invalid_encoder_cuts_throttle_and_integral_without_requesting_reverse(self):
        self.car.encoder_valid = False
        self.car.sel_tracao = 0
        self.car.rc_acelerador = -1.
        self.car.vel_error_integral = 5.
        self.car.set_reverse = Mock()
        self.car.set_vel(1.)
        self.assertEqual(self.car.vref, 0.)
        self.assertEqual(self.car.vel_error_integral, 0.)
        self.car.atuador.set_throttle.assert_called_once_with(0.)
        self.car.set_reverse.assert_not_called()

    def test_invalid_encoder_blocks_direct_throttle_and_rc_steering(self):
        self.car.encoder_valid = False
        self.car.sel_direcao = 0
        self.car.rc_direcao = 1.
        self.car.set_u(1.)
        self.car.set_steer(.2)
        self.car.atuador.set_throttle.assert_called_once_with(0.)
        self.car.atuador.set_steer.assert_called_once_with(0.)

    def test_encoder_timeout_during_gear_shift_cuts_the_new_throttle(self):
        self.car.gear = Gear.REVERSE
        self.car.odometer = Mock()
        self.car.odometer.control_valid.side_effect = [True, False]
        self.car.set_forward = Mock(side_effect=lambda: setattr(self.car, 'gear', Gear.FORWARD))
        self.car.vel_error_integral = 2.
        self.car.set_vel(.3)
        self.car.set_forward.assert_called_once()
        self.car.atuador.set_throttle.assert_called_once_with(0.)
        self.assertEqual(self.car.vref, 0.)
        self.assertEqual(self.car.vel_error_integral, 0.)
        self.assertFalse(self.car.encoder_valid)

    def test_independent_selectors_control_only_their_axis(self):
        self.car.sel_direcao = 0
        self.car.set_steer(-.2)
        self.assertAlmostEqual(self.car.st, .25 * self.module.CAR['STEERMAX'])
        self.car.set_vel(.3)
        self.assertAlmostEqual(self.car.vref, .3)
        self.car.sel_direcao = 1
        self.car.sel_tracao = 0
        self.car.set_steer(-.2)
        self.assertAlmostEqual(self.car.st, -.2)
        self.car.set_vel(.3)
        self.assertAlmostEqual(self.car.vref, .5)

    def test_emergency_has_priority_over_radio_inputs(self):
        self.car.sel_direcao = self.car.sel_tracao = 0
        self.car.rc_direcao = self.car.rc_acelerador = 1.
        self.car.emergencia = True
        self.car.set_steer(.2)
        self.car.set_vel(1.)
        self.assertEqual(self.car.st, 0.)
        self.assertEqual(self.car.vref, 0.)
        self.car.atuador.set_throttle.assert_called_once_with(0.)

    def test_new_encoder_modes_are_read_and_pi_resets_when_traction_changes(self):
        self.car.sel_tracao = 0
        self.car.vel_error_integral = 4.
        self.car.odometer = Mock()
        self.car.odometer.get_rc.return_value = (-.2, .7)
        self.car.odometer.get_control_mode.return_value = (0, 1)
        self.car.odometer.control_valid.return_value = False
        self.car.get_vel = Mock(return_value=(0., 0.))
        self.car.get_accel = Mock(return_value=0.)
        self.car.get_yaw = Mock(return_value=0.)
        self.car.get_pos = Mock(return_value=np.zeros(2))
        self.car.get_time = Mock(return_value=1.)
        self.car.tinit = 0.
        self.car.get_states()
        self.assertEqual((self.car.rc_direcao, self.car.rc_acelerador), (-.2, .7))
        self.assertEqual((self.car.sel_direcao, self.car.sel_tracao), (0, 1))
        self.assertEqual(self.car.vel_error_integral, 0.)
        self.assertFalse(self.car.encoder_valid)


if __name__ == '__main__':
    unittest.main()
