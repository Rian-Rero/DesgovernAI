"""Integracao RC/AUTO preservando o PI local, sem acessar hardware."""
import importlib.util
from enum import Enum
from pathlib import Path
import sys
import threading
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


def load_encoder():
    # Nenhuma porta serial e aberta; apenas a classe e carregada.
    serial = ModuleType('serial')
    tools = ModuleType('serial.tools')
    ports = ModuleType('serial.tools.list_ports')
    serial.tools = tools
    tools.list_ports = ports
    path = Path(__file__).resolve().parents[1] / 'fva_car' / 'encoder.py'
    spec = importlib.util.spec_from_file_location('_rc_encoder_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'serial': serial, 'serial.tools': tools,
                                  'serial.tools.list_ports': ports}):
        spec.loader.exec_module(module)
    return module


class RCInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_car()

    def setUp(self):
        self.car = self.module.Car.__new__(self.module.Car)
        self.car.emergencia = False
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
        self.car.odometer.get_control_inputs.return_value = (-.2, .7, 0, 1)
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

    def test_rc_neutral_cuts_throttle_and_clears_previous_pi_integral(self):
        self.car.sel_tracao = 0
        self.car.rc_acelerador = 0.
        self.car.vel_error_integral = 3.
        self.car.set_vel(1.)
        self.assertEqual(self.car.vref, 0.)
        self.assertEqual(self.car.vel_error_integral, 0.)
        self.car.atuador.set_throttle.assert_called_once_with(0.)
        self.car.atuador.set_reverse.assert_not_called()

    def test_startup_without_rc_pulses_never_requests_reverse(self):
        module = load_encoder()
        encoder = module.Encoder.__new__(module.Encoder)
        encoder.get_data = Mock(return_value=(0., 0, 0, 0, 0, True))
        (self.car.rc_direcao, self.car.rc_acelerador,
         self.car.sel_direcao, self.car.sel_tracao) = encoder.get_control_inputs()
        self.car.set_vel(0.)
        self.car.set_steer(0.)
        self.assertEqual(self.car.gear, Gear.FORWARD)
        self.assertEqual(self.car.st, 0.)
        self.car.atuador.set_reverse.assert_not_called()
        self.car.atuador.set_throttle.assert_called_once_with(0.)


class EncoderRCValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_encoder()

    def setUp(self):
        self.encoder = self.module.Encoder.__new__(self.module.Encoder)
        self.encoder.lock = threading.Lock()
        self.encoder.vel = 0.
        self.encoder.rc_direcao = 1820
        self.encoder.rc_acelerador = 1780
        self.encoder.sel_direcao = self.encoder.sel_tracao = 1
        self.encoder.valid = True
        self.encoder.last_measurement = 10.

    def test_zero_invalid_and_nonfinite_pulses_mean_neutral(self):
        for value in (0, -1, 799, 2201, float('nan'), float('inf')):
            with self.subTest(value=value):
                self.assertEqual(self.encoder.normalize_rc(value, 1210, 1450, 1780), 0.)

    def test_real_minimum_center_and_maximum_still_normalize_correctly(self):
        for pulse, expected in ((1210, -1.), (1450, 0.), (1780, 1.)):
            self.assertAlmostEqual(self.encoder.normalize_rc(pulse, 1210, 1450, 1780), expected)

    def test_fresh_packet_preserves_real_commands_and_modes(self):
        with patch.object(self.module.time, 'monotonic', return_value=10.1):
            self.assertEqual(self.encoder.get_control_inputs(), (1., 1., 1, 1))
            self.assertEqual(self.encoder.get_rc(), (1., 1.))
            self.assertEqual(self.encoder.get_control_mode(), (1, 1))

    def test_stale_missing_or_invalid_packets_force_neutral_manual(self):
        for last, valid in ((9., True), (0., True), (10., False)):
            self.encoder.last_measurement = last
            self.encoder.valid = valid
            with self.subTest(last=last, valid=valid), \
                    patch.object(self.module.time, 'monotonic', return_value=10.1):
                self.assertEqual(self.encoder.get_control_inputs(), (0., 0., 0, 0))


if __name__ == '__main__':
    unittest.main()
