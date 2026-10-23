"""Protocolos explicitos RC/RPM e ausencia segura de radio, sem serial real."""
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


REAL = Path(__file__).resolve().parents[1]


class FakeSerialException(Exception):
    pass


class FakeSerial:
    def __init__(self):
        self.lines = []
        self.owner = None
        self.is_open = True
        self.reset_input_buffer = Mock()

    def readline(self):
        if self.lines:
            item = self.lines.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        # Encerra o laco de leitura sincronamente, sem criar thread.
        self.owner.running = False
        return b''

    def close(self):
        self.is_open = False


class EncoderProtocolTests(unittest.TestCase):
    def setUp(self):
        serial = ModuleType('serial')
        serial.SerialException = FakeSerialException
        serial.Serial = Mock()
        ports = ModuleType('serial.tools.list_ports')
        ports.comports = Mock(return_value=[SimpleNamespace(
            description='Arduino Nano de teste', hwid='USB FAKE', device='FAKE_SERIAL')])
        tools = ModuleType('serial.tools')
        tools.list_ports = ports
        serial.tools = tools
        replacements = {'serial': serial, 'serial.tools': tools,
                        'serial.tools.list_ports': ports}
        name = '_encoder_protocol_under_test'
        spec = importlib.util.spec_from_file_location(name, REAL/'fva_car/encoder.py')
        self.module = importlib.util.module_from_spec(spec)
        replacements[name] = self.module
        with patch.dict(sys.modules, replacements):
            spec.loader.exec_module(self.module)
        self.serial_factory = serial.Serial
        self.now = 10.
        for guard in (patch.object(self.module.time, 'monotonic', side_effect=lambda: self.now),
                      patch.object(self.module.time, 'sleep')):
            guard.start()
            self.addCleanup(guard.stop)
        self.instances = []

    def make_encoder(self, protocol=None):
        connection = FakeSerial()
        self.serial_factory.return_value = connection
        thread = Mock()
        thread.is_alive.return_value = False
        with patch.object(self.module.threading, 'Thread', return_value=thread):
            encoder = (self.module.Encoder() if protocol is None
                       else self.module.Encoder(protocol=protocol))
        connection.owner = encoder
        self.instances.append(encoder)
        self.addCleanup(encoder.close)
        return encoder

    def feed(self, encoder, *lines):
        encoder.ser.lines = [line.encode('utf-8') if isinstance(line, str) else line
                             for line in lines]
        encoder.running = True
        encoder._serial_loop()

    def expected_speed(self, rpm):
        return self.module.RAIO_RODA * math.pi / 30. * rpm / self.module.REDUCAO_EIXO

    def test_default_protocol_is_rc_and_never_autodetects_legacy_rpm(self):
        encoder = self.make_encoder()
        self.feed(encoder, '0.00\r\n', '780.0\r\n', '-780.0\r\n')
        self.assertFalse(encoder.get_vel()[1])
        self.assertFalse(encoder.control_valid())
        self.assertEqual(encoder.get_rc(), (0., 0.))
        self.assertFalse(encoder.get_data()[-1])

    def test_rpm_requires_explicit_mode_and_converts_both_directions(self):
        encoder = self.make_encoder('rpm')
        for rpm in (0., 780., -780., 39.):
            with self.subTest(rpm=rpm):
                self.now += .1
                self.feed(encoder, f'{rpm:.2f}\r\n')
                velocity, valid = encoder.get_vel()
                self.assertTrue(valid)
                self.assertAlmostEqual(velocity, self.expected_speed(rpm))
                self.assertEqual(encoder.get_control_mode(), (1, 1))
                self.assertEqual(encoder.get_rc(), (0., 0.))
                self.assertTrue(encoder.control_valid())

    def test_rpm_zero_frames_remain_fresh_and_never_request_reverse(self):
        encoder = self.make_encoder('rpm')
        for _ in range(4):
            self.now += .2
            self.feed(encoder, '0.00\r\n')
            self.assertEqual(encoder.get_vel(), (0., True))
            self.assertEqual(encoder.get_rc(), (0., 0.))
            self.assertEqual(encoder.get_control_mode(), (1, 1))
            self.assertEqual(len(encoder.get_data()), 6)
            self.assertEqual(encoder.get_data()[3:], (1, 1, True))
            self.assertTrue(encoder.control_valid())

    def test_rpm_mode_never_accepts_a_five_field_rc_frame(self):
        encoder = self.make_encoder('rpm')
        self.feed(encoder, '780,1490,1450,1,1\n')
        self.assertFalse(encoder.get_vel()[1])
        self.assertFalse(encoder.control_valid())
        self.assertEqual(encoder.get_rc(), (0., 0.))

    def test_invalid_protocol_fails_before_opening_serial(self):
        with self.assertRaises(ValueError):
            self.module.Encoder(protocol='auto')
        self.serial_factory.assert_not_called()

    def test_rpm_nonfinite_corrupt_and_multifield_lines_are_not_samples(self):
        for line in ('nan\n', 'inf\n', '-inf\n', 'text\n', '1,2\n', b'\xff\r\n'):
            with self.subTest(line=line):
                encoder = self.make_encoder('rpm')
                self.feed(encoder, line)
                self.assertFalse(encoder.get_vel()[1])
                self.assertFalse(encoder.control_valid())
                self.assertEqual(encoder.get_rc(), (0., 0.))

    def test_rc_startup_and_timeout_return_neutral_commands(self):
        encoder = self.make_encoder('rc')
        self.assertEqual(encoder.get_rc(), (0., 0.))
        self.assertFalse(encoder.control_valid())
        self.feed(encoder, '780,1820,1780,0,0\n')
        self.assertEqual(encoder.get_rc(), (1., 1.))
        self.assertTrue(encoder.control_valid())
        self.now += .31
        self.assertEqual(encoder.get_rc(), (0., 0.))
        self.assertFalse(encoder.control_valid())
        self.assertFalse(encoder.get_vel()[1])
        self.assertFalse(encoder.get_data()[-1])

    def test_rpm_timeout_invalidates_data_and_control(self):
        encoder = self.make_encoder('rpm')
        self.feed(encoder, '780\n')
        self.now += .29
        self.assertTrue(encoder.control_valid())
        self.now += .02
        self.assertFalse(encoder.control_valid())
        self.assertFalse(encoder.get_data()[-1])
        self.assertEqual(encoder.get_rc(), (0., 0.))

    def test_zero_and_outside_receiver_pulses_normalize_to_neutral(self):
        encoder = self.make_encoder('rc')
        for pulse in (0, -1, 799, 2201, 99999):
            with self.subTest(pulse=pulse):
                self.assertEqual(encoder.normalize_rc(pulse,1170,1490,1820), 0.)
                self.assertEqual(encoder.normalize_rc(pulse,1210,1450,1780), 0.)

    def test_rc_normalization_preserves_min_center_max_and_deadzone(self):
        encoder = self.make_encoder('rc')
        for value, expected in ((1170,-1.),(1490,0.),(1820,1.),(1491,0.)):
            self.assertAlmostEqual(encoder.normalize_rc(value,1170,1490,1820),expected)
        for value, expected in ((1210,-1.),(1450,0.),(1780,1.),(1451,0.)):
            self.assertAlmostEqual(encoder.normalize_rc(value,1210,1450,1780),expected)

    def test_serial_validity_is_distinct_from_invalid_selected_rc_pulses(self):
        encoder = self.make_encoder('rc')
        self.feed(encoder, '780,0,0,0,0\n')
        self.assertTrue(encoder.get_vel()[1])
        self.assertTrue(encoder.get_data()[-1])
        self.assertFalse(encoder.control_valid())
        self.assertEqual(encoder.get_rc(), (0., 0.))

    def test_auto_axes_allow_absent_rc_pulses(self):
        encoder = self.make_encoder('rc')
        self.feed(encoder, '780,0,0,1,1\n')
        self.assertEqual(encoder.get_control_mode(), (1,1))
        self.assertTrue(encoder.control_valid())
        self.assertEqual(encoder.get_rc(), (0.,0.))
        self.assertTrue(encoder.get_vel()[1])

    def test_mixed_rc_auto_requires_only_the_selected_radio_axis(self):
        cases = [
            ('0,0,1780,1,0\n', True, (0.,1.)),
            ('0,1170,0,0,1\n', True, (-1.,0.)),
            ('0,0,1780,0,1\n', False, (0.,1.)),
            ('0,1820,0,1,0\n', False, (1.,0.)),
        ]
        for line, control_valid, rc in cases:
            with self.subTest(line=line):
                encoder = self.make_encoder('rc')
                self.feed(encoder,line)
                self.assertEqual(encoder.control_valid(),control_valid)
                self.assertEqual(encoder.get_rc(),rc)

    def test_selected_rc_axis_outside_physical_pulse_range_is_invalid(self):
        for direction, throttle, modes in ((799,1450,'0,1'),(2201,1450,'0,1'),
                                           (1490,799,'1,0'),(1490,2201,'1,0')):
            with self.subTest(direction=direction,throttle=throttle,modes=modes):
                encoder = self.make_encoder('rc')
                self.feed(encoder,f'0,{direction},{throttle},{modes}\n')
                self.assertTrue(encoder.get_data()[-1])
                self.assertFalse(encoder.control_valid())

    def test_get_data_preserves_six_raw_fields_and_serial_validity(self):
        encoder = self.make_encoder('rc')
        self.feed(encoder,'780,1490,1780,0,1\n')
        values = encoder.get_data()
        self.assertEqual(len(values),6)
        self.assertAlmostEqual(values[0],self.expected_speed(780.))
        self.assertEqual(values[1:],(1490,1780,0,1,True))
        self.now += .31
        self.assertEqual(encoder.get_data()[1:],(1490,1780,0,1,False))

    def test_corrupt_lines_do_not_refresh_an_old_sample(self):
        encoder = self.make_encoder('rc')
        self.feed(encoder,'780,1490,1450,0,0\n')
        self.now += .31
        self.feed(encoder,'garbage\n','0.00\n','0,1490,1450,2,0\n')
        self.assertFalse(encoder.get_data()[-1])
        self.assertFalse(encoder.control_valid())
        self.assertEqual(encoder.get_rc(),(0.,0.))

    def test_legacy_zero_rpm_reaches_car_as_auto_without_requesting_reverse(self):
        helper_spec = importlib.util.spec_from_file_location(
            '_encoder_rc_car_helpers', REAL/'tests/test_rc_input.py')
        helpers = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helpers)
        module = helpers.load_car()
        encoder = self.make_encoder('rpm')
        self.feed(encoder, '0.00\r\n')
        car = module.Car.__new__(module.Car)
        car.odometer = encoder
        car.emergencia = False
        car.sel_direcao = car.sel_tracao = 0
        car.rc_direcao = car.rc_acelerador = -1.
        car.gear = helpers.Gear.FORWARD
        car.v = car.vref = car.u = car.st = 0.
        car.dt = car.sample_rate = .02
        car.vel_error_integral = 0.
        car.atuador = Mock()
        car.get_vel = Mock(return_value=(0., 0.))
        car.get_accel = Mock(return_value=0.)
        car.get_yaw = Mock(return_value=0.)
        car.get_pos = Mock(return_value=np.zeros(2))
        car.get_time = Mock(return_value=self.now)
        car.tinit = 0.
        car.get_states()
        self.assertTrue(car.encoder_valid)
        self.assertEqual((car.sel_direcao, car.sel_tracao), (1, 1))
        self.assertEqual((car.rc_direcao, car.rc_acelerador), (0., 0.))
        car.set_reverse = Mock()
        car.set_vel(.3)
        self.assertAlmostEqual(car.vref, .3)
        self.assertGreater(car.u, 0.)
        car.set_reverse.assert_not_called()

    def test_serial_failure_invalidates_even_a_recent_sample(self):
        encoder = self.make_encoder('rc')
        self.feed(encoder,'0,1490,1450,0,0\n',FakeSerialException('USB disconnected'))
        self.assertFalse(encoder.get_vel()[1])
        self.assertFalse(encoder.control_valid())
        self.assertEqual(encoder.get_rc(),(0.,0.))


if __name__ == '__main__':
    unittest.main()
