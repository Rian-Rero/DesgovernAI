"""IMU: driver (imu.py), guinada pelo giroscopio (yaw_rate.py), Car, Stanley e resumo da volta.

A IMU deste carro ja queimou: todo caminho de falha tem que deixar o Car no
modelo da direcao, nunca derrubar a missao."""
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402
from fva_car import imu as imu_module  # noqa: E402
from fva_car.lateral import StanleyController  # noqa: E402
from fva_car.yaw_rate import GyroYaw, start_gyro  # noqa: E402

DEG = math.pi/180


def rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class FakeSensor:
    """MPU montado com a rotacao `mount` (carro -> sensor), com ruido e vies."""
    def __init__(self, mount=np.eye(3), bias=(.5, -.3, .8), noise=.05, seed=0):
        self.mount = np.asarray(mount, float)
        self.bias = np.asarray(bias, float)
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.yaw_rate = 0.     # rad/s, esquerda + (no carro)
        self.fail = 0          # proximas leituras com erro de I2C
        self.frozen = None

    def read_motion(self):
        if self.fail:
            self.fail -= 1
            raise OSError(121, 'Remote I/O error')
        if self.frozen is not None:
            return self.frozen
        accel = self.mount @ np.array([0., 0., 1.]) + self.rng.normal(0, .01, 3)
        gyro = self.mount @ np.array([0., 0., self.yaw_rate/DEG]) + self.bias + self.rng.normal(0, self.noise, 3)
        return tuple(accel), tuple(gyro)


def calibrated(sensor, **kwargs):
    return GyroYaw(sensor, **kwargs).calibrate(sleep=lambda _: None)


class CalibrationTests(unittest.TestCase):
    def test_any_mounting_gives_left_positive_yaw(self):
        for mount in (np.eye(3), rotation('x', math.pi), rotation('x', math.pi/2), rotation('y', -math.pi/2)):
            sensor = FakeSensor(mount)
            gyro = calibrated(sensor)
            sensor.yaw_rate = .8
            rates = [gyro.read(1., .8, .02) for _ in range(20)]
            self.assertAlmostEqual(np.mean(rates), .8, delta=.01)

    def test_bad_sensors_fail_at_start(self):
        stuck = FakeSensor()
        stuck.frozen = ((0., 0., 1.), (0., 0., 0.))
        dead = Mock(read_motion=Mock(return_value=((0., 0., 0.), (0., 0., 0.))))
        far = Mock(read_motion=Mock(side_effect=lambda: ((0., 0., 2.5 + np.random.rand()*1e-3), (0., 0., 0.))))
        biased = FakeSensor(bias=(0., 0., 40.))
        moving = FakeSensor(noise=5.)
        for sensor, text in ((stuck, 'iguais'), (dead, 'iguais'), (far, '2.50 g'), (biased, 'vies'),
                             (moving, 'oscila')):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, text):
                calibrated(sensor)


class ReadTests(unittest.TestCase):
    def test_i2c_glitch_uses_the_model_for_that_step_only(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        sensor.fail = 3
        self.assertTrue(all(math.isnan(gyro.read(1., 0., .02)) for _ in range(3)))
        self.assertTrue(math.isfinite(gyro.read(1., 0., .02)))
        self.assertTrue(gyro.enabled)
        self.assertEqual(gyro.failures, 3)

    def test_repeated_failures_or_frozen_sensor_disable_it(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        sensor.fail = 100
        with redirect_stdout(io.StringIO()) as out:
            for _ in range(12):
                gyro.read(1., 0., .02)
        self.assertFalse(gyro.enabled)
        self.assertIn('IMU desligada', out.getvalue())
        self.assertTrue(math.isnan(gyro.read(1., 0., .02)))
        frozen = FakeSensor()
        gyro = calibrated(frozen)
        frozen.frozen = frozen.read_motion()
        with redirect_stdout(io.StringIO()):
            for _ in range(60):
                gyro.read(1., 0., .02)
        self.assertFalse(gyro.enabled)
        self.assertIn('travado', gyro.reason)

    def test_saturated_reading_is_skipped(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor, full_scale_dps=250.)
        sensor.yaw_rate = 300*DEG
        self.assertTrue(math.isnan(gyro.read(3., 5., .02)))
        self.assertEqual(gyro.saturated, 1)
        self.assertTrue(gyro.enabled)

    def test_bias_drift_is_relearned_only_when_stopped(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        sensor.bias = sensor.bias + np.array([0., 0., 1.])  # +1 grau/s com a temperatura
        moving = np.mean([gyro.read(1., 0., .02) for _ in range(200)])
        self.assertAlmostEqual(moving, DEG, delta=.002)       # andando: nao confunde curva com vies
        for _ in range(1000):                                  # 20 s parado
            gyro.read(0., 0., .02)
        self.assertAlmostEqual(np.mean([gyro.read(0., 0., .02) for _ in range(50)]), 0., delta=.002)

    def test_measures_steering_response_per_side(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        for command, real in ((.6, .88*.6), (-.6, .95*-.6)):
            sensor.yaw_rate = real
            for _ in range(100):
                gyro.read(1., command, .02)
        response = gyro.summary()['resposta_direcao']
        self.assertAlmostEqual(response['esquerda'], .88, delta=.02)
        self.assertAlmostEqual(response['direita'], .95, delta=.02)

    def test_gyro_that_does_not_follow_the_turns_is_disabled(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        sensor.yaw_rate = 0.  # eixo morto: o carro vira e o giroscopio nao ve
        with redirect_stdout(io.StringIO()):
            for _ in range(200):
                gyro.read(1., .6, .02)
        self.assertFalse(gyro.enabled)
        self.assertIn('nao acompanha', gyro.reason)


class StartTests(unittest.TestCase):
    def test_missing_or_burned_chip_leaves_the_car_on_the_model(self):
        logs = []
        for error in (OSError(121, 'Remote I/O error'), RuntimeError('IMU nao reconhecida (WHO_AM_I=0x00)'),
                      ImportError('Sem smbus')):
            def factory(**_):
                raise error
            self.assertEqual(start_gyro({}, factory, logs.append)[:2], (None, None))
        self.assertTrue(all('indisponivel' in text for text in logs))
        self.assertEqual(start_gyro({'imu': False}, factory, logs.append), (None, None, 'desligada na configuracao'))

    def test_failed_calibration_closes_the_bus(self):
        sensor = FakeSensor()
        sensor.frozen = ((0., 0., 1.), (0., 0., 0.))
        sensor.close = Mock()
        sensor_factory = Mock(return_value=sensor)
        from fva_car import yaw_rate
        original = yaw_rate.GyroYaw.calibrate
        yaw_rate.GyroYaw.calibrate = lambda self: original(self, sleep=lambda _: None)
        try:
            result = start_gyro({'imu_bus': 3}, sensor_factory, lambda _: None)
        finally:
            yaw_rate.GyroYaw.calibrate = original
        self.assertEqual(result[:2], (None, None))
        sensor.close.assert_called_once()
        self.assertEqual(sensor_factory.call_args.kwargs['bus_id'], 3)


class FakeBus:
    def __init__(self, who=0x71):
        self.who, self.writes = who, []

    def write_byte_data(self, address, register, value):
        self.writes.append((register, value))

    def read_byte_data(self, address, register):
        if register == imu_module.IMU.WHO_AM_I:
            return self.who
        raise OSError('sem magnetometro')

    def read_i2c_block_data(self, address, register, length):
        values = [16384, -16384, 0, 1234, 655, -655, 0]  # 1 g, -1 g, 0 | temperatura | ~10, -10 graus/s, 0
        data = []
        for value in values:
            value &= 0xFFFF
            data += [value >> 8, value & 0xFF]
        return data


class DriverTests(unittest.TestCase):
    def make(self, who):
        bus = FakeBus(who)
        original = imu_module.smbus
        imu_module.smbus = SimpleNamespace(SMBus=lambda _: bus)
        try:
            with redirect_stdout(io.StringIO()):
                return imu_module.IMU(auto_calibrate=False, gyro_range_dps=500), bus
        finally:
            imu_module.smbus = original

    def test_unknown_chip_is_refused(self):
        for who in (0x00, 0xFF, 0x12):
            with self.subTest(who=who), self.assertRaisesRegex(RuntimeError, 'WHO_AM_I'):
                self.make(who)

    def test_block_read_and_range(self):
        sensor, bus = self.make(0x71)
        self.assertEqual(sensor.chip, 'MPU-9250')
        self.assertIn((imu_module.IMU.GYRO_CONFIG, 0x08), bus.writes)   # +-500 graus/s
        accel, gyro = sensor.read_motion()
        self.assertEqual(accel, (1., -1., 0.))
        self.assertAlmostEqual(gyro[0], 655/65.536)    # 65,5 LSB por grau/s em +-500
        self.assertAlmostEqual(gyro[1], -655/65.536)

    def test_without_smbus_the_error_comes_only_when_creating(self):
        original = imu_module.smbus
        imu_module.smbus = None
        try:
            with self.assertRaisesRegex(ImportError, 'smbus'):
                imu_module.IMU()
        finally:
            imu_module.smbus = original


class CarTests(unittest.TestCase):
    def car(self, gyro):
        from test_rc_input import load_car
        module = load_car()
        car = module.Car.__new__(module.Car)
        car.odometer = Mock(get_vel=Mock(return_value=(1., True)))
        car.v_filt = Mock(filter=lambda v: v)
        car.w_filt = Mock(filter=lambda w: w)
        car.w_gyro_filt = Mock(filter=lambda w: w)
        car.v, car.st, car.dt = 1., .2, .02
        car.steer_response = .89
        car.gyro = gyro
        return car, module

    def test_measured_yaw_replaces_the_model_and_model_returns_when_it_fails(self):
        gyro = Mock(read=Mock(side_effect=[.5, math.nan]))
        car, module = self.car(gyro)
        _, w = car.get_vel()
        self.assertEqual((w, car.yaw_source), (.5, 'imu'))
        command = math.tan(.2)/module.CAR['L']
        self.assertAlmostEqual(gyro.read.call_args[0][1], command)  # resposta medida contra o comando puro
        _, w = car.get_vel()
        self.assertAlmostEqual(w, .89*command)
        self.assertEqual(car.yaw_source, 'modelo')

    def test_without_imu(self):
        car, module = self.car(None)
        _, w = car.get_vel()
        self.assertAlmostEqual(w, .89*math.tan(.2)/module.CAR['L'])
        self.assertTrue(math.isnan(car.w_imu))


class DampingTests(unittest.TestCase):
    def test_term_opposes_extra_yaw_and_vanishes_without_gyro(self):
        controller = StanleyController(gain=2.5, softening=.3, yaw_damping=.3)
        controller.update(0., 0., 1., .02, curvature=.5, yaw_rate=.8)   # pede .5 rad/s, gira .8
        self.assertAlmostEqual(controller.damping_term, -.3*.3)
        controller.update(0., 0., 1., .02, curvature=.5, yaw_rate=math.nan)
        self.assertEqual(controller.damping_term, 0.)
        off = StanleyController(gain=2.5, softening=.3)
        off.update(0., 0., 1., .02, curvature=.5, yaw_rate=.8)
        self.assertEqual(off.damping_term, 0.)
        with self.assertRaises(ValueError):
            StanleyController(yaw_damping=-1.)

    def test_command_line(self):
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['main.py', '--sem-imu', '--amortecer-guinada', '0.2']), \
                patch.object(main, 'run_mission') as run:
            main.main()
        settings = run.call_args[0][0]
        self.assertIs(settings['imu'], False)
        self.assertEqual(settings['stanley_yaw_damping'], .2)


class SummaryTests(unittest.TestCase):
    def test_run_summary_and_scale_suggestion(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        for command, real in ((.6, .80*.6), (-.6, .95*-.6)):
            sensor.yaw_rate = real
            for _ in range(100):
                gyro.read(1., command, .02)
        recorder = SimpleNamespace(notes={})
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as out:
            summary = main.save_imu(SimpleNamespace(gyro=gyro, imu_status='ok'), recorder, folder,
                                    {'steering_response': .89, 'steering_left_scale': 1., 'steering_right_scale': 1.})
            self.assertEqual(json.loads(Path(folder, 'imu.json').read_text())['ativa'], True)
        self.assertIs(recorder.notes['imu'], summary)
        text = out.getvalue()
        self.assertIn('assimetrica', text)
        self.assertIn('--esterco-esquerda 109', text)   # 100 * 0,875 / 0,80
        self.assertIn('--esterco-direita 92', text)

    def test_unused_imu_is_recorded_with_the_reason(self):
        with redirect_stdout(io.StringIO()):
            summary = main.save_imu(SimpleNamespace(gyro=None, imu_status='OSError: Remote I/O error'))
        self.assertEqual(summary, {'ativa': False, 'motivo': 'OSError: Remote I/O error'})
        self.assertIsNone(main.save_imu(SimpleNamespace()))  # carro simulado


class VirtualTrackTests(unittest.TestCase):
    def test_gyro_lap_on_a_car_that_turns_less_than_asked(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        import pista_virtual as pv
        track = pv.build_track(pv.TRACKS['oval'], 1.8, None, .5, rng=np.random.default_rng(100))
        result = pv.run(track, pv.base_config({'sim_gyro': True}), seed=1, fps=12., steer_response=.88)
        self.assertTrue(result['finished'])
        self.assertEqual(result['hits'], 0)


if __name__ == '__main__':
    unittest.main()


class SteadyTurnTests(unittest.TestCase):
    """24/11: com o esterco oscilando, a razao instantanea desligava uma IMU boa."""
    def test_oscillating_steering_does_not_disable_a_good_gyro(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        lag = []
        with redirect_stdout(io.StringIO()):
            for k in range(1500):          # 30 s pedalando a 2 Hz
                command = .6*math.sin(2*math.pi*2.*k*.02)
                lag.append(command)
                sensor.yaw_rate = .85*lag[max(0, len(lag) - 13)]   # guinada real 0,25 s atrasada
                gyro.read(1., command, .02)
        self.assertTrue(gyro.enabled, gyro.reason)

    def test_response_is_measured_in_steady_turns(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        for command in (.6, -.6):
            sensor.yaw_rate = .8*command
            for _ in range(200):
                gyro.read(1., command, .02)
        response = gyro.response()
        self.assertAlmostEqual(response['esquerda'], .8, delta=.03)
        self.assertAlmostEqual(response['direita'], .8, delta=.03)

    def test_dead_axis_still_disabled_in_a_steady_turn(self):
        sensor = FakeSensor()
        gyro = calibrated(sensor)
        sensor.yaw_rate = 0.
        with redirect_stdout(io.StringIO()):
            for _ in range(300):
                gyro.read(1., .6, .02)
        self.assertFalse(gyro.enabled)

    def test_vibration_at_start_is_retried(self):
        sensor = FakeSensor(noise=4.)       # 4 graus/s parado: falha as 2 primeiras, passa na 3a (ate 6)
        sensor.close = Mock()
        logs = []
        from fva_car import yaw_rate
        original = yaw_rate.GyroYaw.calibrate
        yaw_rate.GyroYaw.calibrate = lambda self, **kw: original(self, sleep=lambda _: None, **kw)
        try:
            with unittest.mock.patch.object(yaw_rate.time, 'sleep'):
                result = start_gyro({}, Mock(return_value=sensor), logs.append)
        finally:
            yaw_rate.GyroYaw.calibrate = original
        self.assertIsNotNone(result[1])
        self.assertEqual(sum('tentando de novo' in text for text in logs), 2)
