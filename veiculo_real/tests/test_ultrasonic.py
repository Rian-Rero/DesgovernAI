"""Captura de eco e diagnostico com GPIO substituido, sem hardware."""

import importlib.util
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "_fva_ultrasonic_tests", Path(__file__).resolve().parents[1] / "fva_car" / "ultrasonic.py"
)
ultrasonic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ultrasonic)


class UltrasonicTests(unittest.TestCase):
    def make_sensor(self, version=5):
        gpio = Mock()
        gpio.gpio_read.return_value = 0
        gpio.input.return_value = 0
        gpio.BOTH_EDGES = 3
        gpio.BOTH = 33
        rpi = types.ModuleType("RPi")
        rpi.GPIO = gpio
        with patch.dict(sys.modules, {"lgpio": gpio, "RPi": rpi, "RPi.GPIO": gpio}), patch.object(
            ultrasonic.Ultrasonic, "detect_rpi_version", return_value=version
        ), patch.object(ultrasonic.Ultrasonic, "_find_gpiochip", return_value=0), patch.object(
            threading.Thread, "start"
        ):
            sensor = ultrasonic.Ultrasonic()
        return sensor, gpio

    def echo_on_trigger(self, sensor, duration_ns=2_000_000, delay=0):
        def trigger(state):
            if not state:
                start = time.monotonic_ns()

                def emit():
                    sensor._on_lgpio_echo(0, sensor.echo_pin, 1, start)
                    sensor._on_lgpio_echo(0, sensor.echo_pin, 0, start + duration_ns)

                if delay:
                    timer = threading.Timer(delay, emit)
                    timer.start()
                    self.addCleanup(timer.join)
                else:
                    emit()
        sensor._set_trigger = trigger

    def test_pi5_registers_both_edges_and_cleans_up(self):
        sensor, gpio = self.make_sensor()
        gpio.gpio_claim_alert.assert_called_once_with(sensor.handle_chip, sensor.echo_pin, gpio.BOTH_EDGES)
        gpio.callback.assert_called_once_with(
            sensor.handle_chip, sensor.echo_pin, gpio.BOTH_EDGES, sensor._on_lgpio_echo
        )
        sensor.cleanup()
        sensor.echo_callback.cancel.assert_called_once()
        gpio.gpiochip_close.assert_called_once()

    def test_pi4_registers_edges_and_uses_callback(self):
        sensor, gpio = self.make_sensor(4)
        gpio.add_event_detect.assert_called_once_with(sensor.echo_pin, gpio.BOTH, callback=sensor._on_rpi_echo)

        def trigger(state):
            if not state:
                now = time.monotonic_ns()
                with patch.object(ultrasonic.time, "monotonic_ns", side_effect=[now, now + 2_000_000]):
                    gpio.input.return_value = 1
                    sensor._on_rpi_echo(sensor.echo_pin)
                    gpio.input.return_value = 0
                    sensor._on_rpi_echo(sensor.echo_pin)

        sensor._set_trigger = trigger
        self.assertAlmostEqual(sensor.get_measure(), 0.343)
        sensor.cleanup()
        gpio.remove_event_detect.assert_called_once_with(sensor.echo_pin)

    def test_delayed_callback_uses_edge_timestamps_not_delivery_time(self):
        sensor, gpio = self.make_sensor()
        self.echo_on_trigger(sensor, delay=0.015)
        sensor._sample_once()
        status = sensor.get_status()
        self.assertTrue(status["valid"])
        self.assertAlmostEqual(status["distance"], 0.343)
        self.assertEqual(status["last_error"], "nenhum")
        self.assertAlmostEqual(status["pulse_s"], 0.002)
        gpio.gpio_read.assert_called_once()

    def test_missing_rising_edge_is_diagnosed(self):
        sensor, _ = self.make_sensor()
        sensor._sample_once()
        status = sensor.get_status()
        self.assertFalse(status["valid"])
        self.assertEqual(status["reason"], "sem_leitura")
        self.assertEqual(status["last_error"], "timeout_subida_echo")
        self.assertEqual(status["consecutive_failures"], 1)

    def test_missing_falling_edge_is_diagnosed(self):
        sensor, _ = self.make_sensor()
        sensor._set_trigger = lambda state: None if state else sensor._record_echo(1, time.monotonic_ns())
        sensor._sample_once()
        self.assertEqual(sensor.get_status()["last_error"], "timeout_descida_echo")

    def test_already_high_echo_is_diagnosed_without_trigger(self):
        sensor, gpio = self.make_sensor()
        gpio.gpio_read.return_value = 1
        sensor._sample_once()
        self.assertEqual(sensor.get_status()["last_error"], "echo_alto_antes_trigger")
        gpio.gpio_write.assert_not_called()

    def test_out_of_range_and_nonpositive_pulses_are_rejected(self):
        for pulse in (30_000_000, 0, -1):
            with self.subTest(pulse=pulse):
                sensor, _ = self.make_sensor()
                self.echo_on_trigger(sensor, duration_ns=pulse)
                sensor._sample_once()
                self.assertFalse(sensor.get_status()["valid"])
                self.assertIn("eco_fora_faixa", sensor.get_status()["last_error"])

    def test_old_edges_do_not_complete_new_measurement(self):
        sensor, _ = self.make_sensor()
        old = time.monotonic_ns() - 1_000_000_000
        sensor._set_trigger = lambda state: sensor._record_echo(int(state), old)
        sensor._sample_once()
        self.assertEqual(sensor.get_status()["last_error"], "timeout_subida_echo")

    def test_validity_expires_without_extending_timeout(self):
        sensor, _ = self.make_sensor()
        sensor.get_measure = Mock(return_value=2.49)
        with patch.object(ultrasonic.time, "monotonic", return_value=100.0):
            sensor._sample_once()
        sensor.get_measure.return_value = None
        with patch.object(ultrasonic.time, "monotonic", return_value=100.29):
            sensor._sample_once()
            self.assertTrue(sensor.get_status()["valid"])
        with patch.object(ultrasonic.time, "monotonic", return_value=100.31):
            status = sensor.get_status()
            self.assertFalse(status["valid"])
            self.assertEqual(status["reason"], "leitura_expirada")
            self.assertEqual(status["distance"], 2.49)
            self.assertAlmostEqual(status["age"], 0.31)

    def test_gpio_error_is_visible_and_next_measurement_recovers(self):
        sensor, _ = self.make_sensor()
        sensor.get_measure = Mock(side_effect=[OSError("GPIO indisponivel"), 2.49])
        sensor._sample_once()
        status = sensor.get_status()
        self.assertFalse(status["valid"])
        self.assertIn("erro_gpio:OSError:GPIO indisponivel", status["last_error"])
        sensor._sample_once()
        status = sensor.get_status()
        self.assertTrue(status["valid"])
        self.assertEqual(status["consecutive_failures"], 0)
        self.assertEqual(status["total_failures"], 1)


if __name__ == "__main__":
    unittest.main()
