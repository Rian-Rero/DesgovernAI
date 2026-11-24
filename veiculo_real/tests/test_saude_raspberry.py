"""Saude do Raspberry (temperatura, clock, limitacao) e nucleos fixos (fva_car/system_health.py)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402
from fva_car import system_health as health  # noqa: E402


def fake_run(output):
    def run(*_, **__):
        return SimpleNamespace(stdout=output)
    return run


class DecodeTests(unittest.TestCase):
    def test_flags_now_and_since_boot(self):
        self.assertEqual(health.decode_throttled(0), ([], []))
        now, since = health.decode_throttled(0x50005)  # subtensao + throttling agora e antes
        self.assertEqual(now, ['subtensao', 'throttling'])
        self.assertEqual(since, ['subtensao', 'throttling'])
        self.assertEqual(health.decode_throttled(0x80000), ([], ['limite de temperatura']))
        self.assertEqual(health.decode_throttled(None), ([], []))

    def test_vcgencmd_output_and_absence(self):
        self.assertEqual(health.read_throttled(fake_run('throttled=0x50005\n')), 0x50005)
        self.assertIsNone(health.read_throttled(fake_run('lixo')))

        def missing(*_, **__):
            raise FileNotFoundError('vcgencmd')
        self.assertIsNone(health.read_throttled(missing))

        def slow(*_, **__):
            raise subprocess.TimeoutExpired('vcgencmd', 1.)
        self.assertIsNone(health.read_throttled(slow))

    def test_sysfs_readings(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'temp').write_text('71234\n')
            (folder/'scaling_cur_freq').write_text('1500000\n')
            (folder/'cpuinfo_max_freq').write_text('1800000\n')
            (folder/'scaling_governor').write_text('ondemand\n')
            values = health.snapshot(fake_run('throttled=0x0'), folder/'temp', folder)
        self.assertAlmostEqual(values['temp_c'], 71.234)
        self.assertEqual((values['freq_mhz'], values['max_mhz'], values['governor']), (1500., 1800., 'ondemand'))
        self.assertEqual(values['problems_now'], [])
        missing = health.snapshot(fake_run(''), Path('/nao/existe'), Path('/nao/existe'))
        self.assertIsNone(missing['temp_c'])
        self.assertIsNone(missing['governor'])


class MonitorTests(unittest.TestCase):
    def monitor(self, readings):
        readings = iter(readings)
        return health.HealthMonitor(period_s=60., read=lambda: dict(next(readings)))

    def test_summary_keeps_the_worst_of_the_run(self):
        base = {'max_mhz': 1800., 'governor': 'ondemand', 'problems_since_boot': []}
        monitor = self.monitor([
            {**base, 'temp_c': 55., 'freq_mhz': 600., 'problems_now': []},     # parado, clock baixo
            {**base, 'temp_c': 82., 'freq_mhz': 1200., 'problems_now': ['limite de temperatura']},
            {**base, 'temp_c': 78., 'freq_mhz': 1800., 'problems_now': []},
            {**base, 'temp_c': 70., 'freq_mhz': 600., 'problems_now': []},     # fim, parado
        ])
        for _ in range(4):
            monitor.sample()
        summary = monitor.summary()
        self.assertEqual(summary['temp_start_c'], 55.)
        self.assertEqual(summary['temp_max_c'], 82.)
        self.assertEqual(summary['freq_min_mhz'], 1200.)  # sem a 1a e a ultima leitura
        self.assertEqual(summary['problems_during_run'], ['limite de temperatura'])
        self.assertEqual(summary['limited_samples'], 1)
        text = ' '.join(summary['warnings'])
        self.assertIn('DURANTE a volta', text)
        self.assertIn('82 C', text)
        self.assertIn('performance', text)

    def test_healthy_run_has_no_warnings_and_thread_stops(self):
        reading = {'temp_c': 50., 'freq_mhz': 1800., 'max_mhz': 1800., 'governor': 'performance',
                   'problems_now': [], 'problems_since_boot': []}
        monitor = health.HealthMonitor(period_s=.01, read=lambda: dict(reading)).start()
        summary = monitor.stop()
        self.assertGreaterEqual(summary['samples'], 2)
        self.assertEqual(summary['warnings'], [])

    def test_saved_with_the_run(self):
        summary = {'temp_max_c': 60., 'freq_min_mhz': 1800., 'problems_during_run': [], 'warnings': []}
        recorder = SimpleNamespace(notes={})
        with tempfile.TemporaryDirectory() as folder:
            main.save_health(summary, recorder, folder)
            self.assertEqual(json.loads(Path(folder, 'saude_raspberry.json').read_text()), summary)
        self.assertIs(recorder.notes['raspberry'], summary)


class CoreTests(unittest.TestCase):
    def test_layout_needs_four_cores_and_can_be_turned_off(self):
        self.assertEqual(health.core_layout(main.parameters, count=4), ([0, 1], [2, 3]))
        self.assertEqual(health.core_layout({'cpu_cores': True}, count=4), ([0, 1], [2, 3]))
        self.assertEqual(health.core_layout({'cpu_cores': False}, count=4), (None, None))
        self.assertEqual(health.core_layout({}, count=2), (None, None))

    @unittest.skipUnless(hasattr(os, 'sched_setaffinity'), 'sem afinidade neste sistema')
    def test_pin_and_restore(self):
        before = os.sched_getaffinity(0)
        try:
            core = min(before)
            self.assertEqual(health.pin_process([core, 999]), [core])  # nucleo inexistente ignorado
            self.assertEqual(os.sched_getaffinity(0), {core})
            self.assertIsNone(health.pin_process([999]))
            self.assertIsNone(health.pin_process([]))
        finally:
            os.sched_setaffinity(0, before)

    def test_command_line_turns_it_off(self):
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['main.py', '--sem-nucleos']), patch.object(main, 'run_mission') as run:
            main.main()
        self.assertIs(run.call_args[0][0]['cpu_cores'], False)


if __name__ == '__main__':
    unittest.main()
