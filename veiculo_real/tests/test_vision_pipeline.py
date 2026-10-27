"""Captura recente e diagnostico de bloqueios, sem camera ou motores reais."""
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from test_mission import mission
from test_lateral import lateral, fake_car, calibration_data, calibration_from, metric_line_frame


class VisionPipelineTests(unittest.TestCase):
    def test_reduced_processing_preserves_metric_errors_on_both_sides(self):
        calibration = calibration_from(calibration_data())
        for width in (480, 320):
            for offset, slope in ((.12, 0.), (-.12, 0.), (.08, .15)):
                with self.subTest(width=width, offset=offset, slope=slope):
                    frame = metric_line_frame(offset, slope)
                    full = lateral.RealLineDetector(calibration, render_preview=False).observe_frame(frame, 10.)
                    detector = lateral.RealLineDetector(calibration, render_preview=False, processing_width=width)
                    actual = detector.observe_frame(frame, 10.)
                    self.assertTrue(actual.valid, actual.reason)
                    np.testing.assert_allclose(actual.errors, full.errors, atol=.015)
                    self.assertEqual(actual.processing_size, (width, round(width*.75)))
                    self.assertEqual(actual.path_result.path_mask.shape[1], width)
                    self.assertIs(actual.source_frame, frame)
                    self.assertEqual(detector.with_preview(actual).frame.shape, frame.shape)
                    self.assertEqual(actual.timestamp, 10.)

    def test_reduction_still_requires_original_camera_resolution(self):
        detector = lateral.RealLineDetector(calibration_from(calibration_data()), processing_width=320)
        observation = detector.observe_frame(np.zeros((240, 320, 3), np.uint8), 10.)
        self.assertFalse(observation.valid)
        self.assertIn('Resolucao diferente', observation.reason)

    def test_reduced_processing_keeps_curved_and_sparse_dashes(self):
        from test_yellow_update import dashed_scene
        calibration = lateral.GroundCalibration((640, 480),
            np.array([[0., -.005, 2.4], [-.003, 0., .96], [0., 0., 1.]]))
        for curved, sparse in ((False, True), (True, False), (True, True)):
            frame = dashed_scene(curved=curved, sparse=sparse)
            full = lateral.RealLineDetector(calibration, render_preview=False).observe_frame(frame, 10.)
            for width in (480, 320):
                with self.subTest(width=width, curved=curved, sparse=sparse):
                    detector = lateral.RealLineDetector(calibration, render_preview=False, processing_width=width)
                    observation = detector.observe_frame(frame, 10.)
                    self.assertTrue(observation.valid, observation.reason)
                    self.assertAlmostEqual(observation.errors[0], full.errors[0], delta=.015)
                    self.assertAlmostEqual(observation.errors[1], full.errors[1], delta=.04)
                    self.assertAlmostEqual(observation.errors[2], full.errors[2], delta=.08)
                    missing = detector.observe_frame(np.full_like(frame, 35), 10.05)
                    self.assertFalse(missing.valid)

    def test_processing_width_rejects_upscaling_and_invalid_values(self):
        calibration = calibration_from(calibration_data())
        for width in (-1, 80, 641, True, 320., '320'):
            with self.subTest(width=width), self.assertRaises(ValueError):
                lateral.RealLineDetector(calibration, processing_width=width)

    def test_headless_processing_defers_preview_without_changing_metric_results(self):
        calibration = calibration_from(calibration_data())
        settings = {'clahe': False, 'smoothing': 0}
        normal = lateral.RealLineDetector(calibration, settings)
        headless = lateral.RealLineDetector(calibration, settings, render_preview=False)
        frame = metric_line_frame()
        expected = normal.observe_frame(frame, 10.)
        with patch.object(lateral, 'render_path', wraps=lateral.render_path) as render:
            actual = headless.observe_frame(frame, 10.)
            render.assert_not_called()
            self.assertIsNone(actual.frame)
            self.assertEqual(actual.errors, expected.errors)
            self.assertEqual(actual.color_support, expected.color_support)
            self.assertGreaterEqual(actual.detection_seconds, 0.)
            self.assertGreaterEqual(actual.geometry_seconds, 0.)
            with patch.object(headless.detector, 'process') as process:
                preview = headless.with_preview(actual)
                process.assert_not_called()
            render.assert_called_once()
        self.assertEqual(preview.timestamp, 10.)
        self.assertEqual(preview.errors, actual.errors)
        self.assertEqual(preview.detection_seconds, actual.detection_seconds)
        self.assertIsNone(actual.frame)
        np.testing.assert_array_equal(preview.frame, expected.frame)

    def test_capture_drops_old_frames_and_preserves_acquisition_start(self):
        frames = queue.Queue(maxsize=1)
        stop = threading.Event()
        images = [np.full((4, 4, 3), value, np.uint8) for value in (1, 2, 3)]
        reads = iter(images)
        def capture(gray=False):
            frame = next(reads)
            if frame is images[-1]:
                stop.set()
            return frame
        car = SimpleNamespace(get_image=capture)
        times = iter((1., 1.02, 2., 2.02, 3., 3.02))
        mission.capture_func(car, frames, stop, clock=lambda: next(times))
        self.assertEqual(frames.qsize(), 1)
        frame, timestamp, duration, error = frames.get_nowait()
        self.assertIs(frame, images[-1])
        self.assertEqual(timestamp, 3.)
        self.assertAlmostEqual(duration, .02)
        self.assertIsNone(error)

    def test_processing_uses_capture_time_without_refreshing_old_images(self):
        stop = threading.Event()
        frames = queue.Queue(maxsize=1)
        frame = np.zeros((4, 4, 3), np.uint8)
        frames.put((frame, 10., .04, None))
        observation = mission.LineObservation(10., errors=(.1, 0., 0.))
        def observe(image, timestamp):
            self.assertIs(image, frame)
            self.assertEqual(timestamp, 10.)
            stop.set()
            return observation
        detector = SimpleNamespace(observe_frame=observe)
        clock = iter((10.1, 10.45))
        data = {}
        car = Mock()
        mission.vision_func(car, data, stop, detector, lambda: next(clock), frames=frames)
        actual = data['observation']
        self.assertEqual(actual.timestamp, 10.)
        self.assertAlmostEqual(actual.capture_seconds, .04)
        self.assertAlmostEqual(actual.processing_seconds, .35)
        self.assertEqual(actual.sequence, 1)
        car.get_image.assert_not_called()
        follower = mission.RealStanleyFollower({})
        vehicle = fake_car()
        self.assertFalse(follower.apply(vehicle, actual, 10.45))
        self.assertIn('Imagem atrasada', follower.status)
        vehicle.set_u.assert_called_with(0.)

    def test_capture_error_reaches_vision_and_does_not_authorize_control(self):
        frames = queue.Queue(maxsize=1)
        car = Mock()
        car.get_image.side_effect = RuntimeError('camera desconectada')
        stop = threading.Event()
        mission.capture_func(car, frames, stop, clock=lambda: 10.)
        data = {}
        mission.vision_func(car, data, stop, Mock(), clock=lambda: 10., frames=frames)
        self.assertFalse(data['observation'].valid)
        self.assertIn('camera desconectada', data['observation'].reason)

    def test_short_image_path_reports_metric_rejection_and_preserves_frame(self):
        calibration = mission.GroundCalibration((640, 480),
            np.array([[0., -.0015, .72], [-.003, 0., .96], [0., 0., 1.]]))
        detector = mission.RealLineDetector(calibration, mission.parameters['path_settings'])
        frame = np.full((480, 640, 3), 35, np.uint8)
        cv2.line(frame, (266, 350), (266, 479), (0, 220, 240), 16)
        observation = detector.observe_frame(frame, 10.)
        self.assertTrue(detector.detector.process(frame, detector.settings, timestamp=10.).valid)
        self.assertFalse(observation.valid)
        self.assertIs(observation.source_frame, frame)
        follower = mission.RealStanleyFollower({})
        self.assertFalse(follower.apply(fake_car(), observation, 10.1))
        self.assertIn('cobertura metrica', follower.status)

    def test_obstacle_diagnostic_takes_priority_over_visible_line(self):
        follower = lateral.RealStanleyFollower({})
        observation = lateral.LineObservation(10., errors=(.1, 0., 0.))
        car = fake_car()
        self.assertFalse(follower.apply(car, observation, 10., obstacle_detected=True))
        self.assertIn('obstaculo', follower.status)
        car.set_u.assert_called_with(0.)

    def test_snapshot_saves_original_overlay_and_failure_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = np.full((16, 20, 3), (10, 100, 230), np.uint8)
            observation = mission.LineObservation(
                10., frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                source_frame=frame, reason='Buscando faixa', capture_seconds=.03,
                processing_seconds=.4)
            car = SimpleNamespace(logfile=folder, encoder_valid=True, sel_direcao=1, sel_tracao=1)
            mission.save_vision_snapshot(car, observation, 'Imagem atrasada', 10.5)
            status = json.loads((Path(folder) / 'vision_status.json').read_text())
            self.assertEqual(status['control_status'], 'Imagem atrasada')
            self.assertFalse(status['metric_line_valid'])
            self.assertEqual(status['image_size'], [20, 16])
            self.assertEqual(status['vision_age_s'], .5)
            self.assertEqual(status['processing_s'], .4)
            for name in ('vision_camera.jpg', 'vision_detection.jpg'):
                saved = cv2.imread(str(Path(folder) / name))
                self.assertEqual(saved.shape, frame.shape)
                np.testing.assert_allclose(saved, frame, atol=3)


if __name__ == '__main__':
    unittest.main()
