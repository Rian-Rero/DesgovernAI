"""Encerramento da captura sem abrir webcam, inclusive com OpenCV headless."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.camera import Camera


class CameraCleanupTests(unittest.TestCase):
    def test_camera_requests_resolution_and_minimal_buffer(self):
        capture = Mock()
        capture.isOpened.return_value = True
        with patch('fva_car.camera.cv2.VideoCapture', return_value=capture):
            Camera(cam_index=0)
        properties = {call.args[0]: call.args[1] for call in capture.set.call_args_list}
        self.assertEqual(properties[cv2.CAP_PROP_BUFFERSIZE], 1)
        self.assertEqual(properties[cv2.CAP_PROP_FRAME_WIDTH], 640)
        self.assertEqual(properties[cv2.CAP_PROP_FRAME_HEIGHT], 480)
        for prop in (cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_EXPOSURE):
            self.assertNotIn(prop, properties)

    def test_empty_settings_restore_automatic_exposure_and_white_balance(self):
        # A webcam guarda o ultimo ajuste no driver: sem isto, a missao herdava a
        # exposicao manual deixada pelo ajustar_camera.py.
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 3. if prop == cv2.CAP_PROP_AUTO_EXPOSURE else 0.
        with patch('fva_car.camera.cv2.VideoCapture', return_value=capture):
            Camera(cam_index=0)
        calls = [call.args for call in capture.set.call_args_list]
        self.assertIn((cv2.CAP_PROP_AUTO_EXPOSURE, 3), calls)
        self.assertIn((cv2.CAP_PROP_AUTO_WB, 1), calls)
        self.assertNotIn(cv2.CAP_PROP_EXPOSURE, [c[0] for c in calls])

    def test_manual_exposure_is_applied(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 1. if prop == cv2.CAP_PROP_AUTO_EXPOSURE else 0.
        with patch('fva_car.camera.cv2.VideoCapture', return_value=capture):
            Camera(cam_index=0, exposure=150, auto_wb=False)
        calls = [call.args for call in capture.set.call_args_list]
        self.assertIn((cv2.CAP_PROP_AUTO_EXPOSURE, 1), calls)
        self.assertIn((cv2.CAP_PROP_EXPOSURE, 150.), calls)
        self.assertIn((cv2.CAP_PROP_AUTO_WB, 0), calls)

    def test_headless_cleanup_releases_camera(self):
        camera = Camera.__new__(Camera)
        camera.cap = Mock()
        with patch('fva_car.camera.cv2.destroyAllWindows', side_effect=cv2.error('headless')):
            camera.close()
        camera.cap.release.assert_called_once()

    def test_cleanup_without_capture_does_not_open_hardware(self):
        camera = Camera.__new__(Camera)
        with patch('fva_car.camera.cv2.destroyAllWindows') as close_windows:
            camera.close()
        close_windows.assert_called_once()
