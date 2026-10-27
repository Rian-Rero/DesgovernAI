"""Encerramento da captura sem abrir webcam, inclusive com OpenCV headless."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.camera import Camera


class CameraCleanupTests(unittest.TestCase):
    def test_camera_keeps_hsv_tool_controls_and_requests_minimal_buffer(self):
        capture = Mock()
        capture.isOpened.return_value = True
        with patch('fva_car.camera.cv2.VideoCapture', return_value=capture):
            Camera(cam_index=0)
        properties = {call.args[0]: call.args[1] for call in capture.set.call_args_list}
        self.assertEqual(properties[cv2.CAP_PROP_BUFFERSIZE], 1)
        self.assertEqual(properties[cv2.CAP_PROP_FRAME_WIDTH], 640)
        self.assertEqual(properties[cv2.CAP_PROP_FRAME_HEIGHT], 480)
        for prop in (cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_FOURCC):
            self.assertNotIn(prop, properties)

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
