"""Encerramento da captura sem abrir webcam, inclusive com OpenCV headless."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.camera import Camera


class CameraCleanupTests(unittest.TestCase):
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
