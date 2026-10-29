"""Transmissao MJPEG da visao, em porta local efemera e sem camera ou motores."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import urllib.request

import cv2
import numpy as np

REAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL))
from fva_car.stream import VisionStream
from test_mission import mission
from test_lateral import lateral, calibration_data, calibration_from, metric_line_frame


class VisionStreamTests(unittest.TestCase):
    def test_browser_receives_rendered_jpeg_and_page(self):
        rendered = []

        def render(observation, text):
            rendered.append(text)
            return np.full((60, 80, 3), observation, np.uint8)

        stream = VisionStream(render, port=0, fps=30, host='127.0.0.1').start()
        try:
            base = f'http://127.0.0.1:{stream.port}'
            with urllib.request.urlopen(base + '/', timeout=5) as page:
                self.assertIn(b'src="stream"', page.read())
            self.assertEqual(rendered, [])  # sem espectador, nada e comprimido
            stream.publish(200, 'quadro')
            with urllib.request.urlopen(base + '/stream', timeout=5) as response:
                self.assertIn('multipart/x-mixed-replace', response.headers['Content-Type'])
                self.assertEqual(response.readline().strip(), b'--quadro')
                self.assertEqual(response.readline().strip(), b'Content-Type: image/jpeg')
                size = int(response.readline().split(b':')[1])
                response.readline()
                image = cv2.imdecode(np.frombuffer(response.read(size), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (60, 80, 3))
            self.assertAlmostEqual(float(image.mean()), 200, delta=3)
            self.assertEqual(rendered, ['quadro'])
        finally:
            stream.close()

    def test_close_without_start_does_not_block(self):
        VisionStream(lambda *_: None, port=0, host='127.0.0.1').close()

    def test_mission_overlay_handles_valid_and_missing_observations(self):
        detector = lateral.RealLineDetector(calibration_from(calibration_data()), render_preview=False,
                                            processing_width=320)
        car = SimpleNamespace(v=.5, vref=.7, st=.1)
        valid = detector.observe_frame(metric_line_frame(), 1.)
        self.assertTrue(valid.valid, valid.reason)
        for observation in (valid, lateral.LineObservation(timestamp=1., reason='Camera sem imagem'), None):
            with self.subTest(observation=observation):
                lines = mission.stream_text(car, 'Faixa valida; controle liberado', observation)
                image = mission.render_stream_frame(detector, observation, lines)
                self.assertEqual(image.shape, (240, 320, 3))
        self.assertIsNone(valid.frame)  # o desenho nao altera a observacao da missao

    def test_cli_stream_uses_default_port(self):
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['main.py', '--stream']), \
                patch.object(mission, 'run_mission') as run:
            mission.main()
        run.assert_called_once_with({'velocity_reference': mission.MAIN_VEL, 'stream_port': 8080},
                                    display=False)
        self.assertEqual(mission.parameters['stream_port'], 0)


if __name__ == '__main__':
    unittest.main()
