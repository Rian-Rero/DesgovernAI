import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import YellowLineFollower, yellow_centers


def line_image(color):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, 60:65] = color
    return image


class HsvDetectionTests(unittest.TestCase):
    def test_yellow_keeps_its_center_under_moderate_shadow(self):
        for color in ([255, 220, 0], [140, 120, 0], [70, 60, 0]):
            with self.subTest(rgb=color):
                centers = yellow_centers(line_image(color))
                self.assertGreater(len(centers), 10)
                np.testing.assert_allclose(centers[:, 0], 62.)
        # Abaixo do brilho minimo, a cor deixa de autorizar seguir a linha.
        self.assertEqual(yellow_centers(line_image([45, 40, 0])).shape, (0, 2))

    def test_neutral_and_other_colors_are_not_a_yellow_line(self):
        for color in ([255, 255, 255], [80, 80, 80], [0, 0, 0],
                      [255, 0, 0], [0, 255, 0], [0, 0, 255],
                      [0, 255, 255], [255, 150, 0], [200, 200, 130]):
            with self.subTest(rgb=color):
                self.assertEqual(yellow_centers(line_image(color)).shape, (0, 2))

    def test_camera_channels_are_rgb(self):
        image = line_image([255, 255, 0])
        self.assertGreater(len(yellow_centers(image)), 10)
        # Inverter R/B transforma o amarelo em ciano, que deve ser rejeitado.
        self.assertEqual(yellow_centers(image[..., ::-1]).shape, (0, 2))

    def test_threshold_boundaries_are_inclusive(self):
        for color in ([255, 255, 153], [51, 51, 0]):
            with self.subTest(rgb=color):
                self.assertGreater(len(yellow_centers(line_image(color))), 10)
        # Amarelo puro tem H=60 graus, S=1 e V=1 exatamente.
        centers = yellow_centers(line_image([255, 255, 0]),
                                 hsv_lower=(60., 1., 1.), hsv_upper=(60., 1., 1.))
        self.assertGreater(len(centers), 10)

    def test_custom_lower_and_upper_bounds_change_detection(self):
        orange = line_image([255, 150, 0])  # H aproximadamente 35.3 graus.
        self.assertEqual(len(yellow_centers(orange)), 0)
        self.assertGreater(len(yellow_centers(
            orange, hsv_lower=(30., .4, .2), hsv_upper=(40., 1., 1.))), 10)
        # O teto de brilho pode excluir amarelo claro e manter o sombreado.
        limits = dict(hsv_lower=(40., .4, .2), hsv_upper=(75., 1., .5))
        self.assertEqual(len(yellow_centers(line_image([255, 220, 0]), **limits)), 0)
        self.assertGreater(len(yellow_centers(line_image([70, 60, 0]), **limits)), 10)

    def test_invalid_thresholds_have_clear_failure(self):
        lower, upper = (40., .4, .2), (75., 1., 1.)
        invalid_pairs = [
            ((40., .4), upper), (lower, (75., 1., 1., 1.)),
            (((40., .4, .2),), upper),
            ((math.nan, .4, .2), upper), (lower, (math.inf, 1., 1.)),
            ((-1., .4, .2), upper), (lower, (361., 1., 1.)),
            ((40., -.1, .2), upper), (lower, (75., 1.1, 1.)),
            ((40., .4, -.1), upper), (lower, (75., 1., 1.1)),
            ((80., .4, .2), upper), ((40., .8, .2), (75., .7, 1.)),
            ((40., .4, .8), (75., 1., .7)),
        ]
        for bounds in invalid_pairs:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                yellow_centers(line_image([255, 255, 0]), *bounds)

    def test_follower_applies_configured_thresholds_to_camera_image(self):
        # A faixa laranja e visivel somente com a configuracao personalizada.
        for parameters, expected_detection in [
            ({}, False),
            ({'yellow_hsv_lower': (30., .4, .2),
              'yellow_hsv_upper': (40., 1., 1.)}, True),
        ]:
            with self.subTest(parameters=parameters):
                car = Mock()
                car.parameters = parameters
                car.t, car.dt, car.v = .05, .05, 1.
                car.vref, car.u, car.st, car.motor_pwm_percent = 0., 0., 0., 0.
                car.get_image.return_value = line_image([255, 150, 0])
                car.sim.getObjectMatrix.return_value = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
                car.sim.getObjectPosition.return_value = [0, 0, 0]
                car.sim.getObjectFloatParam.return_value = math.pi / 3
                follower = YellowLineFollower(car)
                follower(car)
                car.get_image.assert_called_once_with(gray=False)
                count = car.log.call_args.kwargs['line_pixels']
                self.assertEqual(count > 10, expected_detection)


if __name__ == '__main__':
    unittest.main()
