"""Percepcao e geometria dos cones sem leitura privilegiada dos objetos da cena."""
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import (ConeObservation, ConePathDetector, detect_cones,
                           reconstruct_cone_path)
from fva_car.cones import _ConeMemory


def world_points(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.column_stack((x, y, np.zeros(x.size)))


class ConeDetectionTests(unittest.TestCase):
    def test_bases_use_ground_contact_and_red_hue_wrap(self):
        image = np.zeros((160, 240, 3), np.uint8)
        cv2.fillConvexPoly(image, np.array([[30, 40], [19, 90], [41, 90]]), (255, 0, 0))
        cv2.fillConvexPoly(image, np.array([[170, 50], [160, 100], [180, 100]]), (0, 255, 0))
        hsv = np.uint8([[[179, 255, 255]]])
        wrapped = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])
        cv2.fillConvexPoly(image, np.array([[90, 50], [80, 100], [100, 100]]), wrapped)
        image[15:30, 110:130] = (255, 0, 0)  # painel quadrado nao e cone
        image[50:130, 215:230] = (0, 0, 255)
        pixels = detect_cones(image)
        self.assertEqual(pixels['red'].shape, (2, 2))
        np.testing.assert_allclose(pixels['red'], [[30, 90], [90, 100]], atol=.1)
        np.testing.assert_allclose(pixels['green'], [[170, 100]], atol=.1)

    def test_missing_image_and_clipped_bases_are_not_cones(self):
        self.assertFalse(ConeObservation().valid)
        for image in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8)):
            self.assertEqual(len(detect_cones(image)['red']), 0)
        image = np.zeros((60, 80, 3), np.uint8)
        cv2.fillConvexPoly(image, np.array([[40, 20], [20, 70], [60, 70]]), (255, 0, 0))
        self.assertEqual(len(detect_cones(image)['red']), 0)

class ConeGeometryTests(unittest.TestCase):
    def test_camera_bobble_keeps_one_metric_landmark_per_cone(self):
        memory = _ConeMemory(6., .5)
        points = world_points([2., 4., 6.], 2.)
        for frame in range(60):
            jitter = np.array([.15*math.sin(frame), .08*math.cos(frame), 0.])
            cloud = memory.update(frame*.02, points+jitter)
            self.assertEqual(len(cloud), 3)
        np.testing.assert_allclose(cloud, points, atol=.08)

    def test_interleaved_cones_from_another_branch_do_not_bend_straight_border(self):
        x = np.array([1.5, 3.5, 5.5, 7.5])
        red = np.vstack((world_points(x, 2.), [4.1, 2.6, 0.]))
        green = world_points(x+.1, -2.)
        observation = reconstruct_cone_path(red, green, [0, 0, 0], [1, 0])
        self.assertTrue(observation.valid, observation.reason)
        self.assertAlmostEqual(observation.errors[0], 0., places=8)
        self.assertAlmostEqual(observation.errors[1], 0., places=8)

    def test_real_scene_parallel_branch_does_not_win_with_an_implausible_quadratic(self):
        red = world_points([3.616, 4.245, 5.631, 6.235, 7.587, 9.67],
                           [1.373, 2.157, 1.063, 1.847, .738, .417])
        green = world_points([2.836, 4.756, 6.718, 8.719, 10.861],
                             [-2.367, -2.702, -3.069, -3.448, -3.877])
        observation = reconstruct_cone_path(red, green, [0, 0, 0], [1, 0])
        self.assertTrue(observation.valid, observation.reason)
        self.assertLess(abs(observation.errors[0]), .15)
        self.assertAlmostEqual(observation.errors[1], -.17, delta=.03)
        self.assertLess(abs(observation.errors[2]), .05)

    def test_invalid_settings_are_rejected_before_perception(self):
        for config in ({'cone_min_area': 0}, {'cone_min_area': 1.5},
                       {'cone_min_width': 8.}, {'cone_max_distance': math.nan},
                       {'cone_freshness': 0.}, {'cone_max_distance': 3.},
                       {'cone_memory_horizon': math.inf}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ConePathDetector(config)

    def test_nonfinite_points_do_not_poison_a_valid_corridor(self):
        x = np.array([1., 3., 5.])
        red = np.vstack((world_points(x, 2.), [math.nan, 1., 0.]))
        green = np.vstack((world_points(x, -2.), [1., math.inf, 0.]))
        observation = reconstruct_cone_path(red, green, [0, 0, 0], [1, 0])
        self.assertTrue(observation.valid, observation.reason)
        self.assertAlmostEqual(observation.errors[0], 0., places=8)

    def test_sparse_straight_borders_center_and_signed_lateral_distances(self):
        # Nao ha pares no mesmo x: nao se pode tirar a media de cones por indice.
        red = world_points([1.5, 3.5, 5.5], [2.4, 2.4, 2.4])
        green = world_points([2., 4., 6.], [-1.6, -1.6, -1.6])
        observation = reconstruct_cone_path(red, green, [0, 0, 0], [1, 0])
        self.assertTrue(observation.valid, observation.reason)
        self.assertAlmostEqual(observation.errors[0], .4, places=8)
        self.assertAlmostEqual(observation.errors[1], 0., places=8)
        self.assertAlmostEqual(observation.errors[2], 0., places=8)
        self.assertAlmostEqual(observation.red_lateral_m, 2.4, places=8)
        self.assertAlmostEqual(observation.green_lateral_m, -1.6, places=8)
        np.testing.assert_allclose(observation.path[:, 1], .4, atol=1e-8)

    def test_curved_concentric_borders_recover_both_turns(self):
        theta = np.array([.2, .45, .7])
        for sign in (-1, 1):
            red_radius, green_radius = (4., 8.) if sign == 1 else (8., 4.)
            def border(radius):
                return world_points(radius*np.sin(theta), sign*(6-radius*np.cos(theta)))
            observation = reconstruct_cone_path(border(red_radius), border(green_radius),
                                                 [0, 0, 0], [1, 0])
            self.assertTrue(observation.valid, observation.reason)
            self.assertAlmostEqual(observation.errors[0], 0., places=8)
            self.assertAlmostEqual(observation.errors[1], 0., places=8)
            self.assertAlmostEqual(observation.errors[2], sign/6, places=8)
            self.assertAlmostEqual(observation.red_lateral_m, 2., places=8)
            self.assertAlmostEqual(observation.green_lateral_m, -2., places=8)

    def test_color_sides_are_not_hardcoded(self):
        x = np.array([1., 3., 5.])
        red, green = world_points(x, -2.), world_points(x, 2.)
        observation = reconstruct_cone_path(red, green, [0, 0, 0], [1, 0])
        self.assertTrue(observation.valid, observation.reason)
        self.assertAlmostEqual(observation.red_lateral_m, -2.)
        self.assertAlmostEqual(observation.green_lateral_m, 2.)
        self.assertAlmostEqual(observation.errors[0], 0., places=8)

    def test_pose_rotation_and_translation_leave_local_errors_unchanged(self):
        x = np.array([1., 3., 5.])
        for angle in (0., .7, 2.4):
            forward = np.array([math.cos(angle), math.sin(angle)])
            left = np.array([-forward[1], forward[0]])
            origin = np.array([5., -3.])
            def border(offset):
                xy = origin + x[:, None]*forward+(offset+.1*x)[:, None]*left
                return np.column_stack((xy, np.zeros(len(x))))
            observation = reconstruct_cone_path(border(2.3), border(-1.7),
                                                [*origin, 0.], forward)
            self.assertTrue(observation.valid, observation.reason)
            self.assertAlmostEqual(observation.errors[0], .3/math.sqrt(1.01), places=8)
            self.assertAlmostEqual(observation.errors[1], math.atan(.1), places=8)

    def test_rejects_single_color_distant_short_and_crossed_corridors(self):
        x = np.array([1., 3., 5.])
        cases = [
            (world_points(x, 2.), np.empty((0, 3))),
            (world_points(x[:1], 2.), world_points(x[:1], -2.)),
            (world_points(x+7., 2.), world_points(x+7., -2.)),
            (world_points(x, 2.-x), world_points(x, -2.+x)),
            (world_points(x, 5.), world_points(x, -5.)),
        ]
        for red, green in cases:
            observation = reconstruct_cone_path(red, green, [0, 0, 0], [1, 0])
            self.assertFalse(observation.valid, observation)

    def test_memory_bridges_short_camera_loss_and_expires(self):
        detector = ConePathDetector({'cone_freshness': .3})
        x = np.array([1., 3., 5.])
        red, green = world_points(x, 2.), world_points(x, -2.)
        detector.memory['red'].update(0., red)
        detector.memory['green'].update(0., green)
        observation = detector.observe(None, np.eye(3, 4), math.pi/3, 0., [0, 0, 0], [1, 0], .1)
        self.assertTrue(observation.valid, observation.reason)
        observation = detector.observe(None, np.eye(3, 4), math.pi/3, 0., [0, 0, 0], [1, 0], .4)
        self.assertFalse(observation.valid)
        self.assertEqual(len(observation.red_points), 0)
        detector.memory['red'].update(.5, world_points([9.], [2.]))
        # Ao readquirir apos expirar, os cones de antes nao ressuscitam.
        self.assertEqual(len(detector.memory['red'].update(.5, np.empty((0, 3)))), 1)


if __name__ == '__main__':
    unittest.main()
