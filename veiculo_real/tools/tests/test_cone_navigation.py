"""Curvas metricas, perdas de cones, movimento e expiracao temporal."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cone_geometry import (image_homography, load_calibration, metric_corridor,
                           normals, virtual_boundary)
from cone_path import ConePathDetector, ConeSettings, render_cones
from cone_tracking import ConeTracker, FloorMotion, ego_transform, transform_points
from test_cone_path import cone, track


H = np.array([[0., 0., 100.], [-1., 0., 320.], [0., 1., -90.]])


def calibration():
    return SimpleNamespace(image_size=(640, 480), homography=H)


def metric_scene(missing=None, lateral=0.):
    frame = np.full((480, 640, 3), 35, np.uint8)
    for color, y, bgr in (('green', .6+lateral, (0, 220, 0)), ('red', -.6+lateral, (0, 0, 240))):
        if color == missing or missing == 'all':
            continue
        ground = np.array([(x, y) for x in (.7, 1.1, 1.6, 2.3, 3.2)])
        pixels = transform_points(ground, np.linalg.inv(H))
        for (distance, _), (x, y) in zip(ground, pixels):
            # Tamanho aparente proporcional a 1/distancia, sem fundir cones.
            x, y = round(x), round(y)
            height, half = max(6, round(20/distance)), max(2, round(7/distance))
            cv2.fillConvexPoly(frame, np.array([(x, y-height), (x-half, y), (x+half, y)], np.int32), bgr)
    return frame


class GroundCurveTests(unittest.TestCase):
    def test_parallel_curves_have_metric_normal_width(self):
        x = np.linspace(.5, 4, 8)
        center = np.c_[x, .4*x]
        n = normals(center)
        result = metric_corridor(center+.6*n, center-.6*n)
        self.assertIsNotNone(result)
        c, left, right, widths, curvature = result
        self.assertLess(np.max(abs(c[:, 1]-.4*c[:, 0])), .002)
        np.testing.assert_allclose(widths, 1.2, atol=.003)
        np.testing.assert_allclose(np.linalg.norm(c-left, axis=1), np.linalg.norm(c-right, axis=1), atol=1e-8)
        self.assertLess(np.max(abs(curvature)), .01)
        # La distancia horizontal entre bordas seria maior que a largura normal.
        self.assertGreater(1.2*np.sqrt(1+.4**2), np.median(widths))

    def test_arc_center_is_between_curves_with_staggered_cones(self):
        a = np.linspace(.12, 1., 11)
        b = np.linspace(.08, 1.05, 9)
        left = np.c_[2.5*np.sin(a), 3-2.5*np.cos(a)]
        right = np.c_[3.5*np.sin(b), 3-3.5*np.cos(b)]
        result = metric_corridor(left, right)
        self.assertIsNotNone(result)
        c, _, _, widths, _ = result
        self.assertLess(np.max(abs(np.linalg.norm(c-[0, 3], axis=1)-3)), .02)
        np.testing.assert_allclose(widths, 1., atol=.02)

    def test_s_curve_and_perpendicular_virtual_boundary(self):
        x = np.linspace(.5, 4., 15)
        center = np.c_[x, .3*np.sin(2*x)]
        n = normals(center)
        left, right = center+.5*n, center-.5*n
        result = metric_corridor(left, right)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[3], 1., atol=.04)
        virtual = virtual_boundary(left, 1., True)
        self.assertGreater(np.ptp(virtual[:, 1]), .4)

    def test_width_margin_crossing_and_turn_radius_reject_invalid_corridors(self):
        x = np.linspace(.5, 3., 8)
        left, right = np.c_[x, x*0+.6], np.c_[x, x*0-.6]
        self.assertIsNone(metric_corridor(right, left))
        self.assertIsNone(metric_corridor(left, right, margin=.7))
        self.assertIsNone(metric_corridor(left, right, max_width=1.))
        t = np.linspace(.15, 1., 12)
        left = np.c_[.5*np.sin(t), 1.-.5*np.cos(t)]
        right = np.c_[1.5*np.sin(t), 1.-1.5*np.cos(t)]
        self.assertIsNone(metric_corridor(left, right, min_radius=2.))

    def test_calibration_loader_and_resize_pixel_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration.json'
            path.write_text(json.dumps({'image_size': [640, 480], 'homography': H.tolist()}))
            loaded = load_calibration(path)
            small = image_homography(loaded, (240, 320))
            pixel = np.array([[100., 150.]])
            np.testing.assert_allclose(transform_points(pixel, small),
                                       transform_points((pixel+.5)*2-.5, H))
            with self.assertRaises(ValueError):
                image_homography(loaded, (360, 640))
            path.write_text(json.dumps({'image_size': [640, 480], 'homography': np.zeros((3, 3)).tolist()}))
            with self.assertRaises(ValueError):
                load_calibration(path)


class TrackingTests(unittest.TestCase):
    def test_per_cone_expiry_is_not_refreshed_by_other_cones(self):
        tracker = ConeTracker()
        points = {'green': np.array([[1., .5], [2., .5]]), 'red': np.array([[1., -.5]])}
        tracker.update(points, 1., .35, metric=True)
        for timestamp in (1.1, 1.2, 1.3, 1.4):
            tracker.update({'green': points['green'][:1], 'red': np.empty((0, 2))}, timestamp, .35, metric=True)
        self.assertEqual(len(tracker.points('green')), 1)
        self.assertEqual(len(tracker.points('red')), 0)

    def test_association_by_color_and_kalman_smooths_jitter(self):
        tracker = ConeTracker()
        tracker.update({'green': [[1., .1]], 'red': [[1., .1]]}, 1., .35, metric=True)
        tracker.update({'green': [[1., .2]], 'red': [[1., .1]]}, 1.1, .35, metric=True)
        self.assertGreater(tracker.points('green')[0, 1], .1)
        self.assertLess(tracker.points('green')[0, 1], .2)
        np.testing.assert_allclose(tracker.points('red'), [[1., .1]])

    def test_ego_motion_rotates_and_translates_remembered_cones(self):
        motion = ego_transform((.2, .1, np.pi/2))
        np.testing.assert_allclose(transform_points([[1.2, .1]], motion), [[0., -1.]], atol=1e-8)
        with self.assertRaises(ValueError):
            ego_transform((np.nan, 0, 0))

    def test_floor_flow_estimates_known_translation_and_rejects_blank_image(self):
        rng = np.random.default_rng(4)
        gray = rng.integers(0, 255, (480, 640), np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        shifted = cv2.warpAffine(image, np.array([[1., 0., 5.], [0., 1., 3.]]), (640, 480))
        motion = FloorMotion()
        self.assertIsNone(motion.update(image))
        matrix = motion.update(shifted)
        self.assertIsNotNone(matrix)
        np.testing.assert_allclose(transform_points([[320., 350.]], matrix), [[325., 353.]], atol=.5)
        self.assertIsNone(motion.update(np.zeros_like(image)))


class TemporalNavigationTests(unittest.TestCase):
    def test_pixel_map_survives_brief_total_loss_then_expires(self):
        detector = ConePathDetector()
        live = detector.process(track(), timestamp=1.)
        missing = np.full((480, 640, 3), 35, np.uint8)
        predicted = detector.process(missing, timestamp=1.1)
        self.assertTrue(live.valid)
        self.assertTrue(predicted.valid, predicted.status)
        self.assertTrue(predicted.estimated)
        self.assertLess(predicted.confidence, live.confidence)
        self.assertEqual(len(predicted.green_points), 0)
        self.assertFalse(detector.process(missing, timestamp=1.4).valid)

    def test_metric_pipeline_normal_distances_and_reprojection(self):
        detector = ConePathDetector(calibration())
        result = detector.process(metric_scene(lateral=.1), timestamp=1.)
        self.assertTrue(result.valid, result.status)
        self.assertEqual(result.source, 'solo')
        self.assertFalse(result.estimated)
        self.assertAlmostEqual(result.corridor_width_m, 1.2, delta=.05)
        self.assertAlmostEqual(result.lateral_error_m, -.1, delta=.03)
        self.assertAlmostEqual(result.left_distance_m, .7, delta=.04)
        self.assertAlmostEqual(result.right_distance_m, .5, delta=.04)
        frame = metric_scene(lateral=.1)
        before = frame.copy()
        output = render_cones(frame, result)
        self.assertEqual(output.shape, frame.shape)
        np.testing.assert_array_equal(before, frame)

    def test_metric_one_side_reconstruction_uses_learned_width_and_expires(self):
        for missing in ('green', 'red'):
            with self.subTest(missing=missing):
                detector = ConePathDetector(calibration())
                live = detector.process(metric_scene(), timestamp=1.)
                self.assertTrue(live.valid, live.status)
                estimate = detector.process(metric_scene(missing=missing), timestamp=1.5)
                self.assertTrue(estimate.valid, estimate.status)
                self.assertTrue(estimate.estimated)
                self.assertIn('largura aprendida', estimate.status)
                self.assertAlmostEqual(estimate.corridor_width_m, live.corridor_width_m, delta=.03)
                self.assertLess(abs(estimate.lateral_error_m), .04)
                self.assertFalse(detector.process(metric_scene(missing=missing), timestamp=2.6).valid)

    def test_no_width_is_invented_at_startup(self):
        result = ConePathDetector(calibration()).process(metric_scene(missing='green'), timestamp=1.)
        self.assertFalse(result.valid)

    def test_odometry_corrects_total_loss_instead_of_freezing_pixels(self):
        detector = ConePathDetector(calibration())
        live = detector.process(metric_scene(lateral=.1), timestamp=1.)
        predicted = detector.process(metric_scene(missing='all'), timestamp=1.1, motion=(.1, .08, 0.))
        self.assertTrue(predicted.valid, predicted.status)
        self.assertEqual(predicted.motion_source, 'odometria')
        self.assertTrue(predicted.estimated)
        self.assertAlmostEqual(predicted.lateral_error_m, live.lateral_error_m+.08, delta=.02)
        self.assertLess(predicted.ground_center[0, 0], live.ground_center[0, 0])

    def test_odometry_yaw_rotates_the_predicted_curve(self):
        detector = ConePathDetector(calibration())
        self.assertTrue(detector.process(metric_scene(), timestamp=1.).valid)
        predicted = detector.process(metric_scene(missing='all'), timestamp=1.1,
                                     motion=(0., 0., .08))
        self.assertTrue(predicted.valid, predicted.status)
        self.assertAlmostEqual(predicted.heading_rad, -.08, delta=.04)
        self.assertLess(abs(predicted.lateral_error_m), .04)

    def test_green_observations_do_not_revalidate_lost_red_cones(self):
        detector = ConePathDetector(calibration())
        self.assertTrue(detector.process(metric_scene(), timestamp=1.).valid)
        # Vermelho vira azul: um distrator nao deve renovar o cone vermelho.
        swapped = metric_scene()[:, :, ::-1].copy()
        self.assertTrue(detector.process(swapped, timestamp=1.1).estimated)
        expired = detector.process(swapped, timestamp=2.6)
        self.assertFalse(expired.valid)

    def test_metric_resize_and_unconfigured_ego_motion(self):
        result = ConePathDetector(calibration()).process(cv2.resize(metric_scene(), (320, 240)), timestamp=1.)
        self.assertTrue(result.valid, result.status)
        self.assertAlmostEqual(result.corridor_width_m, 1.2, delta=.1)
        result = ConePathDetector().process(track(), timestamp=1., motion=(.1, 0., 0.))
        self.assertFalse(result.valid)
        self.assertIn('calibracao', result.status)

    def test_confidence_threshold_and_tracking_disable_remove_predictions(self):
        frame = metric_scene()
        for settings in (ConeSettings(tracking=False), ConeSettings(min_confidence=80)):
            detector = ConePathDetector(calibration())
            self.assertTrue(detector.process(frame, settings, timestamp=1.).valid)
            lost = detector.process(metric_scene(missing='all'), settings, timestamp=1.2)
            self.assertFalse(lost.valid, lost.status)

    def test_reset_mode_calibration_size_and_clock_changes_clear_memory(self):
        detector = ConePathDetector(calibration())
        blank = metric_scene(missing='all')
        for reset in (lambda: detector.reset(), lambda: detector.set_calibration(calibration())):
            self.assertTrue(detector.process(metric_scene(), timestamp=1.).valid)
            reset()
            self.assertFalse(detector.process(blank, timestamp=1.1).valid)
        self.assertTrue(detector.process(metric_scene(), timestamp=2.).valid)
        self.assertFalse(detector.process(blank, timestamp=1.).valid)
        self.assertTrue(detector.process(metric_scene(), timestamp=3.).valid)
        self.assertFalse(detector.process(cv2.resize(blank, (320, 240)), timestamp=3.1).valid)
        self.assertTrue(detector.process(metric_scene(), timestamp=4.).valid)
        self.assertFalse(detector.process(blank, replace(ConeSettings(), green_left=False), timestamp=4.1).valid)

    def test_bad_calibration_aspect_and_nonfinite_timestamp_do_not_reuse_memory(self):
        detector = ConePathDetector(calibration())
        self.assertTrue(detector.process(metric_scene(), timestamp=1.).valid)
        self.assertFalse(detector.process(np.zeros((360, 640, 3), np.uint8), timestamp=1.1).valid)
        self.assertFalse(detector.process(metric_scene(), timestamp=float('nan')).valid)


if __name__ == '__main__':
    unittest.main()
