"""Regressoes do corredor por cones e da troca de modo, sem webcam/display."""

from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cone_path import ConePathDetector, ConeSettings, PURPLE, render_cones
from detectar_linha_amarela_hsv import CameraApp
from yellow_path import YellowPathDetector, PathSettings


def cone(frame, x, y, color):
    height = round(16 + y*.055)
    half = round(height*.35)
    triangle = np.array([(x, y-height), (x-half, y), (x+half, y)], np.int32)
    cv2.fillConvexPoly(frame, triangle, color)


def track(kind="straight", shift=0, swapped=False, staggered=False, red_hue=0):
    frame = np.full((480, 640, 3), 35, np.uint8)
    red = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[red_hue, 230, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])
    colors = ((0, 220, 0), red)
    if swapped:
        colors = colors[::-1]
    for side, color in zip((-1, 1), colors):
        for base in (180, 240, 305, 375, 455):
            y = base - (16 if staggered and side == 1 else 0)
            center = center_at(y, kind) + shift
            half_width = 35 + .27*y
            cone(frame, round(center + side*half_width), y, color)
    return frame


def center_at(y, kind):
    y = np.asarray(y)
    if kind == "curve":
        return 300 + .0012*(y-455)**2
    if kind == "s_curve":
        return 320 + 55*np.sin((y-180)/275*2*np.pi)
    return np.zeros_like(y) + 320


class ConePathTests(unittest.TestCase):
    def test_straight_curves_and_staggered_cones(self):
        for kind in ("straight", "curve", "s_curve"):
            for staggered in (False, True):
                with self.subTest(kind=kind, staggered=staggered):
                    result = ConePathDetector().process(track(kind, staggered=staggered))
                    self.assertTrue(result.valid, result.status)
                    self.assertEqual(len(result.green_points), 5)
                    self.assertEqual(len(result.red_points), 5)
                    np.testing.assert_array_equal(np.diff(result.center[:, 1]), 1)
                    expected = center_at(result.center[:, 1], kind)
                    # Cinco cones por lado: erro de interpolacao < 2.5% da largura,
                    # inclusive quando as bases das duas cores estao alternadas.
                    self.assertLess(np.max(abs(result.center[:, 0]-expected)), 16)
                    midpoint = (result.left_boundary[:, 0]+result.right_boundary[:, 0])/2
                    self.assertLessEqual(np.max(abs(result.center[:, 0]-midpoint)), 1)

    def test_lateral_distances_and_error_are_at_same_section(self):
        result = ConePathDetector().process(track(shift=40))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.left_distance_px-result.right_distance_px,
                               -2*result.lateral_error_px)
        self.assertAlmostEqual(result.lateral_error_px, 40.5, delta=2)
        self.assertGreater(result.left_distance_px, 0)
        self.assertGreater(result.right_distance_px, 0)
        # Alinhar o eixo ao centro deslocado zera o erro medido.
        aligned = ConePathDetector().process(track(shift=40), ConeSettings(camera_center=56))
        self.assertLess(abs(aligned.lateral_error_px), 3)

    def test_color_sides_can_be_swapped(self):
        image = track(swapped=True)
        self.assertFalse(ConePathDetector().process(image).valid)
        result = ConePathDetector().process(image, ConeSettings(green_left=False))
        self.assertTrue(result.valid)
        self.assertLess(abs(result.lateral_error_px), 2)

    def test_red_detects_both_ends_of_hue_circle(self):
        for hue in (0, 5, 174, 179):
            with self.subTest(hue=hue):
                result = ConePathDetector().process(track(red_hue=hue))
                self.assertTrue(result.valid, result.status)
                self.assertEqual(len(result.red_points), 5)

    def test_loss_of_either_side_or_entire_frame_clears_path(self):
        detector = ConePathDetector()
        for missing in ("green", "red", "all"):
            self.assertTrue(detector.process(track(), timestamp=1.).valid)
            frame = track()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 0] > 30 if missing == "green" else hsv[:, :, 0] < 15
            if missing == "all":
                mask[:] = True
            frame[mask] = 35
            result = detector.process(frame, timestamp=1.05)
            self.assertFalse(result.valid)
            self.assertIsNone(result.left_distance_px)
            self.assertIsNone(result.lateral_error_px)

    def test_two_cones_per_side_are_insufficient(self):
        frame = np.full((480, 640, 3), 35, np.uint8)
        for y in (240, 440):
            cone(frame, 200, y, (0, 220, 0))
            cone(frame, 440, y, (0, 0, 240))
        self.assertFalse(ConePathDetector().process(frame).valid)

    def test_gap_and_common_span_limits(self):
        self.assertFalse(ConePathDetector().process(track(), ConeSettings(max_gap=40)).valid)
        self.assertFalse(ConePathDetector().process(track(), ConeSettings(min_span=65)).valid)

    def test_borders_without_shared_depth_do_not_form_corridor(self):
        frame = np.full((480, 640, 3), 35, np.uint8)
        for y in (175, 230, 285):
            cone(frame, 220, y, (0, 220, 0))
        for y in (335, 390, 455):
            cone(frame, 450, y, (0, 0, 240))
        result = ConePathDetector().process(frame)
        self.assertFalse(result.valid)
        self.assertIn("extensao comum", result.status)

    def test_axis_outside_corridor_retains_signed_distances(self):
        result = ConePathDetector().process(track(), ConeSettings(camera_center=90))
        self.assertTrue(result.valid)
        self.assertGreater(result.left_distance_px, 0)
        self.assertLess(result.right_distance_px, 0)
        self.assertLess(result.lateral_error_px, 0)
        self.assertIn("fora do corredor", result.status)

    def test_path_stays_within_both_observed_ranges(self):
        result = ConePathDetector().process(track(staggered=True), ConeSettings(reference_y=98))
        self.assertTrue(result.valid)
        self.assertGreaterEqual(result.center[0, 1], max(result.green_points[:, 1].min(), result.red_points[:, 1].min()))
        self.assertLessEqual(result.center[-1, 1], min(result.green_points[:, 1].max(), result.red_points[:, 1].max()))
        self.assertEqual(result.reference_y, result.center[-1, 1])

    def test_rectangles_stripes_specks_and_yellow_line_are_rejected(self):
        for kind in ("rectangle", "stripe", "speck", "yellow"):
            frame = np.full((480, 640, 3), 35, np.uint8)
            for x, color in ((200, (0, 220, 0)), (440, (0, 0, 240))):
                for y in (180, 300, 450):
                    if kind == "rectangle":
                        cv2.rectangle(frame, (x-12, y-32), (x+12, y), color, -1)
                    elif kind == "stripe":
                        cv2.rectangle(frame, (x-60, y-4), (x+60, y), color, -1)
                    elif kind == "speck":
                        cv2.circle(frame, (x, y), 1, color, -1)
                    else:
                        cv2.line(frame, (x, 160), (x, 470), (0, 220, 240), 16)
            with self.subTest(kind=kind):
                result = ConePathDetector().process(frame)
                self.assertFalse(result.valid)
                self.assertEqual(len(result.green_points)+len(result.red_points), 0)

    def test_off_track_cone_does_not_pull_boundary(self):
        frame = track("curve")
        cone(frame, 40, 280, (0, 220, 0))
        cone(frame, 600, 270, (0, 0, 240))
        result = ConePathDetector().process(frame)
        self.assertTrue(result.valid, result.status)
        self.assertLess(np.max(abs(result.center[:, 0]-center_at(result.center[:, 1], "curve"))), 10)

    def test_resolution_scaling(self):
        for width, height in ((320, 240), (1280, 960)):
            result = ConePathDetector().process(cv2.resize(track(), (width, height)))
            with self.subTest(size=(width, height)):
                self.assertTrue(result.valid, result.status)
                self.assertLess(abs(result.lateral_error_px), 3)

    def test_render_preserves_frame_and_draws_purple_center(self):
        frame = track("curve")
        original = frame.copy()
        result = ConePathDetector().process(frame)
        for path_only in (False, True):
            rendered = render_cones(frame, result, path_only=path_only)
            x, y = result.center[len(result.center)//2]
            np.testing.assert_array_equal(rendered[y, x], PURPLE)
        np.testing.assert_array_equal(frame, original)


class Variable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ModeSwitchTests(unittest.TestCase):
    def make_app(self):
        app = CameraApp.__new__(CameraApp)
        app.mode = Variable("amarela")
        app.yellow_detector, app.cone_detector = YellowPathDetector(), ConePathDetector()
        app.detector = app.yellow_detector
        app.variables = {k: Variable(v) for k, v in asdict(PathSettings()).items()}
        app.cone_variables = {k: Variable(v) for k, v in asdict(ConeSettings()).items()}
        app.notebook = Mock()
        app.yellow_tabs, app.cone_tabs = ("geometry", "color", "gaps"), ("cones", "hsv")
        app.hint, app.legend, app.support_text, app.state = (Variable() for _ in range(4))
        app.last_frame, app.last_result = track(), None
        app.redraw = Mock()
        return app

    def test_switch_reprocesses_current_frame_and_clears_other_detector_history(self):
        app = self.make_app()
        app.yellow_detector.previous = np.ones((4, 2))
        app.mode.set("cones")
        app.switch_mode()
        self.assertIs(app.detector, app.cone_detector)
        self.assertIsInstance(app.settings(), ConeSettings)
        self.assertTrue(app.last_result.valid)
        self.assertIsNone(app.yellow_detector.previous)
        self.assertIn("5 verdes", app.support_text.get())
        app.variables["s_min"].set(38)
        app.cone_variables["s_min"].set(120)
        # Voltar a amarela nao deve conservar a trajetoria roxa dos cones.
        app.mode.set("amarela")
        app.switch_mode()
        self.assertIs(app.detector, app.yellow_detector)
        self.assertFalse(app.last_result.valid)
        self.assertIsInstance(app.settings(), PathSettings)
        self.assertEqual(app.settings().s_min, 38)
        app.mode.set("cones")
        self.assertEqual(app.settings().s_min, 120)

    def test_restore_resets_both_settings_and_cone_mode_ignores_click(self):
        app = self.make_app()
        app.mode.set("cones")
        app.switch_mode()
        app.cone_variables["green_left"].set(False)
        app.variables["s_min"].set(180)
        app.restore()
        self.assertEqual(app.settings(), ConeSettings())
        self.assertEqual(app.variables["s_min"].get(), PathSettings().s_min)
        result = app.last_result
        app.select_path(SimpleNamespace(x=100, y=100))
        self.assertIs(app.last_result, result)


if __name__ == "__main__":
    unittest.main()
