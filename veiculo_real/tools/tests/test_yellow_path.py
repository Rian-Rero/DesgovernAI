"""Regressoes de percepcao/captura sem webcam nem servidor grafico."""
import sys
from pathlib import Path
import queue
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yellow_path import PathSettings, YellowPathDetector, render_path
from detectar_linha_amarela_hsv import CaptureThread, open_camera


def scene(kind="straight", x=320, thickness=16):
    frame = np.full((480, 640, 3), 35, np.uint8)
    ys = np.arange(145, 480)
    if kind == "curve":
        xs = x + .0012 * (ys - 460) ** 2
    elif kind == "s_curve":
        xs = x + 70 * np.sin((ys - 145) / 335 * 2 * np.pi)
    else:
        xs = np.full_like(ys, x)
    cv2.polylines(frame, [np.c_[xs, ys].astype(np.int32)], False, (0, 220, 240), thickness)
    return frame


def dashed_scene(curved=False, sparse=False):
    frame = np.full((480, 640, 3), 35, np.uint8)
    intervals = ((180, 204), (294, 318), (408, 432)) if sparse else ((155, 205), (275, 325), (395, 479))
    for start, end in intervals:
        ys = np.arange(start, end + 1)
        xs = 320 + (.0012 * (ys - 460) ** 2 if curved else np.zeros(len(ys)))
        cv2.polylines(frame, [np.c_[xs, ys].astype(np.int32)], False, (0, 220, 240), 16)
    return frame


class YellowPathTests(unittest.TestCase):
    def test_straight_curve_and_s_curve(self):
        for kind in ("straight", "curve", "s_curve"):
            with self.subTest(kind=kind):
                result = YellowPathDetector().process(scene(kind))
                self.assertTrue(result.valid)
                self.assertGreater(len(result.center), 290)
                self.assertGreater(result.support, .75)

    def test_small_gaps_are_bridged_without_selecting_other_yellow_objects(self):
        frame = scene()
        frame[280:304] = 35
        frame[370:389] = 35
        cv2.circle(frame, (110, 350), 45, (0, 220, 240), -1)
        cv2.rectangle(frame, (470, 350), (620, 460), (0, 220, 240), -1)
        result = YellowPathDetector().process(frame)
        self.assertTrue(result.valid)
        self.assertEqual(result.path_mask[292, 320], 255)
        self.assertEqual(result.observed_mask[292, 320], 0)
        self.assertEqual(result.path_mask[350, 110], 0)
        self.assertEqual(result.path_mask[400, 550], 0)

    def test_gap_limit_can_be_tuned_and_zero_disables_bridging(self):
        frame = scene()
        frame[245:277] = 35
        frame[340:372] = 35
        self.assertTrue(YellowPathDetector().process(frame).valid)
        self.assertFalse(YellowPathDetector().process(frame, PathSettings(max_gap=0, dashed=False)).valid)

    def test_long_gap_does_not_create_a_path_across_missing_evidence(self):
        frame = scene()
        frame[240:405] = 35
        result = YellowPathDetector().process(frame)
        self.assertFalse(result.valid)
        self.assertEqual(cv2.countNonZero(result.path_mask), 0)

    def test_compact_objects_and_horizontal_stripe_are_not_paths(self):
        for kind in ("circle", "rectangle", "horizontal", "whole_frame"):
            frame = np.full((480, 640, 3), 35, np.uint8)
            if kind == "circle":
                cv2.circle(frame, (320, 360), 60, (0, 220, 240), -1)
            elif kind == "rectangle":
                cv2.rectangle(frame, (220, 280), (420, 460), (0, 220, 240), -1)
            elif kind == "horizontal":
                cv2.line(frame, (30, 360), (610, 360), (0, 220, 240), 18)
            else:
                frame[:] = (0, 220, 240)
            with self.subTest(kind=kind):
                self.assertFalse(YellowPathDetector().process(frame).valid)

    def test_click_anchor_selects_one_of_two_separate_paths(self):
        frame = cv2.max(scene(x=180), scene(x=460))
        for x in (180, 460):
            detector = YellowPathDetector()
            detector.reset(anchor=x / 640)
            result = detector.process(frame)
            self.assertTrue(result.valid)
            self.assertLess(abs(np.median(result.center[:, 0]) - x), 3)
            other = 460 if x == 180 else 180
            self.assertEqual(result.path_mask[350, other], 0)

    def test_shadow_and_desaturated_segment_use_tolerant_color_support(self):
        frame = scene()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[270:340, 312:329, 1:] = (40, 35)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = YellowPathDetector().process(frame, PathSettings(clahe=False))
        self.assertTrue(result.valid)
        self.assertEqual(result.path_mask[305, 320], 255)
        self.assertEqual(result.observed_mask[305, 320], 255)
        self.assertEqual(result.raw_mask[305, 320], 0)

    def test_absence_is_not_hidden_by_temporal_memory(self):
        detector = YellowPathDetector()
        self.assertTrue(detector.process(scene(), timestamp=1.).valid)
        empty = np.full((480, 640, 3), 35, np.uint8)
        result = detector.process(empty, timestamp=1.05)
        self.assertFalse(result.valid)
        self.assertEqual(cv2.countNonZero(result.path_mask), 0)

    def test_reset_and_stale_history_do_not_pull_new_path_to_old_position(self):
        detector = YellowPathDetector()
        detector.process(scene(x=200), timestamp=1.)
        result = detector.process(scene(x=430), timestamp=2.)
        self.assertTrue(result.valid)
        self.assertLess(abs(np.median(result.center[:, 0]) - 430), 3)
        detector.reset(anchor=.8)
        self.assertIsNone(detector.previous)
        self.assertAlmostEqual(detector.anchor, .8)

    def test_roi_excludes_upper_image_and_fit_does_not_extrapolate(self):
        frame = scene()
        result = YellowPathDetector().process(frame)
        self.assertEqual(cv2.countNonZero(result.path_mask[:145]), 0)
        self.assertEqual(cv2.countNonZero(result.raw_mask[:144]), 0)

    def test_rendering_marks_reconstruction_and_preserves_input(self):
        frame = scene()
        frame[280:304] = 35
        original = frame.copy()
        settings = PathSettings(dashed=False)
        result = YellowPathDetector().process(frame, settings)
        output = render_path(frame, result, settings)
        np.testing.assert_array_equal(frame, original)
        # Afastado da linha central branca: verde observado e laranja estimado.
        self.assertGreater(output[350, 325, 1], output[350, 325, 2])
        self.assertGreater(output[292, 325, 2], output[292, 325, 1])

    def test_color_distractors_do_not_become_paths(self):
        for color in ((240, 240, 240), (255, 0, 0), (0, 220, 0), (0, 0, 240)):
            frame = np.full((480, 640, 3), 35, np.uint8)
            cv2.line(frame, (320, 160), (320, 479), color, 16)
            self.assertFalse(YellowPathDetector().process(frame).valid)

    def test_random_color_texture_is_not_a_path(self):
        for seed in (1, 2, 7):
            with self.subTest(seed=seed):
                frame = np.random.default_rng(seed).integers(0, 256, (480, 640, 3), dtype=np.uint8)
                self.assertFalse(YellowPathDetector().process(frame).valid)

    def test_dashes_form_one_continuous_center_and_blue_mask(self):
        result = YellowPathDetector().process(dashed_scene())
        self.assertTrue(result.valid)
        self.assertGreater(np.ptp(result.center[:, 1]), 290)
        np.testing.assert_array_equal(np.diff(result.center[:, 1]), 1)
        for y in (240, 360):
            self.assertEqual(result.path_mask[y, 320], 255)
            self.assertEqual(result.observed_mask[y, 320], 0)
            self.assertEqual(result.dash_mask[y, 320], 255)
        self.assertEqual(result.dash_mask[300, 320], 0)

    def test_curved_dashes_are_interpolated_within_current_endpoints(self):
        result = YellowPathDetector().process(dashed_scene(curved=True))
        self.assertTrue(result.valid)
        for y in (240, 360):
            expected = round(320 + .0012 * (y - 460) ** 2)
            self.assertEqual(result.dash_mask[y, expected], 255)
            center_x = result.center[result.center[:, 1] == y, 0][0]
            self.assertLess(abs(center_x - expected), 4)
        self.assertEqual(cv2.countNonZero(result.dash_mask[:144]), 0)

    def test_sparse_dashes_can_have_less_than_half_the_corridor_painted(self):
        result = YellowPathDetector().process(dashed_scene(sparse=True))
        self.assertTrue(result.valid)
        self.assertLess(result.support, .5)
        self.assertGreater(result.support, .28)
        self.assertGreater(cv2.countNonZero(result.dash_mask), 0)

    def test_dashed_mode_can_be_disabled_or_limited(self):
        frame = dashed_scene()
        for settings in (PathSettings(dashed=False), PathSettings(dash_gap=40)):
            with self.subTest(settings=settings):
                result = YellowPathDetector().process(frame, settings)
                self.assertFalse(result.valid)
                self.assertEqual(cv2.countNonZero(result.dash_mask), 0)

    def test_displaced_or_differently_oriented_dashes_are_not_joined(self):
        for kind in ("displaced", "opposite_tangent"):
            frame = np.full((480, 640, 3), 35, np.uint8)
            cv2.line(frame, (320, 390), (320, 479), (0, 220, 240), 16)
            if kind == "displaced":
                cv2.line(frame, (430, 180), (430, 260), (0, 220, 240), 16)
            else:
                cv2.line(frame, (215, 180), (320, 260), (0, 220, 240), 16)
            result = YellowPathDetector().process(frame, PathSettings(dash_gap=160))
            with self.subTest(kind=kind):
                self.assertFalse(result.valid)

    def test_blue_is_rendered_and_does_not_modify_the_camera_frame(self):
        frame = dashed_scene()
        original = frame.copy()
        settings = PathSettings()
        result = YellowPathDetector().process(frame, settings)
        output = render_path(frame, result, settings)
        self.assertGreater(output[240, 325, 0], output[240, 325, 1])
        self.assertGreater(output[240, 325, 0], output[240, 325, 2])
        np.testing.assert_array_equal(frame, original)

    def test_dash_history_never_draws_a_bridge_on_an_empty_frame(self):
        detector = YellowPathDetector()
        self.assertTrue(detector.process(dashed_scene(), timestamp=1.).valid)
        result = detector.process(np.full((480, 640, 3), 35, np.uint8), timestamp=1.05)
        self.assertFalse(result.valid)
        self.assertEqual(cv2.countNonZero(result.dash_mask), 0)


class FakeCapture:
    def __init__(self, color):
        self.frame = np.full((60, 80, 3), color, np.uint8)
        self.released = False

    def read(self):
        time.sleep(.003)
        return (False, None) if self.released else (True, self.frame.copy())

    def release(self):
        self.released = True


class CaptureTests(unittest.TestCase):
    def test_switch_stop_and_shutdown_release_camera_and_tag_new_frames(self):
        captures = []

        def fake_open(source, *_):
            capture = FakeCapture(source)
            captures.append(capture)
            return capture, capture.frame.copy()

        worker = CaptureThread(640, 480, 30)
        with patch('detectar_linha_amarela_hsv.open_camera', side_effect=fake_open):
            worker.start()
            try:
                worker.request('connect', 1, 20)
                self.assertEqual(worker.events.get(timeout=2)[:2], (1, 'ready'))
                worker.request('connect', 2, 80)
                self.assertEqual(worker.events.get(timeout=2)[:2], (2, 'ready'))
                self.assertTrue(captures[0].released)
                deadline = time.monotonic() + 2
                packet = None
                while time.monotonic() < deadline:
                    packet = worker.take_frame()
                    if packet is not None and packet[0] == 2:
                        break
                    time.sleep(.005)
                self.assertIsNotNone(packet)
                self.assertEqual(packet[0], 2)
                self.assertEqual(int(packet[1][0, 0, 0]), 80)
                worker.request('stop', 3)
                self.assertEqual(worker.events.get(timeout=2)[:2], (3, 'stopped'))
                self.assertTrue(captures[-1].released)
                self.assertIsNone(worker.take_frame())
            finally:
                worker.request('quit', 4)
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def test_failed_open_emits_error_and_can_retry(self):
        capture = FakeCapture(30)
        worker = CaptureThread(640, 480, 30)
        with patch('detectar_linha_amarela_hsv.open_camera',
                   side_effect=[RuntimeError('camera ausente'), (capture, capture.frame)]):
            worker.start()
            try:
                worker.request('connect', 1, 0)
                self.assertEqual(worker.events.get(timeout=2)[:2], (1, 'error'))
                worker.request('connect', 2, 1)
                self.assertEqual(worker.events.get(timeout=2)[:2], (2, 'ready'))
            finally:
                worker.request('quit', 3)
                worker.join(timeout=2)
            self.assertTrue(capture.released)

    def test_backend_fallback_releases_failed_capture(self):
        failed, successful = FakeCapture(0), FakeCapture(80)
        failed.isOpened = lambda: False
        successful.isOpened = lambda: True
        successful.set = lambda *_: True
        with patch('detectar_linha_amarela_hsv.sys.platform', 'linux'), \
                patch('detectar_linha_amarela_hsv.cv2.VideoCapture', side_effect=[failed, successful]):
            cap, frame = open_camera(0, 640, 480, 30)
        self.assertTrue(failed.released)
        self.assertIs(cap, successful)
        self.assertEqual(frame.shape, (60, 80, 3))
        cap.release()


if __name__ == '__main__':
    unittest.main()
