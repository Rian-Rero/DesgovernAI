"""Visao em processo separado: odometria compartilhada, observacoes e falhas."""
import json
import math
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fva_car.odometry import PoseHistory, SharedPoseHistory
from fva_car.vision_process import VisionProcess

HOMOGRAPHY = [[0, -.005, 2.4], [-.005, 0, 1.6], [0, 0, 1]]


def cone_frame():
    """Corredor reto de cones (mesma cena de test_cones.scene)."""
    frame = np.full((480, 640, 3), 35, np.uint8)
    inverse = np.linalg.inv(np.asarray(HOMOGRAPHY, float))
    for sign in (-1, 1):
        for x in (.4, .8, 1.2, 1.6):
            u, v = np.rint(cv2.perspectiveTransform(np.array([[[x, sign*.5]]]), inverse)[0, 0]).astype(int)
            color = (0, 0, 255) if sign == 1 else (0, 220, 0)
            cv2.fillPoly(frame, [np.array([[u, v-20], [u-7, v], [u+7, v]])], color)
    return frame


class FakeCamera:
    def __init__(self, fail_after=None, **_):
        self.frame, self.count, self.fail_after = cone_frame(), 0, fail_after

    def get_image(self, gray=False):
        time.sleep(.02)
        self.count += 1
        if self.fail_after is not None and self.count > self.fail_after:
            raise RuntimeError('camera desconectada')
        return self.frame.copy()

    def close(self):
        pass


def failing_camera(**_):
    raise RuntimeError('camera nao encontrada')


def read_delta(poses, result):
    result.put(poses.delta(1., 2.))


class SharedPoseHistoryTests(unittest.TestCase):
    def test_matches_local_history_including_ring_wraparound(self):
        local, shared = PoseHistory(maxlen=50), SharedPoseHistory(maxlen=50)
        for k in range(120):
            t = .02*k
            for poses in (local, shared):
                poses.append(t, .3*t, .1*t*t, .2*t)
        shared.append(1., 0., 0., 0.)  # tempo repetido/antigo e ignorado
        self.assertEqual(shared._snapshot(), local._snapshot())
        np.testing.assert_allclose(shared.delta(1.5, 2.2), local.delta(1.5, 2.2))

    def test_child_process_reads_poses_written_by_parent(self):
        context = multiprocessing.get_context('spawn')
        poses, result = SharedPoseHistory(), context.Queue()
        for k in range(101):
            poses.append(.03*k, .5*.03*k, 0., 0.)  # 0,5 m/s em linha reta
        child = context.Process(target=read_delta, args=(poses, result))
        child.start()
        forward, left, yaw = result.get(timeout=20)
        child.join(5)
        self.assertAlmostEqual(forward, .5, places=6)
        self.assertAlmostEqual(left, 0.)
        self.assertAlmostEqual(yaw, 0.)


class VisionProcessTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.mkdtemp()
        self.calibration = Path(folder) / 'camera.json'
        self.calibration.write_text(json.dumps({'image_size': [640, 480], 'homography': HOMOGRAPHY}))

    def start(self, camera_factory, **kwargs):
        data = {'observation': None}
        vision = VisionProcess(data, 'cones', self.calibration, {}, poses=SharedPoseHistory(),
                               camera_factory=camera_factory, **kwargs)
        return data, vision

    def wait(self, condition, timeout=20.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(.05)
        return False

    def test_observations_arrive_from_child_without_detector_masks(self):
        data, vision = self.start(FakeCamera)
        vision.start()
        try:
            self.assertTrue(self.wait(lambda: vision.frames_received >= 3), 'sem quadros do processo')
            observation = data['observation']
            self.assertTrue(vision.is_alive())
            self.assertTrue(observation.valid, observation.reason)
            self.assertAlmostEqual(observation.errors[0], 0., delta=.03)
            self.assertGreaterEqual(observation.sequence, 3)
            self.assertEqual(observation.cone_result.green_mask.size, 0)
            self.assertEqual(observation.source_frame.shape, (480, 640, 3))
            self.assertLess(time.monotonic() - observation.timestamp, 1.)  # mesmo relogio
        finally:
            vision.stop()
        self.assertFalse(vision.process.is_alive())

    def test_camera_failure_is_reported_before_start_returns(self):
        _, vision = self.start(failing_camera)
        with self.assertRaisesRegex(RuntimeError, 'camera nao encontrada'):
            vision.start()
        self.assertFalse(vision.process.is_alive())

    def test_camera_lost_during_run_ends_process_with_reason(self):
        data, vision = self.start(_failing_after_five)
        vision.start()
        try:
            self.assertTrue(self.wait(lambda: not vision.is_alive()), 'processo nao encerrou')
            self.assertIn('Falha de visao', data['observation'].reason)
        finally:
            vision.stop()


def _failing_after_five(**_):
    return FakeCamera(fail_after=5)


if __name__ == '__main__':
    unittest.main()
