"""Pista virtual (tools/pista_virtual.py): estimador + seguidor em malha fechada."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pista_virtual as pv  # noqa: E402


class VirtualTrackTests(unittest.TestCase):
    def test_short_track_is_completed_near_the_center(self):
        segments = [('reta', 3.), ('curva', 2.5, 90), ('reta', 4.)]
        for law in ('pure_pursuit', 'stanley'):
            with self.subTest(law=law):
                track = pv.build_track(segments, 1.8, rng=np.random.default_rng(1))
                result = pv.run(track, pv.base_config({'steering_law': law}), seed=1)
                self.assertTrue(result['finished'])
                self.assertLess(result['rms_m'], .1)
                self.assertEqual(result['hits'], 0)
                self.assertAlmostEqual(result['width_learned'], 1.8, delta=.1)

    def test_track_cones_follow_the_width_profile(self):
        track = pv.build_track([('reta', 6.)], 1.8, 1.0, cone_noise=0.)
        widths = track.left[:, 1] - track.right[:, 1]
        self.assertAlmostEqual(widths[0], 1.8, places=2)
        self.assertLess(widths[-1], 1.1)


if __name__ == '__main__':
    unittest.main()
