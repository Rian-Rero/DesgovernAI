"""Pista virtual (tools/pista_virtual.py): estimador + seguidor em malha fechada."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pista_virtual as pv  # noqa: E402

# A camera simulada usa uma calibracao FIXA (a de 19/11, origem no eixo dianteiro):
# a do repositorio muda a cada recalibracao e mudava os limites destes testes.
SIM_CALIBRATION = Path(__file__).resolve().parent/'dados'/'camera_pista_virtual.json'


def setUpModule():
    global _calibration
    _calibration, pv.CALIBRATION = pv.CALIBRATION, SIM_CALIBRATION


def tearDownModule():
    pv.CALIBRATION = _calibration


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


class DecoyTests(unittest.TestCase):
    def test_objects_sit_outside_the_track(self):
        track = pv.build_track([('reta', 6.), ('curva', 2., 90)], 1.2, rng=np.random.default_rng(3))
        pv.add_decoys(track, 20, np.random.default_rng(4))
        self.assertEqual(track.decoys.shape, (20, 2))
        for point in track.decoys:
            error, _, half = pv.track_error(track, point, 0)
            self.assertGreaterEqual(abs(error), half + .2 - 1e-6)
            self.assertLessEqual(abs(error), half + .8 + 1e-6)

    def test_band_keeps_the_car_centered_with_objects_beside_the_track(self):
        # Grampo, 15 objetos fixos fora da pista, Stanley (padrao), 3 sementes: sem
        # a faixa o carro se desvia (media ~30 cm RMS, toques); com ela ~9 cm.
        # Uma semente so variava demais com a calibracao da camera (que muda).
        results = {}
        for band in (0., .2):
            runs = []
            for seed in (1, 2, 3):
                track = pv.build_track(pv.TRACKS['grampo'], 1.8, None, .5, rng=np.random.default_rng(100 + seed))
                config = pv.base_config({'velocity_reference': 1.0,
                                         'cone_estimator': {'band_m': band, 'band_growth': .1}})
                runs.append(pv.run(track, config, seed=seed, decoys=15))
            results[band] = runs
        self.assertTrue(all(r['finished'] for r in results[.2]))
        self.assertEqual(sum(r['hits'] for r in results[.2]), 0)
        with_band = np.mean([r['rms_m'] for r in results[.2]])
        self.assertLess(with_band, .12)
        self.assertGreater(np.mean([r['rms_m'] for r in results[0.]]), 2*with_band)
