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


class SteeringDelayTests(unittest.TestCase):
    """Ensaio de 26/11: esterco de lado a lado (~1,7 Hz) com a direcao ~0,18 s atrasada."""
    def test_no_pedaling_with_the_real_steering_delay(self):
        track = pv.build_track(pv.TRACKS['chicane'], 1.3, None, 1., rng=np.random.default_rng(101))
        runs = {}
        for name, changes in (('antes', {'exact_compensation': False, 'steer_delay_s': 0.}), ('agora', {})):
            config = pv.base_config({'sim_gyro': True, 'velocity_reference': 1.5, **changes})
            runs[name] = pv.run(track, config, seed=1, fps=15, latency=(.2, .3),
                                camera_kwargs=dict(pixel_sigma=3., pitch_sigma=6.), steer_response=.9)
        # Antes: guinada de agora x idade da foto, sem prever o atraso -> pedala.
        self.assertGreaterEqual(runs['antes']['steer_reversals'], 10)
        # Agora: historico de poses + preditor do atraso da direcao.
        self.assertTrue(runs['agora']['finished'])
        self.assertLessEqual(runs['agora']['steer_reversals'], 3)
        self.assertEqual(runs['agora']['hits'], 0)


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
        # Grampo, 15 objetos fixos fora da pista, Stanley (padrao). O cenario e
        # caotico (um objeto capturado muda a volta toda): 3 sementes passavam
        # ou falhavam por sorte; 8 sementes e medianas medem a FAIXA. Desde
        # 26/11 o carro simulado tem o atraso real da direcao e o seguidor o
        # preditor dele. Com a faixa: erro mediano ~10 cm; sem ela ~50 cm.
        results = {}
        for band in (0., .2):
            runs = []
            for seed in range(1, 9):
                track = pv.build_track(pv.TRACKS['grampo'], 1.8, None, .5, rng=np.random.default_rng(100 + seed))
                # Sem tracado de curva: o erro e medido ao CENTRO, e o tracado desloca de proposito.
                # Sem memoria do tracado: ela atrasa a correcao de salto, e o teste mede a faixa.
                config = pv.base_config({'velocity_reference': 1.0, 'racing_line_m': 0., 'racing_mode': 'antigo',
                                         'path_memory_s': 0.,
                                         'cone_estimator': {'band_m': band, 'band_growth': .1}})
                runs.append(pv.run(track, config, seed=seed, decoys=15))
            results[band] = runs
        with_band = np.median([r['rms_m'] for r in results[.2]])
        self.assertLess(with_band, .12)
        self.assertGreater(np.median([r['rms_m'] for r in results[0.]]), 2*with_band)
        self.assertGreaterEqual(sum(r['finished'] for r in results[.2]), sum(r['finished'] for r in results[0.]))
        self.assertGreaterEqual(sum(r['finished'] for r in results[.2]), 5)
