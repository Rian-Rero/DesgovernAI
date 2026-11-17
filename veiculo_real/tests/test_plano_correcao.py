"""Etapas do PLANO_CORRECAO.md: versao, compensacao exata do atraso, velocidade pelo
atraso, fita amarela junto com os cones, calibracao pelo notebook e relatorio."""
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fva_car.cones import RealConeDetector  # noqa: E402
from fva_car.lateral import GroundCalibration, LineObservation, RealStanleyFollower  # noqa: E402
from fva_car.version import code_version  # noqa: E402
from test_lateral import fake_car  # noqa: E402
from test_linha_alta import CALIBRATION, floor_with_line  # noqa: E402


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f'{name}_under_test', ROOT/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VersionTests(unittest.TestCase):
    def test_version_changes_with_the_code_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'fva_car').mkdir()
            (root/'tools').mkdir()
            (root/'main.py').write_text('a = 1\n')
            (root/'fva_car'/'x.py').write_text('b = 2\n')
            first = code_version(root)
            self.assertRegex(first, r'^[0-9a-f]{8}$')
            (root/'logs').mkdir()
            (root/'logs'/'car.csv').write_text('t\n1\n')  # dados nao mudam a versao
            self.assertEqual(code_version(root), first)
            (root/'fva_car'/'x.py').write_text('b = 3\n')
            self.assertNotEqual(code_version(root), first)


class ExactCompensationTests(unittest.TestCase):
    settings = dict(velocity_reference=1., curve_min_speed=.2, speed_filter_s=0., steering_law='pure_pursuit')

    def propagate(self, errors, v, w, age):
        follower = RealStanleyFollower(self.settings)
        car = fake_car()
        car.v, car.w = v, w
        observation = LineObservation(10., errors=errors, path_dkappa=0.)
        return follower, follower._propagate(errors, observation, 10. + age, car)

    def test_straight_drive_keeps_the_offset_and_turning_rotates_the_path(self):
        _, ((e, psi, kappa), age) = self.propagate((.1, 0., 0.), 1., 0., .2)
        self.assertAlmostEqual(age, .2)
        self.assertAlmostEqual(e, .1, places=6)
        self.assertAlmostEqual(psi, 0., places=6)
        follower, ((e, psi, _), _) = self.propagate((.1, 0., 0.), 1., .5, .2)
        # O carro girou dth para um lado: o caminho, relativo a ele, gira -dth.
        self.assertAlmostEqual(psi, -follower.steering_sign*.5*.2, places=6)

    def test_compensation_is_capped_and_needs_the_estimator_path(self):
        _, (_, age) = self.propagate((0., 0., 0.), 1., 0., 2.)
        self.assertAlmostEqual(age, .5)
        follower = RealStanleyFollower(self.settings)
        car = fake_car()
        car.w = 0.
        self.assertIsNone(follower._propagate((0., 0., 0.), LineObservation(10., errors=(0., 0., 0.)), 10.2, car))


class LatencySpeedTests(unittest.TestCase):
    def vref(self, age, budget):
        settings = dict(velocity_reference=2., curve_min_speed=.2, speed_filter_s=0.,
                        vision_latency_compensation=False, latency_speed_budget_m=budget, vision_timeout=1.)
        follower = RealStanleyFollower(settings)
        car = fake_car()
        follower.apply(car, LineObservation(10., errors=(0., 0., 0.), path_dkappa=0.), 10. + age)
        return car.vref

    def test_old_image_limits_speed(self):
        self.assertAlmostEqual(self.vref(.3, .3), 1., places=3)   # 0,3 m / 0,3 s
        self.assertAlmostEqual(self.vref(.1, .3), 2., places=3)   # imagem nova: livre
        self.assertAlmostEqual(self.vref(.3, 0.), 2., places=3)   # desligado


class LineFusionTests(unittest.TestCase):
    def test_tape_points_reach_the_corridor_estimator(self):
        detector = RealConeDetector(CALIBRATION, corridor=True, line_fusion=True)
        frame = floor_with_line(np.array([.1, 0., 0., 0., 1.]))  # fita 10 cm a esquerda, reta
        points = detector._line_points(frame, 10.)
        self.assertTrue(detector.line_seen)
        self.assertGreaterEqual(len(points), 5)
        np.testing.assert_allclose(points[:, 1], .1, atol=.03)
        self.assertIsNone(RealConeDetector(CALIBRATION, corridor=True).line)  # fusao desligada

    def test_no_tape_gives_no_center_points(self):
        detector = RealConeDetector(CALIBRATION, corridor=True, line_fusion=True)
        points = detector._line_points(np.full((480, 640, 3), (150, 156, 158), np.uint8), 10.)
        self.assertFalse(detector.line_seen)
        self.assertEqual(len(points), 0)


class NotebookCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool('calibrar_camera_lateral')
        grid = np.array(self.tool.grid_points((.6, 1., 1.5, 2.), (.5, 0., -.5)), float)
        homography = np.linalg.inv(np.asarray(CALIBRATION.homography, float))  # chao -> pixel
        q = np.c_[grid, np.ones(len(grid))] @ homography.T
        self.pixels, self.ground = q[:, :2]/q[:, 2:3], grid

    def test_grid_order_is_near_to_far_and_left_to_right(self):
        points = self.tool.grid_points()
        self.assertEqual(points[0], (.6, 1.))
        self.assertEqual(points[-1], (3., -1.))
        self.assertEqual(len(points), 30)

    def test_a_wrong_point_stands_out_when_left_out_of_the_fit(self):
        _, left_out = self.tool.reprojection_errors(self.pixels, self.ground)
        self.assertLess(np.nanmax(left_out), .005)
        wrong = self.ground.copy()
        wrong[4] = (wrong[4, 0], wrong[4, 1] + .15)  # medida trocada em 15 cm
        _, left_out = self.tool.reprojection_errors(self.pixels, wrong)
        self.assertEqual(int(np.nanargmax(left_out)), 4)
        self.assertGreater(left_out[4], .06)

    def test_saved_file_loads_in_the_mission(self):
        with tempfile.TemporaryDirectory() as folder:
            data = {'image_size': [640, 480], 'image_points_px': self.pixels.tolist(),
                    'ground_points_m': self.ground.tolist(), 'erro_fora_cm': [0.]*len(self.ground)}
            saved = self.tool.save_calibration(data, Path(folder)/'camera.json')
            calibration = GroundCalibration.from_file(saved)
            np.testing.assert_allclose(calibration.project_pixels(self.pixels), self.ground, atol=1e-6)


class ReportTests(unittest.TestCase):
    def test_single_short_reading_is_reported_as_a_false_echo(self):
        import pandas as pd
        report = load_tool('analisar_ensaio')
        distance = np.r_[np.full(50, 4.), np.full(5, .5), np.full(100, 4.)]
        stop = np.r_[np.zeros(52), np.ones(103)]
        car = pd.DataFrame(dict(t=np.arange(155)*.02, distance_m=distance, stopping_distance_m=.7,
                                obstacle_stop=stop))
        (result,) = report.stops(car)
        self.assertEqual(result['seguidas'], 1)
        self.assertEqual(result['depois'][:2], [.5, 4.])

    def test_startup_section_finds_the_stutter(self):
        import pandas as pd
        report = load_tool('analisar_ensaio')
        t = np.arange(500)*.02
        # Liga em 1 s; 2 s presa a 0,35 m/s com o motor parando, depois sobe a 0,8.
        vref = np.where(t < 1., 0., np.where(t < 3., .35, .8))
        v = np.where(t < 1.2, 0., np.where(t < 3., .25 + .1*np.sign(np.sin(8*t)), np.minimum(.8, .2 + (t - 3.))))
        v[(t > 1.6) & (t < 1.7)] = 0.  # motor morreu
        car = pd.DataFrame(dict(t=t, v=v, vref=vref, lateral_active=(t >= 1.).astype(int),
                                steer_cmd_rad=np.where(t < 1.2, .3, 0.)))
        start = report.startup(car, [])
        self.assertAlmostEqual(start['alvo'], .8)
        self.assertAlmostEqual(start['ate_80_s'], 2.44, delta=.03)  # 0,64 m/s em t = 3,44 s
        self.assertAlmostEqual(start['lenta_s'], 2., delta=.05)
        self.assertEqual(start['motor_parou'], 1)
        self.assertGreater(start['esterco_parado_graus'], 15.)
        self.assertIsNone(report.startup(car.assign(lateral_active=0), []))

    def test_report_on_a_synthetic_run(self):
        import json
        import pandas as pd
        report = load_tool('analisar_ensaio')
        n = 500  # 10 s a 50 Hz, quadro novo a cada 3 passos (~17 q/s), imagem com 0,1 s
        car = pd.DataFrame(dict(
            t=np.arange(n)*.02, x=np.arange(n)*.02*.7, y=0., v=.7, vref=.7, lateral_active=1,
            vision_frame_id=np.arange(n)//3, vision_age_s=.1, vision_capture_s=.01, vision_detection_s=.03,
            vision_geometry_s=.005, distance_m=4., stopping_distance_m=.7, obstacle_stop=0))
        records = [dict(frame=k, t_capture=k*.06, control=dict(active=True, steering_law='pure_pursuit'),
                        vision=dict(corridor=dict(state='AMBAS', valid=True, errors=[.02, .01, 0.],
                                                  sides=dict(flipped=False))))
                   for k in range(n//3)]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/'20261111_100000'
            run.mkdir()
            car.to_csv(run/'car.csv', index=False)
            (run/'cones_debug.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            (run/'cones_summary.json').write_text(json.dumps({'code_version': 'abcd1234'}))
            text = report.report(run)
            self.assertTrue((run/'relatorio.txt').exists())
        self.assertIn('abcd1234', text)
        self.assertIn('OK      visao: 16.7 quadros/s', text)
        self.assertIn('OK      erro lateral (estimador): 2.0 cm', text)
        self.assertNotIn('ATENCAO', text)
        self.assertIn('partida: 80% de 0.70 m/s em 0.0 s', text)


if __name__ == '__main__':
    unittest.main()
