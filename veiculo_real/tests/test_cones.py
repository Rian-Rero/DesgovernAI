"""Percepcao de cones e integracao, sem acionar hardware."""
import math
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import CONE_RECOVERY_SPEED, RealConeDetector, ConeObservation
from fva_car.lateral import GroundCalibration, RealStanleyFollower, LineObservation
from test_lateral import fake_car
from test_mission import mission


def scene(bend=0., offset=0., reverse=False):
    calibration = GroundCalibration((640, 480), [[0, -.005, 2.4], [-.005, 0, 1.6], [0, 0, 1]])
    frame = np.full((480, 640, 3), 35, np.uint8)
    inverse = np.linalg.inv(calibration.homography)
    for sign in (-1, 1):
        for x in (.4, .8, 1.2, 1.6):
            y = offset+bend*x*x+sign*.5
            pixel = cv2.perspectiveTransform(np.array([[[x, y]]]), inverse)[0, 0]
            u, v = np.rint(pixel).astype(int)
            red = (sign == 1) != reverse
            color = (0, 0, 255) if red else (0, 220, 0)
            cv2.fillPoly(frame, [np.array([[u, v-20], [u-7, v], [u+7, v]])], color)
    return calibration, frame


class ConeTests(unittest.TestCase):
    def test_camera_metric_center_both_directions_and_curves(self):
        for bend in (0., -.15, .15):
            for reverse in (False, True):
                calibration, frame = scene(bend=bend, offset=.08, reverse=reverse)
                detector = RealConeDetector(calibration)
                observation = detector.observe_frame(frame, 10.)
                self.assertTrue(observation.valid, observation.reason)
                self.assertAlmostEqual(observation.errors[0], .08, delta=.02)
                self.assertAlmostEqual(observation.errors[1], 0., delta=.04)
                self.assertAlmostEqual(observation.errors[2], 2*bend, delta=.04)
                self.assertEqual(len(observation.red_pixels), 4)
                self.assertEqual(len(observation.green_pixels), 4)
                self.assertIsNone(observation.frame)
                self.assertEqual(detector.with_preview(observation).frame.shape, frame.shape)

    def test_corridor_estimator_drives_stanley_errors(self):
        calibration, frame = scene(offset=.08)
        odometry = Mock()
        odometry.delta.return_value = (0., 0., 0.)
        detector = RealConeDetector(calibration, settings={'track_width_cm': 100, 'tracking': False},
                                    corridor=True, odometry=odometry)
        for k in range(4):
            observation = detector.observe_frame(frame, 10. + .1*k)
        self.assertTrue(observation.valid, observation.reason)
        self.assertAlmostEqual(observation.errors[0], .08, delta=.03)
        self.assertAlmostEqual(observation.errors[1], 0., delta=.05)
        self.assertEqual(observation.debug['corridor']['state'], 'AMBAS')
        self.assertEqual(observation.debug['corridor']['used'], [4, 4])
        self.assertIsNotNone(observation.debug['legacy_errors'])
        self.assertEqual(observation.speed_factor, 1.)
        self.assertAlmostEqual(observation.corridor_width_m, 1., delta=.08)
        self.assertEqual(detector.with_preview(observation).frame.shape, frame.shape)
        # A trajetoria valida mantem os limites de velocidade em quadros vazios.
        empty = np.full_like(frame, 35)
        odometry.delta.return_value = (.02, 0., 0.)
        for k in range(5):
            remembered = detector.observe_frame(empty, 10.5 + .1*k)
            self.assertTrue(remembered.valid, remembered.reason)
            self.assertEqual(remembered.debug['corridor']['state'], 'MEMORIA')
            self.assertEqual(remembered.speed_factor, observation.speed_factor)
            self.assertEqual(remembered.speed_limit, observation.speed_limit)

    def test_corridor_estimator_rejects_unknown_settings(self):
        calibration, _ = scene()
        with self.assertRaises(ValueError):
            RealConeDetector(calibration, corridor={'nao_existe': 1})
        self.assertIsNone(RealConeDetector(calibration, corridor=False).corridor)

    def test_reduced_processing_preserves_coordinates(self):
        calibration, frame = scene(bend=.1)
        full = RealConeDetector(calibration).observe_frame(frame, 10.)
        reduced = RealConeDetector(calibration, processing_width=320).observe_frame(frame, 10.)
        self.assertTrue(reduced.valid, reduced.reason)
        np.testing.assert_allclose(reduced.errors, full.errors, atol=.035)

    def test_distant_center_uses_local_direction_instead_of_extrapolated_curve(self):
        from fva_car.lateral import fit_line_errors
        x = np.linspace(1.0, 1.8, 17)
        points = np.column_stack((x, .05 + .2*(x-1.4)**2))
        lateral, heading, curvature = fit_line_errors(
            points, [0, 0], [1, 0], with_curvature=True, limit_curve_extrapolation=True)
        self.assertAlmostEqual(lateral, .05, delta=.02)
        self.assertAlmostEqual(heading, 0.)
        self.assertEqual(curvature, 0.)

    def test_far_straight_center_starts_slowly_but_far_curve_does_not(self):
        # Ensaio parado (logs 20261023_080354): o primeiro portao ficava a 1,9-2,3 m.
        from tools.cone_path import ConePathResult
        calibration, frame = scene()
        detector = RealConeDetector(calibration, settings={'single_border_recovery': True})
        x = np.linspace(2.0, 2.7, 15)
        mask = np.zeros((480, 640), np.uint8)
        for curvature, expected in ((.05, True), (.6, False)):
            result = ConePathResult(mask, mask, mask, center=np.array([[320, 300]], np.int32),
                                    source='solo', ground_center=np.column_stack((x, .03 + 0*x)),
                                    curvature_m_inv=curvature, status='Centro pelos portoes de cones (m)')
            with patch.object(detector.detector, 'process', return_value=result):
                observation = detector.observe_frame(frame, 10.)
            self.assertEqual(observation.valid, expected, observation.reason)
            if expected:
                self.assertAlmostEqual(observation.errors[0], .03, delta=.01)
                self.assertEqual(observation.speed_limit, CONE_RECOVERY_SPEED)

    def test_gate_midpoints_skip_diagonal_pairs_at_end_of_border(self):
        from tools.cone_track_path import gate_midpoints
        left = [[1.0, .5], [1.5, .5]]
        right = [[.4, -.5], [1.0, -.5], [1.5, -.5]]
        midpoints, widths = gate_midpoints(left, right)
        np.testing.assert_allclose(midpoints, [[1., 0.], [1.5, 0.]])
        np.testing.assert_allclose(widths, [1., 1.])

    def test_one_border_and_camera_fault_do_not_authorize_motion(self):
        calibration, frame = scene()
        frame[:, :320] = 35  # elimina uma borda
        detector = RealConeDetector(calibration)
        for image in (frame, None, np.zeros((240, 320, 3), np.uint8)):
            observation = detector.observe_frame(image, 10.)
            self.assertFalse(observation.valid)
            car = fake_car()
            follower = RealStanleyFollower({'line_loss_frames': 0, 'line_loss_tolerance': 0})
            self.assertFalse(follower.apply(car, observation, 10.))
            self.assertEqual(car.vref, 0.)
            self.assertEqual(car.st, 0.)

    def test_recovery_steers_away_from_near_border_regardless_of_color(self):
        for side in (-1, 1):
            for reverse in (False, True):
                with self.subTest(side=side, reverse=reverse):
                    calibration, frame = scene(offset=-side*.25, reverse=reverse)
                    if side > 0:
                        frame[:, 320:] = 35
                    else:
                        frame[:, :320] = 35
                    detector = RealConeDetector(calibration, mission.parameters['cone_settings'])
                    observation = detector.observe_frame(frame, 10.)
                    self.assertTrue(observation.valid, observation.reason)
                    self.assertTrue(observation.debug['recovery'])
                    self.assertLess(side*observation.errors[0], 0.)
                    car = fake_car()
                    follower = RealStanleyFollower({'velocity_reference': .7})
                    self.assertTrue(follower.apply(car, observation, 10.))
                    self.assertLess(side*car.st, 0.)
                    self.assertLessEqual(car.vref, CONE_RECOVERY_SPEED)
                    self.assertTrue(follower.apply(car, LineObservation(10.05), 10.05))
                    self.assertLessEqual(car.vref, CONE_RECOVERY_SPEED)
                    self.assertFalse(follower.apply(car, LineObservation(11.1), 11.1))

    def test_recovery_prefers_the_measured_center_with_both_borders(self):
        calibration, frame = scene(offset=.08)
        detector = RealConeDetector(calibration, mission.parameters['cone_settings'])
        observation = detector.observe_frame(frame, 10.)
        self.assertTrue(observation.valid, observation.reason)
        self.assertFalse(observation.debug['recovery'])
        self.assertAlmostEqual(observation.errors[0], .08, delta=.02)
        self.assertTrue(math.isinf(observation.speed_limit))

    def test_recovery_requires_persistence_for_an_isolated_cone(self):
        calibration, frame = scene(offset=-.25)
        frame[:, 320:] = 35
        frame[:290] = 35
        frame[335:] = 35
        detector = RealConeDetector(calibration, mission.parameters['cone_settings'])
        self.assertFalse(detector.observe_frame(frame, 10.).valid)
        observation = detector.observe_frame(frame, 10.1)
        self.assertTrue(observation.valid, observation.reason)
        self.assertTrue(observation.debug['gates']['recovery_single_cone'])
        self.assertLess(observation.errors[0], 0.)
        self.assertEqual(observation.errors[2], 0.)

    def test_adapter_uses_shared_hsv_detector_and_metric_geometry(self):
        from tools.cone_path import ConePathDetector, ConeSettings
        from fva_car.lateral import fit_line_errors
        calibration, frame = scene(bend=.1, reverse=True)
        detector = RealConeDetector(calibration)
        observation = detector.observe_frame(frame, 10.)
        expected = ConePathDetector(calibration).process(frame, ConeSettings(roi_top=0), timestamp=10.)
        self.assertTrue(expected.valid, expected.status)
        np.testing.assert_array_equal(observation.cone_result.green_mask, expected.green_mask)
        np.testing.assert_allclose(observation.cone_result.ground_center, expected.ground_center)
        np.testing.assert_allclose(observation.errors,
            fit_line_errors(expected.ground_center, [0, 0], [1, 0], with_curvature=True,
                            limit_curve_extrapolation=True))
        self.assertEqual(detector.settings.roi_top, 0)
        self.assertEqual(mission.parameters['cone_settings']['roi_top'], 0)

    def test_invalid_settings_fail_before_hardware(self):
        calibration, _ = scene()
        for settings in ({'red_hsv_ranges': [[[180, 0, 0], [255, 255, 255]]]},
                         {'roi_top': -1}, {'min_width_cm': 200, 'max_width_cm': 100}):
            with self.assertRaises(ValueError):
                RealConeDetector(calibration, settings)

    def test_cli_selects_cones(self):
        with patch.object(sys, 'argv', ['main.py', '--navigation', 'cones']), patch.object(mission, 'run_mission') as run:
            mission.main()
        self.assertEqual(run.call_args.args[0]['navigation_mode'], 'cones')


    def test_invalid_frame_is_processed_once(self):
        from tools.cone_path import ConePathDetector
        calibration, frame = scene()
        frame[:, :320] = 35
        detector = RealConeDetector(calibration)
        with patch.object(ConePathDetector, 'process', autospec=True,
                          side_effect=ConePathDetector.process) as process:
            observation = detector.observe_frame(frame, 10.)
        self.assertFalse(observation.valid)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(observation.debug['valid'], False)

    def test_swapped_colors_valid_on_first_frame_with_debug(self):
        calibration, frame = scene()  # vermelho a esquerda (+y)
        detector = RealConeDetector(calibration, {'green_left': True})
        observation = detector.observe_frame(frame, 10.)
        self.assertTrue(observation.valid, observation.reason)
        self.assertFalse(detector.green_left)
        self.assertFalse(observation.cone_settings.green_left)
        self.assertEqual(len(observation.debug['stanley_errors']), 3)
        self.assertIn('trajectory', observation.debug)

    def test_cone_loss_keeps_trajectory_with_speed_ramp_then_stops(self):
        settings = {'line_loss_frames': 4, 'line_loss_tolerance': 1., 'loss_speed_ramp': True,
                    'line_search_rate_deg_s': 0., 'curve_min_speed': .15, 'velocity_reference': .7,
                    'speed_filter_s': 0}
        car = fake_car()
        follower = RealStanleyFollower(settings)
        valid = lambda t: LineObservation(timestamp=t, errors=(.05, 0., 0.))
        for step in range(10):
            follower.apply(car, valid(10.+.01*step), 10.+.01*step)
        cruise = car.vref
        self.assertGreater(cruise, .5)
        speeds = []
        for frame in range(1, 5):
            now = 10.1+.05*frame
            self.assertTrue(follower.apply(car, LineObservation(timestamp=now, reason='sem cones'), now))
            self.assertTrue(follower.last_state['hold'])
            speeds.append(car.vref)
        # 1 quadro perdido nao derruba a vref (loss_ramp_grace_s, 26/11: o
        # carro engasgava a cada quadro descartado); depois desce ate o minimo.
        self.assertAlmostEqual(speeds[0], cruise, delta=.01)
        self.assertTrue(all(a > b for a, b in zip(speeds, speeds[1:])), speeds)
        self.assertAlmostEqual(speeds[-1], .15, delta=1e-6)
        self.assertIn('vref x', follower.status)
        self.assertEqual(car.st, follower.last_state['steer_rad'])
        # Esgotada a tolerancia: neutro e vref zerada.
        self.assertFalse(follower.apply(car, LineObservation(timestamp=10.4, reason='sem cones'), 10.4))
        self.assertEqual(car.vref, 0.)
        self.assertFalse(follower.last_state['active'])

    def test_cone_mode_keeps_curve_speed_during_hold(self):
        settings = mission.follower_config({'velocity_reference': 1.5, 'speed_filter_s': 0}, 'cones', None)
        follower = RealStanleyFollower(settings)
        car = fake_car()
        curvature = .8
        car.v = 0.
        self.assertTrue(follower.apply(car, LineObservation(10., errors=(0., 0., curvature)), 10.))
        curve_speed = math.sqrt(follower.lateral_accel / curvature)
        self.assertAlmostEqual(car.vref, curve_speed)
        for frame in range(1, follower.loss_frames + 1):
            now = 10. + .05*frame
            self.assertTrue(follower.apply(car, LineObservation(now), now))
            self.assertTrue(follower.last_state['hold'])
            self.assertEqual(follower.last_state['hold_speed_factor'], 1.)
            self.assertAlmostEqual(car.vref, curve_speed)
        follower.external_speed_limit = .25
        self.assertTrue(follower.apply(car, LineObservation(now), now + .01))
        self.assertAlmostEqual(car.vref, .25)
        self.assertFalse(follower.apply(car, LineObservation(now + .05), now + .05))
        self.assertEqual(car.vref, 0.)

    def test_line_mode_keeps_full_speed_during_hold(self):
        car = fake_car()
        follower = RealStanleyFollower({'line_loss_frames': 4, 'speed_filter_s': 0, 'velocity_reference': .7})
        for step in range(10):
            follower.apply(car, LineObservation(timestamp=10.+.01*step, errors=(.05, 0., 0.)), 10.+.01*step)
        cruise = car.vref
        follower.apply(car, LineObservation(timestamp=10.2, reason='sem faixa'), 10.2)
        self.assertAlmostEqual(car.vref, cruise, delta=.02)


if __name__ == '__main__':
    unittest.main()


class CorridorSideTests(unittest.TestCase):
    """Lado de cada cor para o estimador, pelos proprios cones (ensaio de 10/11:
    vermelho a esquerda, detector sem cadeias ficou no padrao verde a esquerda)."""

    def detector(self):
        from fva_car.lateral import GroundCalibration
        from fva_car.cones import RealConeDetector
        calibration = GroundCalibration.from_file(Path(__file__).resolve().parent/'dados'/'camera_pista_virtual.json')
        return RealConeDetector(calibration, corridor=True)

    def test_side_evidence_pairs_cones_at_the_same_distance(self):
        from fva_car.cones import side_evidence
        green = np.array([[.9, -.48], [1.7, -.5], [3.7, -.54]])
        red = np.array([[1.8, .86], [2.7, .92]])
        separation, paired = side_evidence(green, red)
        self.assertTrue(paired)
        self.assertLess(separation, -1.)  # verde a direita
        # Curva a esquerda: o vermelho distante fica "a direita" do verde proximo,
        # mas a mesma distancia o verde continua a esquerda.
        green = np.array([[1., .7], [2., 1.4]])
        red = np.array([[1.1, -.6], [2.1, .2], [2.6, .9]])
        self.assertGreater(side_evidence(green, red)[0], .3)
        self.assertEqual(side_evidence(green, np.empty((0, 2))), (None, False))

    def test_estimator_uses_the_side_seen_in_the_cones_and_flips_with_hysteresis(self):
        detector = self.detector()
        green, red = np.array([[.9, -.48], [1.7, -.5]]), np.array([[1.8, .86], [2.7, .92]])
        side, info = detector._corridor_sides(green, red, fallback=True)
        self.assertFalse(side)  # o detector dizia verde a esquerda; os cones dizem o contrario
        self.assertTrue(info['flipped'])
        self.assertEqual(info['source'], 'pares')
        # Um quadro contrario isolado (reflexo) nao troca; dois seguidos trocam.
        self.assertFalse(detector._corridor_sides(red, green, fallback=True)[0])
        self.assertTrue(detector._corridor_sides(red, green, fallback=True)[0])
        # Uma cor so: mantem o lado decidido.
        self.assertTrue(detector._corridor_sides(red, np.empty((0, 2)), fallback=False)[0])


class HairpinSideTests(unittest.TestCase):
    """Ensaios de 10/11: num grampo a esquerda (raio ~1,3 m) a borda de fora da a
    volta pela frente; comparar cones a mesma distancia trocava o lado e o carro
    saia da pista. Com o corredor estimado, o lado vem da linha central."""

    def test_tight_curve_does_not_flip_the_sides(self):
        from fva_car.cones import SideTracker, side_evidence
        from fva_car.corridor import CorridorEstimator, CorridorSettings, sample_centerline
        width = 1.3
        state = np.array([0., 0., .75, 0., width])  # curva a esquerda, raio 1,33 m
        arcs = np.arange(0., 4., .4)  # um cone a cada 40 cm de caminho
        left = sample_centerline(state, arcs, width/2, max_curvature=5.)
        right = sample_centerline(state, arcs, -width/2, max_curvature=5.)
        keep = lambda p: p[(p[:, 0] > .6) & (p[:, 0] < 3.)]
        red, green = keep(left), keep(right)  # vermelho a esquerda (como na pista de 10/11)
        # A mesma distancia a frente, o verde (fora) aparece "a esquerda" do vermelho.
        self.assertGreater(side_evidence(green, red)[0], .3)
        # Lado decidido antes (na reta) e corredor ja acompanhando a curva, como no ensaio.
        estimator = CorridorEstimator(CorridorSettings(track_width_m=width))
        for k in range(5):
            estimator.step(.1*k, red, green, (0., 0., 0.) if k else None)
        tracker = SideTracker()
        tracker.green_left = False
        for k in range(5, 20):
            green_left, info, flipped = tracker.update(green, red, estimator, fallback=False)
            self.assertFalse(green_left, info)
            self.assertFalse(flipped, info)
            left_pts, right_pts = (green, red) if green_left else (red, green)
            estimator.step(.1*k, left_pts, right_pts, (0., 0., 0.))
        self.assertEqual(info['source'], 'corredor')
        self.assertLess(info['separation_m'], -.5)  # no corredor, verde bem a direita (verde - vermelho < 0)

    def test_wrong_start_is_still_corrected_from_the_corridor(self):
        from fva_car.cones import SIDE_FLIP_FRAMES_TRACKING, SideTracker
        from fva_car.corridor import CorridorEstimator, CorridorSettings
        green = np.array([[1., -.65], [1.5, -.65], [2., -.65], [2.5, -.65]])  # verde a direita, reta
        red = np.array([[1., .65], [1.5, .65], [2., .65], [2.5, .65]])
        estimator = CorridorEstimator(CorridorSettings(track_width_m=1.3))
        estimator.step(0., green, red, None)  # comecou com as cores trocadas
        tracker = SideTracker()
        tracker.green_left = True
        flips = [tracker.update(green, red, estimator, fallback=True)[2] for _ in range(SIDE_FLIP_FRAMES_TRACKING)]
        self.assertEqual(flips, [False]*(SIDE_FLIP_FRAMES_TRACKING-1) + [True])
        self.assertFalse(tracker.green_left)


class RoundaboutSideTests(unittest.TestCase):
    """Rotatoria (24/11): cones de outras vias nao trocam as cores; troca real nao para o carro."""
    def corridor(self):
        from fva_car.corridor import CorridorEstimator, CorridorSettings
        green = np.array([[1., .65], [1.5, .65], [2., .65], [2.5, .65]])
        red = np.array([[1., -.65], [1.5, -.65], [2., -.65], [2.5, -.65]])
        estimator = CorridorEstimator(CorridorSettings(track_width_m=1.3))
        for k in range(5):
            estimator.step(.1*k, green, red, (0., 0., 0.) if k else None)
        return estimator, green, red

    def test_cones_of_another_road_seen_through_a_gap_do_not_vote(self):
        from fva_car.cones import SIDE_FLIP_FRAMES_TRACKING, SideTracker, path_separation
        estimator, _, red = self.corridor()
        # Verdes da via de saida, BEM fora do anel, a direita (~1,6 m do centro).
        far_green = np.array([[1.5, -1.6], [2., -1.7], [2.5, -1.6]])
        self.assertIsNone(path_separation(far_green, red, estimator.x))
        tracker = SideTracker()
        tracker.green_left = True
        flips = [tracker.update(far_green, red, estimator, True)[2] for _ in range(3*SIDE_FLIP_FRAMES_TRACKING)]
        self.assertFalse(any(flips))
        self.assertTrue(tracker.green_left)

    def test_real_color_swap_keeps_the_corridor(self):
        from fva_car.cones import handle_side_flip
        estimator, _, _ = self.corridor()
        before = estimator.x.copy()
        width = estimator.learned_width
        info = {'source': 'corredor'}
        self.assertFalse(handle_side_flip(estimator, info))
        self.assertTrue(info['kept_state'])
        np.testing.assert_allclose(estimator.x, before)       # mesma geometria
        self.assertEqual(estimator.learned_width, width)
        self.assertAlmostEqual(math.sqrt(estimator.P[0, 0]), .3)  # so o deslocamento fica mais incerto
        info = {'source': 'pares'}                              # sem corredor: montado com as cores trocadas
        self.assertTrue(handle_side_flip(estimator, info))
        self.assertIsNone(estimator.x)
        self.assertIsNone(estimator.learned_width)


class VirtualRoundaboutTests(unittest.TestCase):
    def test_no_false_color_flips_in_a_roundabout(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        import pista_virtual as pv
        track = pv.build_track(pv.ROUNDABOUTS['rotatoria'], 1.3, None, 1., rng=np.random.default_rng(100))
        result = pv.run(track, pv.base_config({'steering_law': 'stanley'}), seed=0, fps=10., latency=(.19, .25),
                        color_sides=True, duration=25.)
        self.assertEqual(result['side_flips'], 0)


class FarBaseRefinementTests(unittest.TestCase):
    """Base dos cones longe refeita na imagem cheia (fva_car/cones.py refine_bases)."""
    def setUp(self):
        from fva_car.cones import refine_bases
        self.refine = refine_bases
        self.calibration = GroundCalibration.from_file(
            Path(__file__).resolve().parent/'dados'/'camera_pista_virtual.json')
        self.inverse = np.linalg.inv(self.calibration.homography)
        self.frame = np.full((480, 640, 3), (150, 156, 158), np.uint8)
        self.detector = RealConeDetector(self.calibration, processing_width=320)

    def cone(self, x, y):
        """Cone vermelho com a base em (x, y) m; devolve a base em px da imagem cheia."""
        u, v = cv2.perspectiveTransform(np.array([[[x, y]]]), self.inverse)[0, 0]
        u, v = int(round(u)), int(round(v))
        cv2.fillConvexPoly(self.frame, np.array([(u, v-16), (u-6, v), (u+6, v)], np.int32), (30, 30, 210))
        return np.array([u, v], float)

    def test_far_base_gets_the_full_resolution_row_and_near_ones_stay(self):
        far, near = self.cone(2.4, -.3), self.cone(1., .2)
        reduced = lambda pixel: (pixel + .5)/2 - .5
        # O detector a 320 px entrega a base com 1 linha reduzida (2 px cheios) de erro.
        given = np.array([reduced(far) - [0, 1.], reduced(near) - [0, 1.], [20., 200.]])
        refined, changed = self.refine(self.frame, given, 1, self.detector.detector, self.detector.settings, 2.,
                                       None, self.detector.full_homography)
        self.assertEqual(changed, 1)
        np.testing.assert_allclose(refined[0], reduced(far), atol=.5)  # meio px reduzido = 1 px cheio
        np.testing.assert_allclose(refined[1:], given[1:])  # perto (< 1,3 m) e sem cone: intactos

    def test_detector_uses_it_only_on_reduced_images(self):
        self.assertTrue(self.detector.refine)
        self.assertFalse(RealConeDetector(self.calibration).refine)  # 640: ja e a imagem cheia
        self.assertFalse(RealConeDetector(self.calibration, processing_width=320, refine_bases=False).refine)
