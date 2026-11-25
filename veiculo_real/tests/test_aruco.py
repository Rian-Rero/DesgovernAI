"""Placas ArUco (fva_car/aruco.py): deteccao, PARE perto do cone e integracao com a missao."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fva_car.aruco import ArucoDetector, Marker, SignController  # noqa: E402


def frame_with(marker_id, side=80, at=(300, 180), size=(640, 480)):
    image = np.full((size[1], size[0], 3), 140, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    border = side//5  # borda branca: o detector precisa dela
    patch = np.full((side + 2*border, side + 2*border), 255, np.uint8)
    patch[border:border+side, border:border+side] = marker
    x, y = at
    image[y:y+patch.shape[0], x:x+patch.shape[1]] = patch[:, :, None]
    return image


def seen(t, *markers):
    return SimpleNamespace(timestamp=t, markers=tuple(m.to_tuple() for m in markers))


class DetectorTests(unittest.TestCase):
    def test_finds_id_center_and_size(self):
        markers = ArucoDetector().detect(frame_with(1, side=80, at=(300, 180)))
        self.assertEqual([m.marker_id for m in markers], [1])
        marker = markers[0]
        self.assertAlmostEqual(marker.cx, 300 + 16 + 40, delta=2)
        self.assertAlmostEqual(marker.side_px, 80, delta=3)   # lado vertical
        self.assertEqual(ArucoDetector().detect(np.full((480, 640, 3), 140, np.uint8)), ())

    def test_cone_detector_carries_markers_every_n_frames(self):
        from fva_car.cones import RealConeDetector
        from fva_car.lateral import GroundCalibration
        calibration = GroundCalibration.from_file(Path(__file__).resolve().parent/'dados'/'camera_pista_virtual.json')
        detector = RealConeDetector(calibration, processing_width=320, corridor=True, aruco_every=2)
        frame = frame_with(1)
        first = detector.observe_frame(frame, 1.)
        self.assertEqual(first.markers[0][0], 1)
        self.assertEqual(first.debug['aruco'][0][0], 1)
        second = detector.observe_frame(np.full_like(frame, 140), 1.1)   # sem busca: repete a ultima
        self.assertEqual(second.markers, first.markers)
        self.assertEqual(second.marker_timestamp, first.marker_timestamp)
        self.assertEqual(second.marker_scan, first.marker_scan)
        third = detector.observe_frame(np.full_like(frame, 140), 1.2)
        self.assertEqual(third.markers, ())
        self.assertEqual(third.marker_timestamp, 1.2)
        self.assertEqual(third.marker_scan, first.marker_scan + 1)
        self.assertIsNotNone(detector.preview_bgr(first))
        self.assertIsNone(RealConeDetector(calibration, corridor=True).aruco)
        self.assertTrue(math.isnan(first.markers[0][4]))   # sem o tamanho do marcador: sem posicao
        sized = RealConeDetector(calibration, processing_width=320, corridor=True, aruco_every=1, aruco_marker_m=.1)
        marker = sized.observe_frame(frame, 1.).markers[0]
        self.assertGreater(marker[4], 0.)                    # na frente do carro
        self.assertTrue(math.isfinite(marker[5]) and math.isfinite(marker[6]))


class StopSignTests(unittest.TestCase):
    def test_confirmed_stop_sign_slows_then_stops_close_and_waits_for_the_button(self):
        signs = SignController({1: 'pare'}, confirm_frames=2, approach_speed=.3, stop_gap_m=.08)
        sign = Marker(1, 320, 200, 40, 2., 0., .2)
        pose = (0., 0., 0.)
        self.assertEqual(signs.update(seen(1., sign), 1.).state, 'LIVRE')   # 1 quadro: ainda nao
        command = signs.update(seen(1.1, sign), 1.1, pose, lambda _: pose, .3)
        self.assertEqual(command.state, 'APROXIMANDO')
        self.assertIn('PARE', command.message)
        self.assertEqual(command.speed_limit, .3)
        self.assertTrue(command.in_path)
        self.assertAlmostEqual(command.remaining_m, 2. - .07 - .08)
        # A placa saiu do quadro. Sem eco, a odometria ainda dispara a parada.
        command = signs.update(seen(1.5), 1.5, (1.75, 0., 0.), speed=.3)
        self.assertTrue(command.stop)
        self.assertEqual(command.state, 'PARADO')
        self.assertEqual(signs.update(seen(1.6, sign), 1.6).speed_limit, 0.)
        for t in (5., 30.):  # sem hold_s: fica parado
            self.assertEqual(signs.update(seen(t, sign), t).state, 'PARADO')
        self.assertIn('liberado', signs.release(31.))
        for t in (31.1, 31.2, 31.3):  # a mesma placa ainda a vista: ignorada por ignore_s
            self.assertEqual(signs.update(seen(t, sign), t).speed_limit, math.inf)
        self.assertEqual(signs.update(seen(36., sign), 36.).state, 'LIVRE')

    def test_confirmed_sign_without_geometry_stops_and_honors_hold_time(self):
        signs = SignController({1: 'pare'}, hold_s=3.)
        sign = Marker(1, 320, 200, 40)
        signs.update(seen(0., sign), 0.)
        command = signs.update(seen(.1, sign), .1)
        self.assertTrue(command.stop)
        self.assertTrue(math.isnan(command.remaining_m))
        self.assertEqual(signs.update(seen(2., sign), 2.).state, 'PARADO')
        self.assertIn('liberado', signs.update(seen(3.6, sign), 3.6).message)

    def test_confirmation_requires_fresh_consecutive_scans(self):
        signs = SignController()
        sign = Marker(1, 320, 200, 40)
        def sample(stamp, scan):
            observation = seen(stamp, sign)
            observation.marker_timestamp = stamp
            observation.marker_scan = scan
            return observation

        first = sample(1., 1)
        self.assertEqual(signs.update(first, 1.1).state, 'LIVRE')
        first.timestamp = 1.2  # quadro intermediario reutiliza o ArUco anterior
        self.assertEqual(signs.update(first, 1.2).state, 'LIVRE')
        self.assertEqual(signs.update(sample(1.3, 2), 3.).state, 'LIVRE')
        self.assertEqual(signs.update(sample(3.1, 3), 3.1).state, 'LIVRE')
        self.assertTrue(signs.update(sample(3.3, 4), 3.3).stop)

    def test_other_ids_and_bad_actions(self):
        signs = SignController({1: 'pare', 7: 'ignorar'})
        for t in (0., .1):
            command = signs.update(seen(t, Marker(7, 1, 1, 30), Marker(23, 1, 1, 30)), t)
        self.assertEqual(command.state, 'LIVRE')
        self.assertIn('ArUco 23', command.message)   # avisa uma vez
        with self.assertRaises(ValueError):
            SignController({1: 'voar'})


class SideSignTests(unittest.TestCase):
    """Placa AO LADO: posicao pela imagem, lembrada pela odometria depois que sai da vista."""
    def test_stops_with_the_front_beside_the_sign(self):
        from fva_car.aruco import SignController
        L, overhang = .36, .07
        sign_world = np.array([3.5, .9])   # 3,5 m a frente, 0,9 m a esquerda
        signs = SignController({1: 'pare'}, approach_speed=.3, front_overhang_m=overhang, reaction_s=.45,
                               deceleration=1.75, wheelbase=L, stop_at_m=0.)
        x, v, dt, t = -L, .3, .02, 0.   # eixo traseiro; frente do eixo dianteiro em 0
        history = {}
        stopped_at = None
        rng = np.random.default_rng(0)
        while t < 30.:
            history[round(t, 2)] = (x, 0., 0.)
            front = x + L
            rel = sign_world - [front, 0.]
            visible = rel[0] > .9/math.tan(math.radians(31.5))   # sai do campo (63 graus) a ~1,5 m
            markers = ()
            if visible and round(t/.1) == t/.1 or visible and abs(t/.1 - round(t/.1)) < 1e-6:
                noise = 1. + rng.normal(0., .05)            # tamanho medido com 5% de erro
                markers = (Marker(1, 100., 200., 30., rel[0]*noise, rel[1]*noise, .2),)
            frame_time = round(math.floor(t/.1 + 1e-9)*.1, 2)
            observation = SimpleNamespace(timestamp=frame_time, markers=tuple(m.to_tuple() for m in markers))
            command = signs.update(observation, t, (x, 0., 0.), lambda s: history.get(round(s, 2)), v)
            if command.stop and stopped_at is None:
                signs.stopped(t)
                stopped_at = t
            if stopped_at is not None:      # freia: reacao e depois desaceleracao
                if t - stopped_at > .45:
                    v = max(0., v - 1.75*dt)
            x += v*dt
            t = round(t + dt, 2)
            if v == 0.:
                break
        self.assertIsNotNone(stopped_at)
        front_of_car = x + L + overhang
        self.assertAlmostEqual(front_of_car, sign_world[0], delta=.12)   # frente na altura da placa


class MissionTests(unittest.TestCase):
    """O laco de controle para perto do cone da placa e libera pelo botao."""
    def test_mission_uses_coasting_deceleration_and_marker_size(self):
        import main
        self.assertEqual(main.aruco_every(main.parameters), 2)
        self.assertEqual(main.aruco_every({'aruco': False}), 0)
        signs = main.sign_controller(main.parameters, 'cones')
        self.assertEqual(signs.actions, {1: 'pare'})
        self.assertIsNone(main.sign_controller(main.parameters, 'line'))
        self.assertEqual(signs.deceleration, main.parameters['coast_deceleration'])
        self.assertEqual(main.aruco_marker_m({'aruco': True}), .1)

    def test_command_line(self):
        from unittest.mock import patch
        import main
        with patch.object(sys, 'argv', ['main.py', '--pare-segundos', '3']), patch.object(main, 'run_mission') as run:
            main.main()
        self.assertEqual(run.call_args[0][0]['aruco']['hold_s'], 3.)
        with patch.object(sys, 'argv', ['main.py', '--sem-aruco']), patch.object(main, 'run_mission') as run:
            main.main()
        self.assertIs(run.call_args[0][0]['aruco'], False)


if __name__ == '__main__':
    unittest.main()
