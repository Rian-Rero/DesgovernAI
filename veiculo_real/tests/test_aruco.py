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

    def test_small_oblique_stop_sign_is_decoded_at_original_coordinates(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker = cv2.aruco.generateImageMarker(dictionary, 1, 70)
        patch = cv2.copyMakeBorder(marker, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        source = np.float32([[0, 0], [89, 0], [89, 89], [0, 89]])
        target = np.float32([[101, 159], [111, 155], [110, 173], [100, 174]])
        transform = cv2.getPerspectiveTransform(source, 4*target)
        image = cv2.warpPerspective(patch, transform, (1280, 960), borderValue=135)
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        marks = ArucoDetector().detect(image)
        self.assertEqual([m.marker_id for m in marks], [1])
        self.assertAlmostEqual(marks[0].cx, 105., delta=1.)
        self.assertAlmostEqual(marks[0].cy, 165., delta=2.)
        self.assertAlmostEqual(marks[0].side_px, 13., delta=2.)

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
    def test_isolated_read_survives_a_newer_frame_without_markers(self):
        from collections import deque
        from fva_car.cones import ConeObservation
        from fva_car.vision_process import publish_observation
        import main

        pending = deque(maxlen=64)
        vision = {'marker_observations': pending}
        pose = (0., 0., 0.)
        marker = Marker(1, 320, 350, 14, 1.7, .6).to_tuple()
        publish_observation(vision, ConeObservation(1., markers=(marker,), marker_timestamp=1., marker_scan=1))
        publish_observation(vision, ConeObservation(1.06, marker_timestamp=1.06, marker_scan=2))
        signs = main.sign_controller(main.parameters, 'cones')
        command = signs.update(vision['observation'], 1.1, pose, lambda _: pose, .7, pending)
        self.assertEqual(command.state, 'APROXIMANDO')
        self.assertEqual(command.marker_id, 1)
        self.assertGreater(command.remaining_m, 1.)

    def test_lateral_cone_ahead_of_a_curve_is_classified_against_the_path(self):
        from fva_car.corridor import CorridorEstimate, sample_centerline
        vector = (0., 0., .7, 0., 1.4)
        x, y = sample_centerline(vector, [1.35], offset=-.65)[0]
        self.assertLess(abs(y), .25)  # alinhada com a tangente atual, mas fora da curva
        obs = seen(1., Marker(1, 320, 350, 16, x, y))
        obs.corridor = CorridorEstimate(True, 'AMBAS', vector[:3], 1.4, state_vector=vector)
        signs = SignController(confirm_frames=1)
        pose = (0., 0., 0.)
        command = signs.update(obs, 1., pose, lambda _: pose, .3)
        self.assertFalse(command.in_path)
        self.assertAlmostEqual(command.remaining_m, 1.35 - .07, delta=.05)

    def test_distinct_cone_rearms_the_same_id_after_passing_the_first(self):
        signs = SignController(confirm_frames=1, hold_s=.5)
        pose = (0., 0., 0.)
        first = Marker(1, 320, 350, 16, .08, .7)
        self.assertTrue(signs.update(seen(1., first), 1., pose, lambda _: pose, .1).stop)
        self.assertEqual(signs.update(seen(1.6), 1.6, pose, speed=0.).state, 'PARADO')
        self.assertEqual(signs.update(seen(2.2), 2.2, pose, speed=0.).state, 'SAINDO')
        self.assertEqual(signs.update(seen(2.3, first), 2.3, pose, lambda _: pose, .2).state, 'SAINDO')
        second = Marker(1, 330, 360, 16, 2., -.7)
        advanced = (.65, 0., 0.)
        command = signs.update(seen(2.5, second), 2.5, advanced, lambda _: advanced, .4)
        self.assertEqual(command.state, 'APROXIMANDO')
        self.assertGreater(command.remaining_m, 1.5)

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

    def test_confirms_with_misses_in_between(self):
        # 26/11: com a placa a vista so 22-65% das buscas a achavam; exigir 2
        # SEGUIDAS quase nunca confirmava em velocidade. 2 nas ultimas 4 bastam.
        sign = Marker(1, 320, 200, 40)
        signs = SignController({1: 'pare'}, confirm_frames=2, confirm_window=4)
        states = [signs.update(seen(t, *markers), t).state
                  for t, markers in ((1., (sign,)), (1.05, ()), (1.1, ()), (1.15, (sign,)))]
        self.assertEqual(states[-1], 'PARADO')   # sem posicao: para na hora
        old = SignController({1: 'pare'}, confirm_frames=2)  # janela = 2: seguidas, como antes
        states = [old.update(seen(t, *markers), t).state
                  for t, markers in ((1., (sign,)), (1.05, ()), (1.1, (sign,)))]
        self.assertEqual(states[-1], 'LIVRE')
        far = SignController({1: 'pare'}, confirm_frames=2, confirm_window=4)
        states = [far.update(seen(t, *markers), t).state   # 2 acertos a 4 buscas de distancia: nao
                  for t, markers in ((1., (sign,)), (1.05, ()), (1.1, ()), (1.15, ()), (1.2, (sign,)))]
        self.assertEqual(states[-1], 'LIVRE')

    def test_approach_slows_down_smoothly_to_the_stop(self):
        # 26/11: a vref caia em degrau para 0,3 m/s (aproximacao aos trancos).
        signs = SignController({1: 'pare'}, approach_speed=.3, approach_deceleration=.4,
                               deceleration=.5, reaction_s=.3)
        sign = Marker(1, 320, 200, 40, 3., .8, .1)   # placa ao lado, 3 m a frente
        pose = (0., 0., 0.)
        signs.update(seen(1., sign), 1.)
        command = signs.update(seen(1.05, sign), 1.05, pose, lambda _: pose, 1.2)
        self.assertEqual(command.state, 'APROXIMANDO')
        self.assertGreater(command.speed_limit, 1.)   # nada de degrau para 0,3
        limits, x = [], 0.
        while command.state == 'APROXIMANDO' and x < 3.:
            limits.append(command.speed_limit)
            x += .02
            command = signs.update(seen(1.05), 1.05 + x, (x, 0., 0.), speed=min(limits[-1], 1.2))
        steps = -np.diff(limits)
        self.assertTrue(np.all(steps >= -1e-9))          # so desce
        self.assertLess(np.max(steps), .03)              # sem degrau: a/v por 2 cm (antes ~0,9 de uma vez)
        self.assertAlmostEqual(limits[-1], .3, places=6)  # termina em approach_speed
        self.assertEqual(command.state, 'PARADO')
        self.assertAlmostEqual(signs.approach_limit(math.nan), .3)

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
        self.assertEqual(main.aruco_every(main.parameters), 1)  # todo quadro (26/11)
        self.assertEqual(main.aruco_every({'aruco': False}), 0)
        signs = main.sign_controller(main.parameters, 'cones')
        self.assertEqual(signs.actions, {1: 'pare'})
        self.assertIsNone(main.sign_controller(main.parameters, 'line'))
        self.assertEqual(signs.deceleration, main.parameters['coast_deceleration'])
        self.assertEqual(main.aruco_marker_m({'aruco': True}), .1)

    def test_steering_delay_command_line(self):
        import main
        from unittest.mock import patch
        self.assertEqual(main.parameters['steer_delay_s'], .16)
        with patch.object(sys, 'argv', ['main.py', '--atraso-direcao', '0.2']), patch.object(main, 'run_mission') as run:
            main.main()
        self.assertEqual(run.call_args[0][0]['steer_delay_s'], .2)

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


class MarkerGeometryTests(unittest.TestCase):
    """Posicao pela altura do centro e a correcao "abaixo do chao" (ensaio de 24/11)."""
    def setUp(self):
        from fva_car.camera_pose import camera_pose
        f, self.size = 520., (640, 480)
        c, s = math.cos(math.radians(20)), math.sin(math.radians(20))
        R = np.array([[1, 0, 0], [0, c, s], [0, -s, c]]) @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.]])
        self.R, self.t = R, -R @ np.array([0., 0., .25])
        self.K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1.]])
        self.H = np.linalg.inv(self.K @ np.c_[R[:, 0], R[:, 1], self.t])
        self.pose = camera_pose(self.H, self.size)

    def project(self, point, side_m):
        camera = self.R @ np.asarray(point, float) + self.t
        pixel = (self.K @ camera)[:2]/camera[2]
        return tuple(pixel), self.K[0, 0]*side_m/camera[2]

    def test_height_gives_the_position_without_the_size(self):
        from fva_car.camera_pose import implied_side, marker_at_height
        pixel, side_px = self.project([1.8, -.6, .05], .075)
        point = marker_at_height(self.pose, self.H, pixel, .05)
        np.testing.assert_allclose(point, [1.8, -.6, .05], atol=1e-6)
        self.assertAlmostEqual(implied_side(self.pose, pixel, side_px, point), .075, places=4)
        self.assertIsNone(marker_at_height(self.pose, self.H, pixel, .3))   # acima da camera

    def test_size_too_big_puts_the_sign_below_ground_and_is_corrected(self):
        from fva_car.aruco import Marker
        from fva_car.cones import RealConeDetector
        pixel, side_px = self.project([1.8, -.6, 0.], .075)       # marcador de 7,5 cm no chao
        detector = RealConeDetector.__new__(RealConeDetector)
        detector.camera_pose, detector.full_homography = self.pose, self.H
        detector.aruco_marker_m, detector.aruco_marker_height_m = .10, None   # configurado 10 cm
        fields = detector._marker_position(Marker(1, pixel[0], pixel[1], side_px))
        self.assertEqual(fields['source'], 'chao')
        self.assertAlmostEqual(fields['x'], 1.8, delta=.01)       # sem a correcao: ~2,4 m
        self.assertAlmostEqual(fields['size_m'], .075, places=3)
        detector.aruco_marker_m = .075                             # tamanho certo: vale o tamanho
        self.assertEqual(detector._marker_position(Marker(1, pixel[0], pixel[1], side_px))['source'], 'tamanho')
        detector.aruco_marker_height_m = 0.                        # altura dada: vale a altura
        self.assertEqual(detector._marker_position(Marker(1, pixel[0], pixel[1], side_px))['source'], 'altura')


class EndToEndStopTests(unittest.TestCase):
    """run_mission de verdade (modo cones) ate parar na placa, a frente ou ao lado."""
    def run_to_stop(self, cone_x, side_y, ultrasound_range, settings=None, finish=None):
        """finish(car): fim da missao; padrao: parou depois de andar."""
        import json
        import tempfile
        from unittest.mock import Mock, patch
        import test_mission as tm
        from fva_car.cones import ConeObservation
        mission, L = tm.mission, .36
        car = tm.FakeCar([])
        car.p, car.th, car.v, car.steps_left, car.dt = np.array([-L, 0.]), 0., 0., 10**6, .02
        shared = {'scan': 0}

        def step():
            car.t += .02
            target = max(0., car.vref)
            car.v = target if target >= car.v else max(target, car.v - .5*.02)
            car.p = car.p + [car.v*.02, 0.]
            if 'data' in shared:
                x_rel = cone_x - (car.p[0] + L) + .05
                visible = x_rel > .9 and math.atan2(abs(side_y), x_rel) < math.radians(31)
                stamp = round(car.t, 1)
                if stamp != shared.get('stamp'):
                    shared['stamp'], shared['scan'] = stamp, shared['scan'] + 1
                markers = ((1, 320., 400., 40., x_rel, side_y, 0., .075, 'altura'),) if visible else ()
                shared['data']['observation'] = ConeObservation(
                    timestamp=stamp, errors=(0., 0., 0.), markers=markers, path_dkappa=0., corridor_width_m=1.3,
                    marker_timestamp=stamp, marker_scan=shared['scan'])
            self.history.append((car.t, car.v, car.p[0]))
            if finish is not None:
                return not (car.t > 40 or finish(car))
            return not (car.t > 40 or (car.v == 0 and car.t > 3 and car.p[0] > .5))
        self.history = []
        self.car = car
        car.step = step
        rel = lambda: cone_x - (car.p[0] + L)
        car.get_distance = lambda: ((rel() + .07, True) if side_y == 0 and rel() < ultrasound_range else (4., True))

        def fake_thread(**kwargs):
            args = kwargs['args']

            class Thread:
                def start(self):
                    if kwargs.get('name') != 'captura-camera':
                        shared['data'] = args[1]

                def is_alive(self):
                    return True

                def join(self, timeout=None):
                    pass
            return Thread()
        path = Path(tempfile.mkdtemp())/'camera.json'
        path.write_text(json.dumps({'image_size': [640, 480],
                                    'image_points_px': [[160, 400], [480, 400], [260, 200], [380, 200]],
                                    'ground_points_m': [[.4, .3], [.4, -.3], [1.5, .3], [1.5, -.3]]}))
        detector = Mock()
        detector.with_preview.side_effect = lambda observation: observation
        with patch.object(mission.threading, 'Thread', side_effect=fake_thread), \
                patch.object(mission, 'RealConeDetector', return_value=detector), \
                patch('builtins.print'):
            mission.run_mission({'camera_calibration_file': str(path), 'ts': 60., 'save': False,
                                 'navigation_mode': 'cones', **(settings or {})},
                                car_factory=lambda _: car, clock=lambda: float(car.t))
        return cone_x - (car.p[0] + L + .07), car.v   # folga da frente do pneu ate o cone/placa

    def test_cone_in_front_even_without_the_ultrasound(self):
        for ultrasound in (4., 0.):
            gap, speed = self.run_to_stop(3., 0., ultrasound)
            self.assertEqual(speed, 0.)
            self.assertTrue(0. < gap < .3, gap)

    def test_stops_five_seconds_with_its_own_beep_then_goes_on(self):
        # Pedido de 26/11: no ID 1 para 5 s e segue; bipe diferente do ultrassom.
        import main
        self.run_to_stop(2.5, .9, 0., finish=lambda car: car.p[0] > 3.5)
        stopped = [t for t, v, _ in self.history if v == 0. and t > 1.]
        self.assertTrue(stopped, 'nao parou')
        still = max(stopped) - min(stopped)
        self.assertAlmostEqual(still, 5., delta=.15)   # 5 s parado de fato
        self.assertGreater(self.car.p[0], 3.5)           # seguiu e passou da placa
        patterns = {tuple(call.args[0]) for call in self.car.bz.beep.call_args_list}
        self.assertEqual(patterns, {tuple(main.STOP_BEEPS['aruco'][0])})
        self.assertNotEqual(main.STOP_BEEPS['aruco'][0], main.STOP_BEEPS['ultrassom'][0])

    def test_cone_in_front_stays_stopped_by_the_ultrasound_after_the_hold(self):
        # Placa num cone A FRENTE: depois dos 5 s seguir bateria; o ultrassom
        # (sem carencia) trava a parada, e o bipe passa a ser o dele.
        import main
        self.run_to_stop(3., 0., 4., finish=lambda car: car.t > 20.)
        self.assertLess(self.car.p[0] + .36 + .07, 3.)  # nao encostou no cone
        patterns = [tuple(call.args[0]) for call in self.car.bz.beep.call_args_list]
        self.assertEqual(patterns[0], tuple(main.STOP_BEEPS['aruco'][0]))
        self.assertEqual(patterns[-1], tuple(main.STOP_BEEPS['ultrassom'][0]))

    def test_follower_compensates_latency_with_the_recorded_poses(self):
        # 26/11: o main nunca ligava o historico de poses ao seguidor; a foto era
        # levada ao instante do comando pela guinada de agora x idade (oscilava).
        from unittest.mock import patch
        import test_mission as tm
        created, real = [], tm.mission.RealStanleyFollower

        def follower(config):
            created.append(real(config))
            return created[-1]
        with patch.object(tm.mission, 'RealStanleyFollower', side_effect=follower):
            self.run_to_stop(2.5, .9, 0.)
        self.assertEqual(len(created), 1)
        self.assertIsNotNone(created[0].odometry)
        self.assertTrue(hasattr(created[0].odometry, 'delta'))
        self.assertGreater(created[0].steer_delay, 0.)

    def test_fast_car_with_unreliable_detection_still_stops_at_the_sign(self):
        # 26/11: a 1,5 m/s a placa nao parava o carro. Busca que acha a placa so
        # 40% das vezes (como nos ensaios), 1,5 m/s: confirma (2 nas ultimas 4) e
        # desce no perfil ate parar na altura da placa.
        import unittest.mock as um
        import main
        from fva_car.cones import ConeObservation
        for seed in range(3):
            rng = np.random.default_rng(seed)

            def flaky(*args, **kwargs):
                if kwargs.get('markers') and rng.random() > .4:
                    kwargs['markers'] = ()
                return ConeObservation(*args, **kwargs)
            with self.subTest(seed=seed), um.patch.object(sys.modules['fva_car.cones'], 'ConeObservation',
                                                          side_effect=flaky):
                self.run_to_stop(6., .9, 0., settings={'velocity_reference': 1.5},
                                 finish=lambda car: car.t > 14 or (car.v == 0 and car.t > 3))
                self.assertEqual(self.car.v, 0.)
                self.assertGreater(max(v for _, v, _ in self.history), 1.2)   # andou rapido antes
                gap = 6. - (self.car.p[0] + .36 + .07)
                self.assertTrue(-.05 < gap < .2, gap)

    def test_sign_beside_the_track_on_both_sides(self):
        for side in (.9, -.8):
            gap, speed = self.run_to_stop(2.5, side, 0.)
            self.assertEqual(speed, 0.)
            self.assertTrue(-.05 < gap < .2, gap)
