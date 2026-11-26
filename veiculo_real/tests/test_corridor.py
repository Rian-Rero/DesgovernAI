"""Estimador do corredor de cones (fva_car/corridor.py), sem camera nem hardware."""
import math
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.corridor import (CorridorEstimator, CorridorSettings, SPEED_FACTOR, offsets,
                              sample_centerline)
from fva_car.lateral import GroundCalibration, RealStanleyFollower, LineObservation
from test_lateral import fake_car

CALIBRATION = Path(__file__).resolve().parent/'dados'/'camera_pista_virtual.json'  # fixa dos testes
WIDTH = 1.8
SPACING = .5


def cones(state, start=.8, stop=3., spacing=SPACING):
    """Cones da borda esquerda e direita entre start e stop m de caminho."""
    arcs = np.arange(-2., 6., spacing)
    left = sample_centerline(state, arcs, state[4]/2, max_curvature=5.)
    right = sample_centerline(state, arcs, -state[4]/2, max_curvature=5.)
    keep = lambda p: p[(p[:, 0] >= start) & (p[:, 0] <= stop)]
    return keep(left), keep(right)


def estimator(**changes):
    return CorridorEstimator(CorridorSettings(track_width_m=WIDTH, **changes))


class GeometryTests(unittest.TestCase):
    def test_borders_are_at_half_width_on_straights_and_clothoids(self):
        for state in ([0., 0., 0., 0., WIDTH], [.2, -.3, .3, .1, WIDTH], [-.1, .5, -.5, -.05, WIDTH]):
            arcs = np.linspace(0., 3., 7)
            for side in (1, -1):
                points = sample_centerline(state, arcs, side*WIDTH/2)
                d, along, _, _ = offsets(points, state)
                np.testing.assert_allclose(d, side*WIDTH/2, atol=2e-3)
                np.testing.assert_allclose(along, arcs, atol=2e-2)

    def test_anchor_is_closest_point_to_front_axle(self):
        state = [.3, .4, -.5, 0., WIDTH]
        d, along, _, _ = offsets([[0., 0.]], state)
        self.assertAlmostEqual(-d[0], .3, places=4)  # e: centro a esquerda do eixo
        self.assertAlmostEqual(along[0], 0., places=3)

    def test_cone_uncertainty_is_large_in_range_small_in_bearing(self):
        calibration = GroundCalibration.from_file(CALIBRATION)
        est = CorridorEstimator(CorridorSettings(), calibration.homography)
        covariance = est._point_covariance(np.array([[1., 0.], [2.5, 0.]]))
        sigma_x, sigma_y = np.sqrt(covariance[:, 0, 0]), np.sqrt(covariance[:, 1, 1])
        self.assertTrue(np.all(sigma_x > 3*sigma_y))
        self.assertGreater(sigma_x[1], 4*sigma_x[0])  # cresce ~x^2 com a distancia

    def test_settings_validation(self):
        self.assertEqual(CorridorSettings.from_value(True, track_width_m=1.2).track_width_m, 1.2)
        self.assertEqual(CorridorSettings.from_value({'memory_m': .5}).memory_m, .5)
        with self.assertRaises(ValueError):
            CorridorSettings.from_value({'nao_existe': 1})
        with self.assertRaises(ValueError):
            CorridorSettings.from_value({'memory_m': -1})
        with self.assertRaises(ValueError):
            CorridorSettings.from_value('sim')


class EstimatorTests(unittest.TestCase):
    def drive(self, est, truth, frames=20, step=.05, drop=None, start=.8, stop=3.):
        """Carro andando reto `step` m por quadro; a pista e `truth` (referencial inicial)."""
        estimate = None
        state = np.array(truth, float)
        for k in range(frames):
            left, right = cones(state, start, stop)
            if drop == 'left':
                left = left[:0]
            elif drop == 'right':
                right = right[:0]
            estimate = est.step(.1*k, left, right, (step, 0., 0.) if k else None, pose_frame=False)
            moved = est._move(state, step, 0., 0.) if est.x is not None else None
            state = moved if moved is not None else CorridorEstimator(est.settings)._move(state, step, 0., 0.)
        return estimate, state

    def test_straight_corridor_offset_and_heading(self):
        estimate, truth = self.drive(estimator(), [.15, -.05, 0., 0., WIDTH])
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertEqual(estimate.state, 'AMBAS')
        np.testing.assert_allclose(estimate.errors, truth[:3], atol=.03)
        self.assertAlmostEqual(estimate.width_m, WIDTH, delta=.05)

    def test_left_curve_seen_only_by_outer_border_turns_more_than_border(self):
        # Curva a esquerda: os cones internos (esquerda) saem da imagem primeiro.
        estimate, truth = self.drive(estimator(), [0., 0., .5, 0., WIDTH], drop='left')
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertEqual(estimate.state, 'SO_EXTERNA')
        self.assertAlmostEqual(estimate.errors[2], truth[2], delta=.12)
        # O centro e mais fechado que a borda externa (raio 2 m contra 2,9 m).
        outer_curvature = 1/(1/truth[2] + WIDTH/2)
        self.assertGreater(estimate.errors[2], outer_curvature)
        self.assertEqual(estimate.speed_factor, SPEED_FACTOR['SO_EXTERNA'])

    def test_right_curve_inner_border_only(self):
        # Raio 3,3 m: a borda interna (2,4 m) ainda aparece ate ~2,4 m a frente.
        estimate, truth = self.drive(estimator(), [0., 0., -.3, 0., WIDTH], drop='left')
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertEqual(estimate.state, 'SO_INTERNA')
        self.assertAlmostEqual(estimate.errors[2], truth[2], delta=.12)

    def test_curve_entry_ahead_is_tracked_as_curvature_change(self):
        # Reta ate 1,5 m e depois curva a esquerda (clotoide): kappa cresce a frente.
        est = estimator()
        estimate, truth = self.drive(est, [0., 0., -.2, .3, WIDTH], frames=12, stop=2.6)
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertGreater(est.x[3], .1)
        self.assertAlmostEqual(estimate.errors[0], truth[0], delta=.06)
        self.assertAlmostEqual(estimate.errors[1], truth[1], delta=.08)

    def test_outlier_cone_is_rejected(self):
        est = estimator()
        self.drive(est, [0., 0., 0., 0., WIDTH], frames=10, step=0.)
        before = np.array(est.x)
        left, right = cones(est.x)
        estimate = est.step(5., np.vstack([left, [[1.5, 0.]]]), right, (0., 0., 0.))  # porta no meio
        self.assertEqual(estimate.rejected, 1)
        np.testing.assert_allclose(est.x[:3], before[:3], atol=.02)

    def test_object_beside_the_track_is_left_out_by_the_band(self):
        # Mochila 50 cm fora da borda esquerda, 2,5 m a frente: o portao estatistico
        # (largo longe) a aceita; a faixa em torno da borda ajustada a tira.
        results = {}
        for band in (0., .2):
            est = estimator(band_m=band, band_growth=.1)
            self.drive(est, [0., 0., 0., 0., WIDTH], frames=3, step=.05)
            left, right = cones(est.x)
            estimate = est.step(1., np.vstack([left, [[2.5, WIDTH/2 + .5]]]), right, (.05, 0., 0.),
                                pose_frame=False)
            results[band] = (estimate.rejected, est.band_rejected, estimate.width_m, estimate.errors)
        self.assertEqual(results[0.][0], 0)                    # sem faixa: entrou
        self.assertEqual(results[.2][:2], (1, 1))               # com faixa: saiu pela faixa
        self.assertAlmostEqual(results[.2][2], WIDTH, delta=.05)
        self.assertGreater(abs(results[0.][2] - WIDTH), abs(results[.2][2] - WIDTH))

    def test_band_keeps_the_cones_of_a_curve_starting_ahead(self):
        # Curva comecando a frente: todos os cones longe saem da previsao reta,
        # juntos; o ajuste os acompanha e a faixa nao tira nenhum.
        est = estimator()
        self.assertGreater(est.settings.band_m, 0.)  # ligada por padrao
        removed = []
        original = est.update

        def update(*args, **kwargs):
            result = original(*args, **kwargs)
            removed.append(est.band_rejected)
            return result
        est.update = update
        estimate, _ = self.drive(est, [0., 0., -.2, .3, WIDTH], frames=12, stop=2.6)
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertEqual(sum(removed), 0)

    def test_memory_then_lost(self):
        est = estimator(memory_m=.5)
        self.drive(est, [0., 0., 0., 0., WIDTH], frames=10)
        left, right = cones(est.x)
        observed = est.step(.95, left, right[:0], (0., 0., 0.), pose_frame=False)
        self.assertEqual(observed.state, 'UMA_BORDA_RETA')
        self.assertTrue(observed.valid)
        t = 1.
        travelled = 0.
        estimate = None
        while travelled <= .45:
            t += .05
            travelled += .05
            estimate = est.step(t, [], [], (.05, 0., 0.), pose_frame=False)
            self.assertEqual(estimate.state, 'MEMORIA')
            self.assertTrue(estimate.valid)
            self.assertEqual(estimate.reference_state, observed.state)
            self.assertEqual(estimate.speed_factor, observed.speed_factor)
            self.assertEqual(estimate.slow, observed.slow)
        estimate = est.step(t + .05, [], [], (.1, 0., 0.), pose_frame=False)
        self.assertEqual(estimate.state, 'PERDIDO')
        self.assertFalse(estimate.valid)
        self.assertIsNone(est.x)

    def test_memory_follows_odometry_turn(self):
        est = estimator()
        self.drive(est, [0., 0., 0., 0., WIDTH], frames=10, step=0.)
        estimate = est.step(1., [], [], (0., 0., .1), pose_frame=False)  # carro virou a esquerda
        self.assertAlmostEqual(estimate.errors[1], -.1, delta=.01)       # pista fica a direita

    def test_rear_axle_pose_is_moved_to_front_axle(self):
        est = estimator(pose_offset_m=.36)
        self.drive(est, [0., 0., 0., 0., WIDTH], frames=10, step=0.)
        estimate = est.step(1., [], [], (0., 0., .1))  # girando sobre o eixo traseiro
        # O eixo dianteiro foi 0,036 m para a esquerda: o centro fica a direita.
        self.assertAlmostEqual(estimate.errors[0], -.36*math.sin(.1), delta=.005)

    def test_stationary_repeated_frames_do_not_drift(self):
        est = estimator()
        self.drive(est, [.1, .05, .2, 0., WIDTH], frames=5, step=0.)
        first = np.array(est.x)
        left, right = cones(first)
        for k in range(300):
            est.step(1. + .1*k, left, right, (0., 0., 0.))
        np.testing.assert_allclose(est.x[:3], first[:3], atol=.03)

    def test_far_cones_only_still_allow_slow_start(self):
        # Partida do ensaio 1 (logs 20261023_080354), carro parado: cones so a
        # partir de 1,5 m a direita e 2,2 m a esquerda, na calibracao real.
        calibration = GroundCalibration.from_file(CALIBRATION)
        est = CorridorEstimator(CorridorSettings(track_width_m=WIDTH), calibration.homography)
        left = np.array([[2.23, .9], [2.84, .92]])
        right = np.array([[1.5, -.84], [2.03, -.84], [2.63, -.9]])
        for k in range(10):
            estimate = est.step(.15*k, left, right, (0., 0., 0.) if k else None)
        self.assertTrue(estimate.valid, estimate.reason)
        # So cones longe (1,5 m+): os perto pesam mais no ajuste (model_growth), entao
        # o caminho a 1 m ainda e incerto -> parte devagar (sem engasgo desde o
        # feedforward do acelerador), mirando longe. Com cones a partir de 0,8 m
        # a partida e normal (StartupTests em test_controle_cones.py).
        self.assertTrue(estimate.slow)
        self.assertTrue(estimate.near_uncertain)
        self.assertLess(abs(estimate.errors[0]), .2)
        self.assertLess(abs(estimate.errors[2]), .2)

    def test_width_is_learned_from_a_wrong_guess_and_survives_a_reset(self):
        # Palpite 0,9 m numa pista de 1,8 m (a largura muda a cada dia).
        est = CorridorEstimator(CorridorSettings(track_width_m=.9))
        estimate, truth = self.drive(est, [0., 0., 0., 0., WIDTH], frames=15)
        self.assertEqual(estimate.state, 'AMBAS')
        self.assertAlmostEqual(estimate.width_m, WIDTH, delta=.05)
        self.assertAlmostEqual(est.learned_width, WIDTH, delta=.05)
        est.reset()  # perdeu os cones
        self.assertAlmostEqual(est.learned_width, WIDTH, delta=.05)
        # Volta vendo so a borda direita: o centro fica a meia largura APRENDIDA.
        left, right = cones(np.array([.2, 0., 0., 0., WIDTH]))
        for k in range(6):
            estimate = est.step(10. + .1*k, [], right, (0., 0., 0.) if k else None)
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertAlmostEqual(estimate.errors[0], .2, delta=.05)
        self.assertAlmostEqual(estimate.width_m, WIDTH, delta=.08)

    def test_wide_and_narrow_tracks_from_the_same_guess(self):
        for width in (1.0, 2.4):
            with self.subTest(width=width):
                est = CorridorEstimator(CorridorSettings(track_width_m=1.8))
                state = np.array([0., 0., 0., 0., width])
                for k in range(10):
                    left, right = cones(state)
                    estimate = est.step(.1*k, left, right, (0., 0., 0.) if k else None)
                self.assertAlmostEqual(estimate.width_m, width, delta=.06)

    def test_single_border_start_with_unknown_width_is_valid_but_slow(self):
        est = estimator()
        state = np.array([0., 0., 0., 0., WIDTH])
        _, right = cones(state)
        for k in range(8):
            estimate = est.step(.1*k, [], right, (0., 0., 0.) if k else None)
        self.assertIsNone(est.learned_width)
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertTrue(estimate.slow)  # largura so pelo palpite: devagar

    def test_needs_three_cones_or_both_sides_to_start(self):
        est = estimator()
        estimate = est.step(0., [[1.5, .9]], [], None)
        self.assertFalse(estimate.valid)
        self.assertIsNone(est.x)
        est.step(.1, [[1.5, .9]], [[1.5, -.9], [2., -.9]], None)
        self.assertIsNotNone(est.x)


class FollowerSpeedFactorTests(unittest.TestCase):
    def test_reference_speed_factor_reduces_vref_but_not_below_minimum(self):
        settings = dict(velocity_reference=1., curve_min_speed=.3, speed_filter_s=0.,
                        vision_latency_compensation=False)
        speeds = {}
        for factor in (1., .6, .1):
            follower = RealStanleyFollower(settings)
            car = fake_car()
            observation = LineObservation(10., errors=(0., 0., 0.), speed_factor=factor)
            follower.apply(car, observation, 10.)
            speeds[factor] = car.vref
        self.assertAlmostEqual(speeds[.6], .6*speeds[1.], delta=.01)
        self.assertAlmostEqual(speeds[.1], .3, delta=.01)



class DynamicWidthTests(unittest.TestCase):
    """A largura muda ao longo da pista (24/11: 1,1 a 1,8 m, ate 30 cm na mesma volta)."""
    @staticmethod
    def right_border(x):
        return -.6 - .4*np.clip((x - 3.)/1., 0., 1.)  # 1,2 m ate x = 3, 1,6 m depois de x = 4

    def drive(self, drop_at, frames=200, **changes):
        est = CorridorEstimator(CorridorSettings(track_width_m=1.2, **changes))
        xs, errors = np.arange(0., 14., .5), []
        for k in range(frames):
            car = .05*k
            ahead = xs - car
            keep = (ahead >= .8) & (ahead <= 3.)
            left = np.c_[ahead[keep], np.full(keep.sum(), .6)]
            right = np.c_[ahead[keep], self.right_border(xs[keep])]
            if car > drop_at:
                right = right[:0]  # a borda direita some (curva, cone caido)
            estimate = est.step(.1*k, left, right, (.05, 0., 0.) if k else None, pose_frame=False)
            if estimate.valid and car > drop_at:
                errors.append(abs(estimate.errors[0] - (.6 + self.right_border(car))/2))
        return np.asarray(errors), est

    def test_center_follows_a_widening_track_seen_by_both_borders(self):
        errors, est = self.drive(drop_at=99., frames=140)
        self.assertLess(errors.size, 1)  # nunca caiu: so confere a largura
        self.assertAlmostEqual(est.learned_width, 1.6, delta=.03)

    def test_border_lost_right_after_the_track_widens_keeps_the_new_center(self):
        # O centro com uma borda so usa a largura aprendida: ela tem de ja ser a nova.
        errors, est = self.drive(drop_at=4.2)
        self.assertGreater(errors.size, 50)
        self.assertLess(np.median(errors), .015)
        self.assertAlmostEqual(est.learned_width, 1.6, delta=.03)
        # Com a largura quase fixa de antes (width_noise .05) o centro ficava a ~2,6 cm.
        old, _ = self.drive(drop_at=4.2, width_noise=.05)
        self.assertGreater(np.median(old), 2*np.median(errors))


class KnownWidthTests(unittest.TestCase):
    """known_width_m (so pela configuracao; --largura-pista hoje e so o palpite
    inicial): o estimador comeca sabendo W e nao foge dele."""
    def test_known_width_overrides_the_start_pair_and_bounds_learning(self):
        settings = CorridorSettings(track_width_m=1.3, known_width_m=1.3)
        estimator = CorridorEstimator(settings)
        self.assertEqual(estimator.learned_width, 1.3)
        # Par de partida mal medido (1,8 m): a largura nao vem dele.
        left, right = np.array([[1.5, .9], [2.5, .9]]), np.array([[1.5, -.9], [2.5, -.9]])
        for k in range(20):
            estimator.step(.1*k, left, right, (0., 0., 0.) if k else None)
        self.assertLessEqual(estimator.learned_width, 1.3*1.15 + 1e-9)
        estimator.learned_width = 9.
        estimator.forget_width()
        self.assertEqual(estimator.learned_width, 1.3)
        self.assertIsNone(CorridorEstimator(CorridorSettings()).learned_width)


class SteerDelayPreviewTests(unittest.TestCase):
    """RealStanleyFollower._preview_actuator: estado quando o esterco de agora chegar a roda."""
    def follower(self, delay):
        return RealStanleyFollower({'steer_delay_s': delay, 'steering_response': 1., 'wheelbase': .36})

    def test_commands_in_flight_turn_the_path_seen_by_the_controller(self):
        car = fake_car()
        car.v = 1.
        steer = math.radians(10.)
        for delay in (0., .2):
            with self.subTest(delay=delay):
                follower = self.follower(delay)
                for k in range(20):  # 0,4 s esterçando 10 graus a esquerda
                    follower._record_command(.02*k, steer)
                errors = follower._preview_actuator((0., 0., 0.), car, .4)
                if delay == 0.:
                    self.assertEqual(errors, (0., 0., 0.))
                    continue
                # O carro ainda vai girar v/L*tan(st)*atraso para a esquerda: o
                # caminho reto fica a direita e com rumo negativo.
                turn = 1./.36*math.tan(steer)*delay
                self.assertAlmostEqual(errors[1], -turn, delta=.01)
                self.assertLess(errors[0], 0.)
                self.assertAlmostEqual(follower.steer_preview[2], turn, places=6)

    def test_only_the_commands_still_in_flight_count(self):
        car = fake_car()
        car.v = 1.
        follower = self.follower(.2)
        for k in range(10):
            follower._record_command(.02*k, math.radians(15.))  # antigo: ja chegou a roda
        for k in range(10, 30):
            follower._record_command(.02*k, 0.)                  # ultimos 0,4 s: reto
        errors = follower._preview_actuator((.1, .05, 0.), car, .58)
        self.assertAlmostEqual(errors[1], .05, places=6)  # sem giro previsto
        self.assertAlmostEqual(follower.steer_preview[2], 0., places=9)


if __name__ == '__main__':
    unittest.main()
