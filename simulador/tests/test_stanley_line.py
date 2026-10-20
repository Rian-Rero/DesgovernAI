"""Geometria, convencoes de sinal, convergencia e perda da linha."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy.stanley import StanleyConfig, StanleyController, curve_speed
from autonomy.yellow_line import GroundProjection, LineEstimate, YellowLineDetector
from seguir_linha import STANLEY_GAIN, YellowLineMission, camera_projection, place_car
from avaliar_stanley import reference_from_shape


def virtual_camera():
    """Pinhole independente: camera a 34 cm de altura, inclinacao de 10 graus."""
    focal, cx, cy = 520.0, 299.5, 199.5
    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
    offset, height = 0.05, 0.34
    z0 = c * offset + s * height
    h = np.array([[cx*c, -focal, cx*z0],
                  [cy*c - focal*s, 0, cy*z0 - focal*(s*offset-c*height)],
                  [c, 0, z0]])
    return GroundProjection(np.linalg.inv(h), (600, 400))


def line_image(projection, intercept=0.0, slope=0.0):
    image = np.full((400, 600, 3), 75, np.uint8)
    # Branco nao pode ser selecionado como amarelo.
    cv2.line(image, (500, 399), (320, 180), (255, 255, 255), 12)
    x = np.linspace(0.45, 5.0, 300)
    y = intercept + slope*x
    points = projection.image_points(np.column_stack((x, y)))
    cv2.polylines(image, [points.astype(np.int32)], False, (255, 230, 0), 5)
    return image


class StanleyTests(unittest.TestCase):
    def test_left_and_right_errors_give_opposite_steering(self):
        for sign in [-1, 1]:
            controller = StanleyController()
            self.assertGreater(sign * controller.update(sign*0.2, 0, 0.6, 0.1), 0)
            controller.reset()
            self.assertGreater(sign * controller.update(0, sign*0.2, 0.6, 0.1), 0)

    def test_zero_speed_saturation_and_rate_limit(self):
        c = StanleyController()
        command = c.update(10, 0, 0, 0.05)
        self.assertAlmostEqual(command, c.config.max_steer_rate * 0.05)
        for _ in range(100):
            command = c.update(10, 0, 0, 0.05)
        self.assertAlmostEqual(command, c.config.max_steer)

    def test_invalid_and_reverse_inputs_are_rejected(self):
        for values in [(0, 0, -1, .05), (0, 0, .5, 0), (math.nan, 0, .5, .05)]:
            with self.assertRaises(ValueError):
                StanleyController().update(*values)
        with self.assertRaises(ValueError):
            StanleyConfig(gain=-1)

    def test_bicycle_converges_to_straight_path_at_different_speeds(self):
        length, dt = 0.302, 0.02
        for velocity in [0.5, 1.0, 1.5]:
            for initial_y in [-0.4, 0.4]:
                y, yaw = initial_y, math.copysign(0.15, initial_y)
                control = StanleyController(StanleyConfig(gain=STANLEY_GAIN))
                for _ in range(1500):
                    front_y = y + length * math.sin(yaw)
                    delta = control.update(-front_y, -yaw, velocity, dt)
                    y += velocity * math.sin(yaw) * dt
                    yaw += velocity / length * math.tan(delta) * dt
                self.assertLess(abs(y), 0.01)
                self.assertLess(abs(yaw), 0.01)

    def test_bicycle_tracks_circle_using_front_axle(self):
        length, radius, velocity, dt = .302, 2.0, .6, .02
        x, y, yaw = 0.0, -0.2, 0.0
        control = StanleyController(StanleyConfig(gain=STANLEY_GAIN))
        for _ in range(2000):
            front = np.array([x + length*math.cos(yaw), y + length*math.sin(yaw)])
            radial = front - np.array([0, radius])
            error = np.linalg.norm(radial) - radius
            tangent = math.atan2(radial[1], radial[0]) + math.pi/2
            delta = control.update(error, tangent-yaw, velocity, dt)
            x += velocity*math.cos(yaw)*dt
            y += velocity*math.sin(yaw)*dt
            yaw += velocity/length*math.tan(delta)*dt
        self.assertLess(abs(error), 0.03)

    def test_curve_speed_respects_lateral_acceleration(self):
        self.assertEqual(curve_speed(1, 0), 1)
        speed = curve_speed(1.5, -2, 0.5)
        self.assertAlmostEqual(speed, 0.5)


class CameraLineTests(unittest.TestCase):
    def test_simulator_calibration_origin_is_front_axle_not_body_center(self):
        sim = Mock()
        angle = math.radians(30)
        camera = np.array([[0, math.sin(angle), math.cos(angle), .1],
                           [1, 0, 0, 0], [0, math.cos(angle), -math.sin(angle), .34]])
        robot = [0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, .14]
        sim.getVisionSensorRes.return_value = [600, 400]
        sim.getObjectInt32Param.return_value = 1
        sim.getObjectFloatParam.return_value = math.radians(60)
        sim.getObjectMatrix.side_effect = lambda h, _: camera.ravel().tolist() if h == 'camera' else robot
        sim.getObjectPosition.side_effect = lambda h, _: [.156, .108 if h == 'L' else -.108, .076]
        car = SimpleNamespace(sim=sim, cam='camera', robot='robot', steerL='L', steerR='R')
        projection = camera_projection(car, ground_z=.002)
        forward = .1 + (.34-.002)/math.tan(angle) - .156
        np.testing.assert_allclose(projection.ground_points([[299.5, 199.5]]), [[forward, 0]], atol=1e-10)

    def test_projection_round_trip_and_image_left_is_vehicle_left(self):
        projection = virtual_camera()
        ground = np.array([[1, 0], [1, 0.25], [2, -0.4]])
        pixels = projection.image_points(ground)
        self.assertLess(pixels[1, 0], pixels[0, 0])
        np.testing.assert_allclose(projection.ground_points(pixels), ground, atol=1e-10)

    def test_detects_left_right_offset_and_heading_in_metric_units(self):
        projection = virtual_camera()
        for offset, slope in [(.3, 0), (-.3, 0), (0, .12), (0, -.12)]:
            estimate = YellowLineDetector(projection).detect(line_image(projection, offset, slope))
            self.assertIsNotNone(estimate)
            self.assertAlmostEqual(estimate.cross_track, offset/math.sqrt(1+slope*slope), delta=.035)
            self.assertAlmostEqual(estimate.heading_error, math.atan(slope), delta=.035)
            self.assertGreater(estimate.confidence, .35)

    def test_bgr_real_camera_adapter_matches_rgb_simulation(self):
        projection = virtual_camera()
        image = line_image(projection, .25)
        rgb = YellowLineDetector(projection).detect(image)
        bgr = YellowLineDetector(projection, color_order='BGR').detect(image[:, :, ::-1])
        self.assertIsNotNone(bgr)
        self.assertAlmostEqual(rgb.cross_track, bgr.cross_track)

    def test_distant_s_bend_does_not_change_near_straight_tangent(self):
        projection = virtual_camera()
        image = np.full((400, 600, 3), 75, np.uint8)
        x = np.linspace(.45, 5, 400)
        y = np.where(x < 1.8, 0, .75*np.sin(3*(x-1.8)))
        pixels = projection.image_points(np.column_stack((x, y)))
        cv2.polylines(image, [pixels.astype(np.int32)], False, (255, 230, 0), 5)
        line = YellowLineDetector(projection).detect(image)
        self.assertIsNotNone(line)
        self.assertLess(abs(line.cross_track), .025)
        self.assertLess(abs(line.heading_error), .04)

    def test_no_yellow_or_uniform_camera_produces_no_path(self):
        for color in [(75, 75, 75), (0, 178, 255), (255, 255, 255), (255, 230, 0)]:
            estimate = YellowLineDetector(virtual_camera()).detect(
                np.full((400, 600, 3), color, np.uint8))
            self.assertIsNone(estimate)

    def test_tight_circular_arc_preserves_front_axle_tangent(self):
        projection = virtual_camera()
        image = np.full((400, 600, 3), 75, np.uint8)
        x = np.linspace(.45, 1.9, 300)
        y = .1 + 2 - np.sqrt(4-x*x)
        pixels = projection.image_points(np.column_stack((x, y)))
        cv2.polylines(image, [pixels.astype(np.int32)], False, (255, 230, 0), 5)
        line = YellowLineDetector(projection).detect(image)
        self.assertIsNotNone(line)
        self.assertTrue(line.circular)
        self.assertAlmostEqual(line.cross_track, .1, delta=.04)
        self.assertAlmostEqual(line.heading_error, 0, delta=.05)
        self.assertAlmostEqual(line.curvature, .5, delta=.12)

    def test_wrong_resolution_is_not_silently_recalibrated(self):
        with self.assertRaises(ValueError):
            YellowLineDetector(virtual_camera()).detect(np.zeros((480, 640, 3), np.uint8))

    def test_missing_frame_can_show_stop_overlay(self):
        detector = YellowLineDetector(virtual_camera())
        self.assertIsNone(detector.detect(None))
        self.assertEqual(detector.annotate(None, None).shape, (400, 600, 3))

    def test_short_visible_segment_at_a_corner_is_not_discarded(self):
        projection = virtual_camera()
        image = np.full((400, 600, 3), 75, np.uint8)
        x = np.linspace(.45, 1.01, 150)
        pixels = projection.image_points(np.column_stack((x, np.zeros_like(x))))
        cv2.polylines(image, [pixels.astype(np.int32)], False, (255, 230, 0), 5)
        line = YellowLineDetector(projection).detect(image)
        self.assertIsNotNone(line)
        self.assertGreater(line.confidence, .35)
        self.assertLess(abs(line.cross_track), .025)


class MissionTests(unittest.TestCase):
    def make_car(self):
        car = SimpleNamespace(t=1.0, dt=.05, v=0.0, st=0.0, vref=0.0,
                              speed_controller=Mock(), get_image=Mock(return_value=None),
                              set_coast=Mock(), set_u=Mock(), set_vel=Mock(), log=Mock())
        car.set_steer = Mock(side_effect=lambda st: setattr(car, 'st', st))
        return car

    def test_camera_pitch_is_temporary_and_can_be_restored(self):
        car = self.make_car()
        car.t = .1
        car.cam, car.robot, car.sim = 'camera', 'robot', Mock()
        angle = math.radians(10)
        world = np.array([[0, math.sin(angle), math.cos(angle), .1],
                          [1, 0, 0, 0], [0, math.cos(angle), -math.sin(angle), .34]])
        body_rotation = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        relative = world.copy()
        relative[:, :3] = body_rotation.T @ world[:, :3]
        original = relative.ravel().tolist()
        car.sim.getObjectMatrix.side_effect = lambda _, ref: original if ref == 'robot' else world.ravel().tolist()
        mission = YellowLineMission(camera_pitch=30)
        mission(car)
        applied = np.array(car.sim.setObjectMatrix.call_args.args[1]).reshape(3, 4)
        optical_axis = (body_rotation @ applied[:, :3])[:, 2]
        self.assertAlmostEqual(math.atan2(-optical_axis[2], np.linalg.norm(optical_axis[:2])), math.radians(30))
        mission.restore_camera()
        car.sim.setObjectMatrix.assert_called_with('camera', original, 'robot')

    def test_test_placement_preserves_height_and_sets_body_heading(self):
        sim = Mock()
        sim.getSimulationState.return_value = sim.simulation_stopped
        original = [0, 0, 1, -24, 0, -1, 0, -8.45, 1, 0, 0, .149]
        sim.getObjectMatrix.return_value = original
        _, saved = place_car(sim, [21.7, 9.66, 180])
        matrix = np.asarray(sim.setObjectMatrix.call_args.args[1]).reshape(3, 4)
        np.testing.assert_allclose(matrix[:, 3], [21.7, 9.66, .149])
        self.assertAlmostEqual(abs(math.atan2(matrix[1, 2], matrix[0, 2])), math.pi)
        self.assertEqual(saved, original)

    def test_acquisition_sign_and_stop_on_stale_or_low_confidence_vision(self):
        for failure in [None, LineEstimate(0.2, 0, 0, .1, (0, 0, .2), .5, 2.5, 0)]:
            mission = YellowLineMission()
            mission.detector = Mock()
            mission.detector.detect.return_value = LineEstimate(.2, 0, 0, 1, (0, 0, .2), .5, 2.5, 0)
            car = self.make_car()
            for _ in range(3):
                mission(car)
                car.t += .05
            self.assertFalse(mission.stopped)
            # Adaptacao medida: positivo a esquerda no nucleo, negativo no simulador.
            self.assertLess(car.st, 0)
            car.v = .6
            car.t += .2
            mission.detector.detect.return_value = failure
            mission(car)
            self.assertTrue(mission.stopped)
            car.set_u.assert_called_with(-1.0)
            self.assertEqual(car.vref, 0)
            self.assertEqual(mission.target_speed, 0)


class IndependentReferenceTests(unittest.TestCase):
    def make_reference(self, stretch_y=1.0):
        image = np.full((101, 101, 4), 75, np.uint8)
        image[:, 49:52, :] = (255, 230, 0, 255)
        return {
            'vertices': [-.505, -.505*stretch_y, 0, .505, -.505*stretch_y, 0,
                         .505, .505*stretch_y, 0, -.505, .505*stretch_y, 0],
            'indices': [0, 1, 2, 0, 2, 3],
            'texture': {'texture': image.tobytes(), 'resolution': [101, 101],
                        'coordinates': [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]},
        }

    def test_uv_mapping_and_independent_distances_are_in_meters(self):
        matrix = [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0]
        _, transform, distances, scale = reference_from_shape(self.make_reference(), matrix)
        self.assertAlmostEqual(scale, .01)
        pixel = transform @ np.array([2.1, 3, 1])
        self.assertAlmostEqual(pixel[0], 60)
        self.assertAlmostEqual(pixel[1], 50)
        self.assertAlmostEqual(distances[50, 60], .1, delta=.011)
        self.assertLess(distances[50, 50], .001)

    def test_nonisotropic_texture_cannot_silently_produce_metric_errors(self):
        matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        with self.assertRaises(ValueError):
            reference_from_shape(self.make_reference(stretch_y=1.5), matrix)


if __name__ == '__main__':
    unittest.main()
