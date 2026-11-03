"""Cones esparsos -> percepcao -> Stanley -> bicicleta em malha fechada.

Os erros nao sao simulados: sao reconstruidos a cada passo a partir das
bordas fixas. A centralizacao e medida separadamente na geometria da pista.
"""
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cones import ConePathDetector, reconstruct_cone_path
from fva_car.lateral import BicycleModel, StanleyController


def front_axle(state, model):
    forward = np.array([math.cos(state[2]), math.sin(state[2])])
    return state[:2] + model.wheelbase*forward, forward


def border_points(x, y):
    x, y = np.broadcast_arrays(x, y)
    return np.column_stack((x, y, np.zeros(len(x))))


def render_cones(state, front, red, green):
    """Camera pinhole a 35cm do chao e cones triangulares de 28cm.

    A funcao so desenha pixels. O detector recebe a imagem e a calibracao,
    sem receber a lista dos cones nem o deslocamento lateral verdadeiro.
    """
    forward = np.array([math.cos(state[2]), math.sin(state[2]), 0.])
    left = np.array([-forward[1], forward[0], 0.])
    up = np.array([0., 0., 1.])
    pitch, angle = math.radians(10), math.pi/3
    transform = np.column_stack((
        left, math.sin(pitch)*forward+math.cos(pitch)*up,
        math.cos(pitch)*forward-math.sin(pitch)*up, [*front, .35]))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    focal = 640/(2*math.tan(angle/2))
    objects = [(point, color)
               for points, color in ((red, (255, 0, 0)), (green, (0, 255, 0)))
               for point in points]
    objects.sort(key=lambda item: np.linalg.norm(item[0][:2]-front), reverse=True)
    for point, color in objects:
        triangle = np.array([point-left*.06, point+left*.06, point+[0., 0., .28]])
        camera = (triangle-transform[:, 3]) @ transform[:, :3]
        if np.min(camera[:, 2]) <= .1:
            continue
        pixels = np.column_stack((319.5-focal*camera[:, 0]/camera[:, 2],
                                  239.5-focal*camera[:, 1]/camera[:, 2]))
        # A base recortada nao tem contato com o chao observavel.
        if (np.min(pixels[:, 0]) < 1 or np.max(pixels[:, 0]) >= 639
                or np.min(pixels[:, 1]) < 1 or np.max(pixels[:, 1]) >= 479):
            continue
        cv2.fillConvexPoly(image, np.rint(pixels).astype(np.int32), color)
    return image, transform, angle


class ConeClosedLoopTests(unittest.TestCase):
    def test_sparse_straight_corridor_centers_from_either_side(self):
        model = BicycleModel()
        x = np.arange(-3., 30., 1.5)
        red = border_points(x, .9)
        green = border_points(x+.35, -.9)
        for offset in (-.4, .4):
            with self.subTest(initial_offset=offset):
                state = np.array([0., offset, .1*math.copysign(1., offset)])
                controller = StanleyController()
                cross_track = []
                for _ in range(350):
                    front, forward = front_axle(state, model)
                    observation = reconstruct_cone_path(red, green, [*front, 0.], forward)
                    self.assertTrue(observation.valid, observation.reason)
                    steer = controller.update(*observation.errors[:2], 1.2, .03)
                    # Medida independente: distancia do eixo dianteiro a y=0.
                    cross_track.append(float(front[1]))
                    state += .03*model.derivative(state, 1.2, steer)
                self.assertGreater(abs(cross_track[0]), .35)
                self.assertLess(max(abs(value) for value in cross_track[-50:]), .01)
                self.assertLess(abs(state[2]), .01)
                self.assertGreater(state[0], 10.)

    def test_complete_left_and_right_curves_converge_with_sparse_cones(self):
        model = BicycleModel()
        radius, half_width, speed, dt = 4.5, .9, 1., .03
        for turn in (-1, 1):
            center = np.array([0., turn*radius])
            # As bordas nao possuem cones emparelhados na mesma secao.
            angles = {'red': np.linspace(0., 2*math.pi, 64, endpoint=False),
                      'green': np.linspace(0., 2*math.pi, 64, endpoint=False)+.035}
            borders = {}
            for color, border_radius in (('red', radius-turn*half_width),
                                          ('green', radius+turn*half_width)):
                theta = angles[color]
                borders[color] = border_points(
                    border_radius*np.sin(theta), turn*(radius-border_radius*np.cos(theta)))
            for offset in (-.4, .4):
                with self.subTest(turn=turn, initial_offset=offset):
                    state = np.array([0., offset, turn*.08])
                    initial_heading = state[2]
                    controller = StanleyController()
                    radial_errors = []
                    for step in range(1100):
                        front, forward = front_axle(state, model)
                        progress = math.atan2(front[0], radius-turn*front[1])
                        visible = {}
                        # Janela local de cones observados; nao entrega o mapa
                        # inteiro, que conteria o outro lado da volta no campo.
                        for color in ('red', 'green'):
                            delta = (angles[color]-progress+math.pi) % (2*math.pi)-math.pi
                            keep = (delta >= -.3/radius) & (delta <= 4.5/radius)
                            visible[color] = borders[color][keep]
                        observation = reconstruct_cone_path(
                            visible['red'], visible['green'], [*front, 0.], forward)
                        self.assertTrue(observation.valid, f'passo {step}: {observation.reason}')
                        steer = controller.update(*observation.errors[:2], speed, dt)
                        self.assertLessEqual(abs(steer), math.radians(20.)+1e-12)
                        # Distancia geometrica ao circulo central, independente
                        # da estimativa retornada pela percepcao.
                        radial_errors.append(abs(float(np.linalg.norm(front-center)-radius)))
                        state += dt*model.derivative(state, speed, steer)
                    self.assertGreater(turn*(state[2]-initial_heading), 2*math.pi)
                    self.assertGreater(radial_errors[0], .35)
                    self.assertLess(max(radial_errors[-100:]), .02)
                    self.assertLess(max(radial_errors), .75*half_width)

    def test_camera_pixels_to_stanley_center_the_vehicle_without_geometry_shortcut(self):
        model = BicycleModel()
        x = np.arange(-2., 35., 1.5)
        red = border_points(x, .9)
        green = border_points(x+.3, -.9)
        for offset in (-.4, .4):
            with self.subTest(initial_offset=offset):
                state = np.array([0., offset, 0.])
                controller, detector = StanleyController(), ConePathDetector()
                errors = []
                for step in range(160):
                    front, forward = front_axle(state, model)
                    image, camera, angle = render_cones(state, front, red, green)
                    observation = detector.observe(image, camera, angle, 0.,
                                                   [*front, 0.], forward, step*.05)
                    self.assertTrue(observation.valid, f'passo {step}: {observation.reason}')
                    self.assertGreater(len(observation.red_pixels), 0)
                    self.assertGreater(len(observation.green_pixels), 0)
                    steer = controller.update(*observation.errors[:2], 1., .05)
                    errors.append(abs(float(front[1])))
                    state += .05*model.derivative(state, 1., steer)
                self.assertLess(max(errors[-20:]), .03)
                self.assertLess(abs(state[2]), .02)
                self.assertGreater(state[0], 7.)


if __name__ == '__main__':
    unittest.main()
