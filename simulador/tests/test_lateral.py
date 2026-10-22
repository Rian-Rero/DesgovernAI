import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import (BicycleModel, StanleyController, YellowLineFollower,
                             fit_line_errors, project_ground, yellow_centers, curve_speed, LocalLineMemory)


class LateralTests(unittest.TestCase):
    def test_memory_preserves_measured_near_path_then_expires(self):
        memory = LocalLineMemory()
        x = np.linspace(0, 1.5, 40)
        points = np.column_stack((x, .2*x*x, np.zeros(len(x))))
        memory.update(0., points, [0, 0, 0])
        # Imagem seguinte perdeu o trecho proximo, mas ele ja foi observado.
        cloud = memory.update(.1, points[x > .9], [.3, 0, 0])
        self.assertLess(np.min(cloud[:, 0]), .1)
        # Uma falha curta mantem os pontos; uma falha persistente nao.
        self.assertGreater(len(memory.update(.2, np.empty((0, 3)), [.3, 0, 0])), 0)
        self.assertEqual(len(memory.update(.5, np.empty((0, 3)), [.3, 0, 0])), 0)
        cloud = memory.update(4., points[x > .9], [.3, 0, 0])
        self.assertGreater(np.min(cloud[:, 0]), .9)

    def test_tight_arc_recovers_tangent_without_false_opposite_heading(self):
        for sign in [-1, 1]:
            radius = 1.1
            theta = np.linspace(.4, 1.1, 40)
            points = np.column_stack((radius*np.sin(theta),
                                      sign*radius*(1-np.cos(theta))))
            lateral, heading, curvature = fit_line_errors(
                points, [0, 0], [1, 0], with_curvature=True)
            self.assertAlmostEqual(lateral, 0., places=8)
            self.assertAlmostEqual(heading, 0., places=8)
            self.assertAlmostEqual(curvature, sign/radius, places=8)

    def test_offset_arc_is_selected_even_if_extrapolated_parabola_is_flat(self):
        # Um arco exato deve prevalecer sobre uma parabola cuja extrapolacao
        # muda o sinal da orientacao e da distancia no eixo dianteiro.
        radius, angle = 2., .75
        theta = np.linspace(.15, .8, 60)
        for sign in [-1, 1]:
            tangent = np.array([math.cos(angle), sign*math.sin(angle)])
            normal = np.array([-tangent[1], tangent[0]])
            lateral = -.7*sign
            points = (lateral*normal + radius*np.sin(theta)[:, None]*tangent
                      + sign*radius*(1-np.cos(theta))[:, None]*normal)
            cross, heading, curvature = fit_line_errors(
                points, [0, 0], [1, 0], with_curvature=True)
            self.assertAlmostEqual(cross, lateral, places=8)
            self.assertAlmostEqual(heading, sign*angle, places=8)
            self.assertAlmostEqual(curvature, sign/radius, places=8)

    def test_braking_holds_last_steer_only_briefly(self):
        car = Mock()
        car.t = 0.
        car.parameters = {}
        car.get_image.return_value = np.zeros((40, 60, 3), dtype=np.uint8)
        car.sim.getObjectMatrix.return_value = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
        car.sim.getObjectPosition.return_value = [0, 0, 0]
        car.sim.getObjectFloatParam.return_value = math.pi/3
        car.v, car.dt, car.t = .8, .05, 1.
        car.vref, car.u, car.st, car.motor_pwm_percent = 0., 0., 0., 0.
        follower = YellowLineFollower(car)
        follower.last_valid_time = .9
        follower.controller.steer = .2
        follower(car)
        car.set_vel.assert_called_with(0.)
        car.set_steer.assert_called_with(-.2)
        car.t = 1.5
        follower(car)
        car.set_steer.assert_called_with(0.)

    def test_curve_fit_does_not_invent_cross_track_or_heading(self):
        for bend in [-.4, .4]:
            x = np.linspace(.3, 1.2, 40)
            points = np.column_stack((x, bend*x*x, np.zeros(len(x))))
            lateral, heading, curvature = fit_line_errors(
                points, [0, 0], [1, 0], with_curvature=True)
            self.assertAlmostEqual(lateral, 0., places=10)
            self.assertAlmostEqual(heading, 0., places=10)
            self.assertAlmostEqual(curvature, 2*bend, places=10)

    def test_curve_speed_reduces_for_bends_and_tracking_error(self):
        self.assertEqual(curve_speed(1.5, 0, 0, 0), 1.5)
        self.assertLess(curve_speed(1.5, 1., 0, 0), 1.)
        self.assertLess(curve_speed(1.5, 0, .8, .5), .7)
        self.assertEqual(curve_speed(.3, 0, 1, 1), .3)

    def test_offset_curve_uses_normal_projection_and_tangent_at_same_point(self):
        # Ponto conhecido cuja normal passa pelo eixo dianteiro (origem local).
        # x + y*y' = 0 em x=-0.1; isso fixa c sem resolver raizes no teste.
        foot_x = -.1
        slope = .5 + 2*.4*foot_x
        foot_y = -foot_x/slope
        intercept = foot_y - .4*foot_x**2 - .5*foot_x
        x = np.linspace(-.3, 1.2, 60)
        for sign in [-1, 1]:
            for angle in [0., 1.1]:
                front = np.array([3., -2.])
                forward = np.array([math.cos(angle), math.sin(angle)])
                left = np.array([-forward[1], forward[0]])
                y = sign*(.4*x*x + .5*x + intercept)
                points = front + x[:, None]*forward + y[:, None]*left
                lateral, heading, curvature = fit_line_errors(
                    points, front, forward, with_curvature=True)
                self.assertAlmostEqual(lateral, sign*math.hypot(foot_x, foot_y), places=8)
                self.assertAlmostEqual(heading, math.atan(sign*slope), places=8)
                self.assertAlmostEqual(curvature, sign*.8/(1+slope*slope)**1.5, places=8)

    def test_follower_converges_with_scene_actuator_sign(self):
        # Regressao do ultimo log: juntas positivas giram a frente para direita.
        # Exercita o adaptador real do seguidor, nao somente Stanley isolado.
        for initial in [-.38, .38]:
            car = Mock()
            car.t = 0.
            car.parameters = {}
            car.get_image.return_value = np.zeros((40, 60, 3), dtype=np.uint8)
            car.sim.getObjectMatrix.return_value = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
            car.sim.getObjectPosition.return_value = [0, 0, 0]
            car.sim.getObjectFloatParam.return_value = math.pi/3
            car.v, car.dt = 1.5, .05
            car.vref, car.u, car.st, car.motor_pwm_percent = 1.5, 0., 0., 0.
            car.set_steer.side_effect = lambda command: setattr(car, 'st', command)
            follower = YellowLineFollower(car)
            model = BicycleModel()
            state = np.array([0., initial, 0.])
            with patch('fva_car.lateral.fit_line_errors') as perception:
                for step in range(200):
                    car.t = step*car.dt
                    perception.return_value = (
                        -(state[1] + model.wheelbase*math.sin(state[2])), -state[2], 0.)
                    follower(car)
                    if step == 0:
                        self.assertEqual(math.copysign(1., car.st), math.copysign(1., initial))
                    state += car.dt*model.derivative(state, car.v, -car.st)
            self.assertLess(abs(state[1]), .005)
            self.assertLess(abs(state[2]), .005)
            self.assertAlmostEqual(car.log.call_args.kwargs['steer_geometric_rad'], -car.st)

    def test_stanley_converges_from_both_sides_of_straight_line(self):
        model = BicycleModel()
        for initial in [-.6, .6]:
            state = np.array([0., initial, .15])
            controller = StanleyController()
            for _ in range(1000):
                # Eixo dianteiro projetado na reta y=0.
                front_y = state[1] + model.wheelbase * math.sin(state[2])
                steer = controller.update(-front_y, -state[2], 1.5, .01)
                state += .01 * model.derivative(state, 1.5, steer)
            self.assertLess(abs(state[1]), .005)
            self.assertLess(abs(state[2]), .005)

    def test_curve_tracking_with_fitted_errors_in_closed_loop(self):
        # Cadeia geometria -> erros -> Stanley -> bicicleta, sem mock dos erros.
        # Em curva o erro angular de equilibrio e diferente de zero: o carro
        # precisa estercar para que seu eixo dianteiro acompanhe o circulo.
        model = BicycleModel()
        radius, speed, dt = 2., .8, .02
        for sign in [-1, 1]:
            controller = StanleyController()
            state = np.array([0., -.3*sign, .1*sign])
            tail_errors = []
            for step in range(1000):
                forward = np.array([math.cos(state[2]), math.sin(state[2])])
                front = state[:2] + model.wheelbase*forward
                theta = math.atan2(front[0], radius-sign*front[1])
                angles = theta + np.linspace(-.3, 1.2, 60)/radius
                points = np.column_stack((radius*np.sin(angles),
                                          sign*radius*(1-np.cos(angles))))
                errors = fit_line_errors(points, front, forward, with_curvature=True)
                self.assertIsNotNone(errors)
                lateral, heading, curvature = errors
                steer = controller.update(lateral, heading, speed, dt)
                self.assertLessEqual(abs(steer), math.radians(20.))
                state += dt*model.derivative(state, speed, steer)
                if step >= 900:
                    tail_errors.append(lateral)
                    self.assertAlmostEqual(curvature, sign/radius, places=6)
            self.assertLess(max(abs(error) for error in tail_errors), .01)

    def test_saturation_rate_and_zero_speed(self):
        control = StanleyController()
        self.assertAlmostEqual(control.update(10, 1, 0, .05), math.radians(3))
        for _ in range(100):
            steer = control.update(10, 1, 0, .05)
        self.assertAlmostEqual(steer, math.radians(20))
        with self.assertRaises(ValueError):
            control.update(float('nan'), 0, 1, .05)

    def test_model_linearization(self):
        model = BicycleModel()
        a, b = model.linearize(1.5)
        self.assertAlmostEqual(a[0, 1], 1.5)
        self.assertAlmostEqual(b[1, 0], 1.5/.302)
        self.assertAlmostEqual(model.yaw_rate(1.5, 1e-6)/1e-6, b[1, 0], places=6)

    def test_yellow_detection_ignores_white_and_blue(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:, 60:65] = [255, 220, 0]
        image[:, 100:110] = [255, 255, 255]
        image[:, 20:30] = [0, 0, 255]
        centers = yellow_centers(image)
        self.assertGreater(len(centers), 10)
        np.testing.assert_allclose(centers[:, 0], 62)

    def test_projection_and_fit_recover_metric_line(self):
        # Camera a 1 m, olhando para frente e para baixo a 45 graus.
        s = math.sqrt(.5)
        rotation = np.array([[0, s, s], [1, 0, 0], [0, s, -s]])
        origin = np.array([0., 0., 1.])
        transform = np.column_stack((rotation, origin))
        x = np.linspace(.4, 1.5, 30)
        world = np.column_stack((x, .15 + .1*x, np.zeros(len(x))))
        camera = (world-origin) @ rotation
        focal = 320/(2*math.tan(math.pi/6))
        pixels = np.column_stack((160-.5-focal*camera[:, 0]/camera[:, 2],
                                  120-.5-focal*camera[:, 1]/camera[:, 2]))
        recovered = project_ground(pixels, (240, 320, 3), transform, math.pi/3, 0.)
        np.testing.assert_allclose(recovered, world, atol=1e-12)
        cross, heading = fit_line_errors(recovered, [0, 0], [1, 0])
        self.assertAlmostEqual(cross, .15/math.sqrt(1.01))
        self.assertAlmostEqual(heading, math.atan(.1))
        self.assertIsNone(fit_line_errors(np.empty((0, 3)), [0, 0], [1, 0]))

    def test_line_loss_commands_stop_and_logs_invalid(self):
        car = Mock()
        car.t = 0.
        car.parameters = {}
        car.get_image.return_value = np.zeros((120, 160, 3), dtype=np.uint8)
        matrix = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
        car.sim.getObjectMatrix.return_value = matrix
        car.sim.getObjectPosition.return_value = [0, 0, 0]
        car.sim.getObjectFloatParam.return_value = math.pi/3
        car.vref, car.u, car.st, car.motor_pwm_percent = 0., 0., 0., 0.
        follower = YellowLineFollower(car)
        follower(car)
        car.set_vel.assert_called_once_with(0.)
        car.set_steer.assert_called_once_with(0.)
        self.assertEqual(car.log.call_args.kwargs['line_valid'], 0)


if __name__ == '__main__':
    unittest.main()
