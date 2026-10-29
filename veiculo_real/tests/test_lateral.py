"""Stanley e geometria metricos com imagens sinteticas, sem abrir hardware."""
import importlib.util
import json
import math
from enum import Enum
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

REAL = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REAL))
SPEC = importlib.util.spec_from_file_location('real_lateral_under_test', REAL/'fva_car/lateral.py')
lateral = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lateral
SPEC.loader.exec_module(lateral)


def calibration_data():
    # Coordenadas no chao relativas ao centro do eixo dianteiro: frente, esquerda.
    return {'image_size': [640,480],
            'image_points_px': [[160,400],[480,400],[260,200],[380,200]],
            'ground_points_m': [[.4,.3],[.4,-.3],[1.5,.3],[1.5,-.3]]}


def calibration_from(data):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'camera.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        return lateral.GroundCalibration.from_file(path)


def metric_line_frame(offset=.12, slope=0., color=(0,220,240)):
    """Desenha uma reta metrica no plano por projecao inversa da calibracao."""
    data = calibration_data()
    transform = cv2.getPerspectiveTransform(np.float32(data['image_points_px']),
                                             np.float32(data['ground_points_m']))
    x = np.linspace(.25,2.4,200)
    ground = np.c_[x, offset+slope*x].astype(np.float32)
    pixels = cv2.perspectiveTransform(ground[None],np.linalg.inv(transform))[0]
    frame = np.full((480,640,3),35,np.uint8)
    cv2.polylines(frame,[np.rint(pixels).astype(np.int32)],False,color,12)
    return frame


class Gear(Enum):
    FORWARD='forward'
    REVERSE='reverse'


def fake_car():
    car = SimpleNamespace(v=.4,dt=.02,t=1.,vref=.5,u=.7,st=0.,emergencia=False,
                          gear=Gear.FORWARD,vel_error_integral=3.)
    car._set_ref=Mock(side_effect=lambda v:setattr(car,'vref',v))
    car.set_u=Mock(side_effect=lambda u:setattr(car,'u',u))
    car.set_steer=Mock(side_effect=lambda st:setattr(car,'st',st))
    car.set_vel=Mock(side_effect=lambda v:setattr(car,'vref',v))
    car.log=Mock()
    return car


class CalibrationTests(unittest.TestCase):
    def test_metric_mapping_recovers_both_lateral_sides_and_heading(self):
        for offset,slope in ((.12,0.),(-.12,0.),(.08,.15)):
            with self.subTest(offset=offset,slope=slope):
                detector=lateral.RealLineDetector(calibration_from(calibration_data()),
                    path_settings={'clahe':False,'smoothing':0})
                observation=detector.observe_frame(metric_line_frame(offset,slope),10.)
                self.assertIsNotNone(observation.errors,observation.reason)
                error,heading,curvature=observation.errors
                self.assertAlmostEqual(error,offset/math.sqrt(1+slope*slope),delta=.025)
                self.assertAlmostEqual(heading,math.atan(slope),delta=.07)
                self.assertAlmostEqual(curvature,0.,delta=.15)
                self.assertEqual(observation.timestamp,10.)

    def test_invalid_calibration_does_not_produce_a_metric_controller(self):
        cases=[]
        data=calibration_data();data['image_size']=[0,480];cases.append(data)
        data=calibration_data();data['image_points_px']=[[0,0],[1,1],[2,2],[3,3]];cases.append(data)
        data=calibration_data();data['ground_points_m']=[[0,0],[1,0],[2,0],[3,0]];cases.append(data)
        data=calibration_data();data['image_points_px']=data['image_points_px'][:3];cases.append(data)
        data=calibration_data();data['ground_points_m'][0][0]=float('nan');cases.append(data)
        for data in cases:
            with self.subTest(data=data),self.assertRaises(ValueError):
                calibration_from(data)

    def test_missing_calibration_file_fails(self):
        with tempfile.TemporaryDirectory() as folder,self.assertRaises((FileNotFoundError,ValueError)):
            lateral.GroundCalibration.from_file(Path(folder)/'absent.json')

    def test_calibration_tool_validates_before_overwriting_output(self):
        spec=importlib.util.spec_from_file_location('real_calibration_tool_under_test',
                                                   REAL/'tools/calibrar_camera_lateral.py')
        tool=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'camera.json'
            tool.save_calibration(calibration_data(),output)
            original=output.read_text()
            invalid=calibration_data()
            invalid['ground_points_m']=[[0,0],[1,0],[2,0],[3,0]]
            with self.assertRaises(ValueError):tool.save_calibration(invalid,output)
            self.assertEqual(output.read_text(),original)

    def test_wrong_camera_resolution_is_invalid(self):
        detector=lateral.RealLineDetector(calibration_from(calibration_data()))
        observation=detector.observe_frame(np.zeros((240,320,3),np.uint8),10.)
        self.assertIsNone(observation.errors)


class SharedPerceptionTests(unittest.TestCase):
    def detector(self):
        return lateral.RealLineDetector(calibration_from(calibration_data()),
            path_settings={'clahe':False,'smoothing':0})

    def test_native_opencv_yellow_hue_and_bgr_boundary(self):
        hsv=np.uint8([[[30,255,230]]])
        yellow=tuple(int(v) for v in cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0,0])
        observation=self.detector().observe_frame(metric_line_frame(color=yellow),1.)
        self.assertIsNotNone(observation.errors,observation.reason)
        self.assertEqual(observation.frame.shape,(480,640,3))

    def test_metric_errors_follow_the_selected_path_result_center(self):
        detector=self.detector()
        frame=metric_line_frame(.12)
        result=detector.detector.process(frame,detector.settings,timestamp=1.)
        self.assertTrue(result.valid)
        # O contrato entre percepcao e Stanley e center; nao toda a mascara amarela.
        transform=detector.calibration.homography
        ground=np.c_[np.linspace(.35,1.8,100),np.full(100,-.12)].astype(np.float32)
        result.center=np.rint(cv2.perspectiveTransform(
            ground[None],np.linalg.inv(transform))[0]).astype(np.int32)
        with patch.object(detector.detector,'process',return_value=result) as process:
            observation=detector.observe_frame(frame,1.)
        process.assert_called_once()
        self.assertIsNotNone(observation.errors)
        self.assertAlmostEqual(observation.errors[0],-.12,delta=.01)

    def test_color_distractors_are_not_lateral_references(self):
        for color in ((255,0,0),(240,240,240),(0,220,0),(0,0,240)):
            with self.subTest(color=color):
                observation=self.detector().observe_frame(metric_line_frame(color=color),1.)
                self.assertIsNone(observation.errors)

    def test_loss_is_not_hidden_by_previous_path(self):
        detector=self.detector()
        self.assertIsNotNone(detector.observe_frame(metric_line_frame(),1.).errors)
        lost=detector.observe_frame(np.full((480,640,3),35,np.uint8),1.05)
        self.assertIsNone(lost.errors)

    def test_absent_frame_yields_invalid_observation(self):
        observation=self.detector().observe_frame(None,2.)
        self.assertIsNone(observation.errors)
        self.assertEqual(observation.timestamp,2.)


class RealFollowerTests(unittest.TestCase):
    def test_default_vref_uses_this_steps_reference_even_when_starting_from_zero(self):
        car = fake_car()
        car.v = car.vref = 0.
        car.dt = .2
        follower = lateral.RealStanleyFollower({'velocity_reference': .8})
        self.assertTrue(follower.apply(car, self.valid_observation(.04, 0.), 10.))
        self.assertGreater(car.vref, 0.)
        self.assertAlmostEqual(follower.controller.cross_track_term,
                               math.atan2(1.5*.04, abs(car.vref)+.3))
        self.assertAlmostEqual(car.log.call_args.kwargs['stanley_speed_m_s'], car.vref)
        self.assertEqual(car.log.call_args.kwargs['stanley_use_vref'], 1)

    def test_toggle_restores_measured_velocity_in_denominator(self):
        car = fake_car()
        car.v = .2
        car.dt = .2
        follower = lateral.RealStanleyFollower({'stanley_use_vref': False, 'velocity_reference': .8})
        self.assertTrue(follower.apply(car, self.valid_observation(.04, 0.), 10.))
        self.assertNotAlmostEqual(car.vref, car.v)
        self.assertAlmostEqual(follower.controller.cross_track_term, math.atan2(1.5*.04, .2+.3))
        self.assertAlmostEqual(car.log.call_args.kwargs['stanley_speed_m_s'], .2)
        self.assertEqual(car.log.call_args.kwargs['stanley_use_vref'], 0)

    def test_invalid_toggle_is_rejected(self):
        for value in (0, 1, 'True', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lateral.RealStanleyFollower({'stanley_use_vref': value})

    def test_nonfinite_vref_cuts_commands_instead_of_entering_equation(self):
        car = fake_car()
        car.set_vel.side_effect = lambda _: setattr(car, 'vref', float('nan'))
        follower = lateral.RealStanleyFollower({})
        self.assertFalse(follower.apply(car, self.valid_observation(), 10.))
        car.set_u.assert_called_with(0.)
        car.set_steer.assert_called_with(0.)
        self.assertEqual(car.vref, 0.)
        self.assertIn('Stanley invalida', follower.status)

    def test_invalid_encoder_overrides_a_fresh_visible_line(self):
        car = fake_car()
        car.encoder_valid = False
        follower = lateral.RealStanleyFollower({})
        self.assertFalse(follower.apply(car, self.valid_observation(), 10.))
        self.assert_neutral(car)
        self.assertEqual(car.log.call_args.kwargs['lateral_active'], 0)

    def test_encoder_expiring_during_actuation_is_logged_as_inactive(self):
        car = fake_car()
        car.encoder_valid = True
        car.set_vel.side_effect = lambda _: setattr(car, 'encoder_valid', False)
        follower = lateral.RealStanleyFollower({})
        self.assertFalse(follower.apply(car, self.valid_observation(), 10.))
        car._set_ref.assert_called_with(0.)
        car.set_u.assert_called_with(0.)
        car.set_steer.assert_called_with(0.)
        self.assertEqual(car.vel_error_integral, 0.)
        self.assertEqual(car.log.call_args.kwargs['lateral_active'], 0)

    def test_rc_reverse_request_does_not_run_forward_stanley(self):
        car = fake_car()
        car.sel_tracao, car.rc_acelerador = 0, -.4
        follower = lateral.RealStanleyFollower({})
        self.assertFalse(follower.apply(car, self.valid_observation(), 10.))
        self.assert_neutral(car)

    def valid_observation(self,lateral_error=.15,heading=0.,timestamp=10.):
        return lateral.LineObservation(timestamp=timestamp,errors=(lateral_error,heading,0.))

    def assert_neutral(self,car):
        car._set_ref.assert_called_with(0.)
        car.set_u.assert_called_with(0.)
        car.set_steer.assert_called_with(0.)
        self.assertEqual(car.vel_error_integral,0.)
        car.set_vel.assert_not_called()

    def test_orientation_and_distance_each_act_with_real_positive_left_sign(self):
        for cross,heading in ((.15,0.),(0.,.2),(-.15,0.),(0.,-.2)):
            with self.subTest(cross=cross,heading=heading):
                car=fake_car();follower=lateral.RealStanleyFollower({})
                self.assertTrue(follower.apply(car,self.valid_observation(cross,heading),10.))
                self.assertEqual(math.copysign(1,car.st),math.copysign(1,cross+heading))
                self.assertLessEqual(abs(car.st),math.radians(60)*car.dt+1e-12)
                car.set_vel.assert_called_once()
                self.assertGreater(car.vref,0.)

    def test_configured_actuator_sign_changes_only_steering_command(self):
        outputs=[]
        for sign in (-1.,1.):
            car=fake_car()
            lateral.RealStanleyFollower({'steering_sign':sign}).apply(car,self.valid_observation(),10.)
            outputs.append((car.st,car.vref))
        self.assertAlmostEqual(outputs[0][0],-outputs[1][0])
        self.assertAlmostEqual(outputs[0][1],outputs[1][1])

    def test_missing_lost_stale_future_and_nonfinite_observations_cut_throttle(self):
        observations=[None,lateral.LineObservation(timestamp=10.,reason='Linha perdida'),
                      self.valid_observation(timestamp=9.5),self.valid_observation(timestamp=11.),
                      self.valid_observation(timestamp=float('nan')),
                      self.valid_observation(lateral_error=float('nan'))]
        for observation in observations:
            with self.subTest(observation=observation):
                car=fake_car();follower=lateral.RealStanleyFollower({})
                self.assertFalse(follower.apply(car,observation,10.))
                self.assert_neutral(car)

    def test_emergency_obstacle_and_reverse_override_a_fresh_valid_path(self):
        for condition in ('emergency','obstacle','gear_reverse','negative_speed'):
            car=fake_car();obstacle=False
            if condition=='emergency':car.emergencia=True
            if condition=='obstacle':obstacle=True
            if condition=='gear_reverse':car.gear=Gear.REVERSE
            if condition=='negative_speed':car.v=-.1
            with self.subTest(condition=condition):
                self.assertFalse(lateral.RealStanleyFollower({}).apply(
                    car,self.valid_observation(),10.,obstacle_detected=obstacle))
                self.assert_neutral(car)

    def test_resume_after_loss_restarts_the_rate_limiter_from_neutral(self):
        car=fake_car();follower=lateral.RealStanleyFollower({})
        for _ in range(20):follower.apply(car,self.valid_observation(),10.)
        follower.apply(car,None,10.6)
        self.assertEqual(car.st,0.)
        follower.apply(car,self.valid_observation(-.15,timestamp=10.6),10.6)
        self.assertLess(car.st,0.)
        self.assertLessEqual(abs(car.st),math.radians(60)*car.dt+1e-12)

    def test_brief_line_loss_keeps_following_last_trajectory(self):
        car=fake_car();follower=lateral.RealStanleyFollower({})
        for _ in range(20):follower.apply(car,self.valid_observation(),10.)
        steer,vref=car.st,car.vref
        lost=lateral.LineObservation(timestamp=10.3,reason='Buscando faixa')
        self.assertTrue(follower.apply(car,lost,10.3))
        self.assertIn('Faixa perdida ha 0.30 s',follower.status)
        self.assertGreater(car.vref,0.)
        self.assertAlmostEqual(car.vref,vref,delta=.05)
        self.assertGreater(car.st,0.)
        self.assertAlmostEqual(car.st,steer,delta=math.radians(60)*car.dt+1e-9)
        log=car.log.call_args.kwargs
        self.assertEqual((log['line_valid'],log['line_hold'],log['lateral_active']),(0,1,1))
        self.assertAlmostEqual(log['line_lost_s'],.3)
        # Passada a tolerancia, a perda e tratada como real e o carro para.
        car.set_vel.reset_mock()
        self.assertFalse(follower.apply(car,lost,10.51))
        self.assert_neutral(car)
        self.assertIn('Faixa perdida por mais de 0.50 s: Buscando faixa',follower.status)
        # A memoria e esquecida: uma nova perda curta nao reutiliza a faixa antiga.
        self.assertFalse(follower.apply(car,lost,10.52))

    def test_prediction_moves_last_line_with_vehicle_motion(self):
        follower=lateral.RealStanleyFollower({})
        car=fake_car();car.v,car.dt,car.st,car.w=1.,.1,0.,0.
        # Linha a frente apontando para a esquerda: andar reto a afasta para a esquerda.
        lateral_error,heading,_=follower._predict_line((0.,.2,0.),car)
        self.assertAlmostEqual(lateral_error,.1*math.sin(.2))
        self.assertAlmostEqual(heading,.2)
        # Girar para a esquerda reduz o erro de orientacao; curvatura o aumenta.
        car.w=.5
        self.assertAlmostEqual(follower._predict_line((0.,.2,0.),car)[1],.15)
        car.w=0.
        self.assertAlmostEqual(follower._predict_line((0.,0.,.5),car)[1],.05)
        # Atuador invertido: car.st e car.w voltam para a convencao geometrica.
        inverted=lateral.RealStanleyFollower({'steering_sign':-1.})
        car.st,car.w=-.1,-.5
        self.assertEqual(inverted._predict_line((.05,.1,.2),car),
                         follower._predict_line((.05,.1,.2),SimpleNamespace(v=1.,dt=.1,st=.1,w=.5)))

    def test_curvature_feedforward_steers_into_the_curve_without_error(self):
        controller=lateral.StanleyController(max_rate=10.,curvature_gain=1.)
        controller.update(0.,0.,1.,.02,curvature=.5)
        self.assertAlmostEqual(controller.feedforward_term,math.atan(.36*.5))
        self.assertAlmostEqual(controller.raw_steer,math.atan(.36*.5))
        for gain,expected in ((1.,1),(0.,0)):
            with self.subTest(gain=gain):
                car=fake_car();follower=lateral.RealStanleyFollower({'stanley_curvature_ff':gain})
                observation=lateral.LineObservation(timestamp=10.,errors=(0.,0.,1.))
                self.assertTrue(follower.apply(car,observation,10.))
                self.assertEqual(int(car.st>0),expected)
                self.assertEqual(car.st<0,False)
        with self.assertRaises(ValueError):
            lateral.RealStanleyFollower({'stanley_curvature_ff':-1.})

    def test_latency_compensation_moves_errors_to_command_time_with_cap(self):
        observation=lateral.LineObservation(timestamp=9.9,errors=(0.,.2,0.))
        logged={}
        for compensate in (True,False):
            car=fake_car();car.v=1.
            follower=lateral.RealStanleyFollower({'vision_latency_compensation':compensate})
            self.assertTrue(follower.apply(car,observation,10.))
            logged[compensate]=car.log.call_args.kwargs
        # Linha apontando para a esquerda: andando 0,1 s ela se afasta para a esquerda.
        self.assertAlmostEqual(logged[True]['cross_track_m'],.1*math.sin(.2))
        self.assertAlmostEqual(logged[True]['vision_compensated_s'],.1)
        self.assertEqual(logged[False]['cross_track_m'],0.)
        self.assertEqual(logged[False]['vision_compensated_s'],0.)
        car=fake_car();car.v=1.
        follower=lateral.RealStanleyFollower({'vision_timeout':1.})
        follower.apply(car,lateral.LineObservation(timestamp=9.2,errors=(0.,.2,0.)),10.)
        self.assertAlmostEqual(car.log.call_args.kwargs['vision_compensated_s'],
                               lateral.MAX_LATENCY_COMPENSATION)

    def test_curvature_is_clamped_for_steering_and_filtered_for_speed(self):
        follower=lateral.RealStanleyFollower({'velocity_reference':1.5,'speed_filter_s':0})
        car=fake_car()
        # Sinal trocando a cada quadro (como no log real) quase nao muda o filtro.
        for index in range(20):
            now=10.+.05*index
            follower.apply(car,lateral.LineObservation(timestamp=now,errors=(0.,0.,.9*(-1)**index)),now)
        self.assertLess(abs(follower.curvature),.25)
        log=car.log.call_args.kwargs
        # O esterco usa a curvatura do proprio quadro, que se cancela com a orientacao.
        self.assertAlmostEqual(follower.controller.feedforward_term,math.atan(.36*.9*(-1)**19))
        self.assertAlmostEqual(log['curvature_filtered_m_inv'],follower.curvature)
        # A velocidade usa a filtrada: nao cai para sqrt(0,4/0,9) a cada quadro.
        self.assertGreater(car.vref,lateral.curve_speed(1.5,.9,0.,0.,lateral_accel=.4)+.2)
        # O mesmo quadro reaproveitado em varios passos nao reaplica o filtro.
        repeated=follower.curvature
        follower.apply(car,lateral.LineObservation(timestamp=now,errors=(0.,0.,5.)),now+.02)
        self.assertEqual(follower.curvature,repeated)
        # Acima do batente da direcao a curvatura e limitada nos dois usos.
        for index in range(40):
            now=12.+.05*index
            follower.apply(car,lateral.LineObservation(timestamp=now,errors=(0.,0.,5.)),now)
        limit=math.tan(math.radians(20))/.36
        self.assertAlmostEqual(follower.max_curvature,limit)
        self.assertAlmostEqual(follower.curvature,limit,places=3)
        self.assertAlmostEqual(follower.controller.feedforward_term,math.atan(.36*limit))
        self.assertEqual(car.log.call_args.kwargs['curvature_raw_m_inv'],5.)

    def test_speed_target_is_smoothed_against_one_step_spikes(self):
        car=fake_car();follower=lateral.RealStanleyFollower({'velocity_reference':.7})
        for index in range(30):
            now=10.+.02*index
            follower.apply(car,lateral.LineObservation(timestamp=now,errors=(0.,0.,0.)),now)
        self.assertAlmostEqual(car.vref,.7)
        # Um passo isolado com erro angular alto pedia ~0,53 m/s de uma vez.
        follower.apply(car,lateral.LineObservation(timestamp=10.6,errors=(0.,.25,0.)),10.6)
        self.assertAlmostEqual(car.log.call_args.kwargs['curve_speed_raw_m_s'],.7/1.25)
        self.assertGreater(car.vref,.68)
        unfiltered=lateral.RealStanleyFollower({'velocity_reference':.7,'speed_filter_s':0})
        unfiltered.apply(car,lateral.LineObservation(timestamp=10.6,errors=(0.,.25,0.)),10.6)
        self.assertAlmostEqual(car.vref,.7/1.25)

    def test_impossible_jump_is_discarded_but_a_persistent_change_is_accepted(self):
        car=fake_car();follower=lateral.RealStanleyFollower({})
        for index in range(5):
            now=10.+.05*index
            follower.apply(car,lateral.LineObservation(timestamp=now,errors=(0.,0.,0.)),now)
        # Um quadro com a orientacao saltando 0,8 rad em 50 ms (log real: 0,29 -> 0,88).
        glitch=lateral.LineObservation(timestamp=10.25,errors=(-.15,.8,0.))
        self.assertTrue(follower.apply(car,glitch,10.25))
        log=car.log.call_args.kwargs
        self.assertEqual((log['line_outlier'],log['line_hold'],log['line_valid']),(1,1,0))
        self.assertAlmostEqual(log['heading_error_rad'],0.,places=6)
        self.assertIn('salto impossivel',follower.status)
        # Reaproveitado no passo seguinte, o mesmo quadro continua descartado.
        follower.apply(car,glitch,10.27)
        self.assertEqual(car.log.call_args.kwargs['line_outlier'],1)
        # Medida coerente volta a ser usada normalmente.
        follower.apply(car,lateral.LineObservation(timestamp=10.3,errors=(.01,.02,0.)),10.3)
        self.assertEqual(car.log.call_args.kwargs['line_outlier'],0)
        self.assertAlmostEqual(car.log.call_args.kwargs['heading_error_rad'],.02,places=2)
        # Mudanca real persistente: descarta MAX_REJECTED_FRAMES quadros e aceita o seguinte.
        outliers=[]
        for index in range(1,lateral.MAX_REJECTED_FRAMES+2):
            now=10.3+.05*index
            follower.apply(car,lateral.LineObservation(timestamp=now,errors=(.2,.5,0.)),now)
            outliers.append(car.log.call_args.kwargs['line_outlier'])
        self.assertEqual(outliers,[1]*lateral.MAX_REJECTED_FRAMES+[0])
        self.assertAlmostEqual(car.log.call_args.kwargs['heading_error_rad'],.5,places=2)

    def test_zero_tolerance_stops_on_first_lost_frame_and_rejects_invalid_values(self):
        car=fake_car();follower=lateral.RealStanleyFollower({'line_loss_tolerance':0})
        follower.apply(car,self.valid_observation(),10.)
        car.set_vel.reset_mock()
        self.assertFalse(follower.apply(car,lateral.LineObservation(timestamp=10.02),10.02))
        self.assert_neutral(car)
        for value in (-.1,float('nan'),True,'0.5'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                lateral.RealStanleyFollower({'line_loss_tolerance':value})

    def test_safety_stops_are_not_bridged_by_line_memory(self):
        car=fake_car();follower=lateral.RealStanleyFollower({})
        follower.apply(car,self.valid_observation(),10.)
        car.set_vel.reset_mock()
        self.assertFalse(follower.apply(car,self.valid_observation(),10.05,obstacle_detected=True))
        self.assertFalse(follower.apply(car,None,10.1))
        self.assert_neutral(car)


if __name__=='__main__':
    unittest.main()
