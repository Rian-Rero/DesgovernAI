"""Integracao da missao e captura com carro falso; nenhum GPIO ou camera real."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

REAL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REAL))
SPEC=importlib.util.spec_from_file_location('real_mission_under_test',REAL/'main.py')
mission=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=mission
SPEC.loader.exec_module(mission)


class FakeCar:
    def __init__(self,events):
        self.events=events
        self.t=0.;self.dt=.02;self.v=.4;self.vref=.5;self.u=.3;self.st=.1
        self.a=0.;self.w=0.;self.th=0.;self.p=np.zeros(2)
        self.gear='forward';self.emergencia=False;self.vel_error_integral=4.
        self.traj=[{'t':0.}]
        self.steps_left=2
        self.distances=iter([(10.,True)]*10)
        self.bz=Mock();self.bz.beep.return_value=True
        self.start_mission=Mock(side_effect=lambda:self.events.append('start'))
        self.save=Mock(side_effect=lambda:self.events.append('save'))
        self.close=Mock(side_effect=lambda:self.events.append('close'))
        self.log=Mock()

    def step(self):
        if self.steps_left==0:return False
        self.steps_left-=1
        self.t+=1.
        return True

    def get_distance(self):return next(self.distances)
    def _set_ref(self,value):
        self.vref=value
        self.events.append(('reference',value))
    def set_u(self,value):
        self.u=value
        self.events.append(('throttle',value))
    def set_steer(self,value):
        self.st=value
        self.events.append(('steer',value))
    def set_vel(self,value):
        self.vref=value
        self.events.append(('speed',value))


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        path=Path(self.folder.name)/'camera.json'
        path.write_text(json.dumps({'image_size':[640,480],
            'image_points_px':[[160,400],[480,400],[260,200],[380,200]],
            'ground_points_m':[[.4,.3],[.4,-.3],[1.5,.3],[1.5,-.3]]}))
        self.settings={'camera_calibration_file':str(path),'ts':3.,'save':True}
        self.events=[]
        self.car=FakeCar(self.events)

    def fake_thread(self,**kwargs):
        events=self.events
        args=kwargs['args']
        capture = kwargs.get('name') == 'captura-camera'
        class Thread:
            def start(self):
                if capture:
                    events.append('capture_start')
                    return
                events.append('vision_start')
                args[1]['observation']=mission.LineObservation(timestamp=10.,errors=(.1,0.,0.))
            def is_alive(self):return True
            def join(self,timeout=None):events.append('join')
        return Thread()

    def run_fake(self,settings=None):
        with patch.object(mission.threading,'Thread',side_effect=self.fake_thread):
            mission.run_mission(self.settings if settings is None else settings,
                                car_factory=lambda _:self.car,clock=lambda:10.)

    def test_missing_calibration_is_rejected_before_car_factory(self):
        factory=Mock()
        settings=dict(self.settings,camera_calibration_file=str(Path(self.folder.name)/'missing.json'))
        with self.assertRaises((FileNotFoundError,ValueError)):
            mission.run_mission(settings,car_factory=factory)
        factory.assert_not_called()

    def test_moving_or_disabled_camera_is_rejected_before_hardware(self):
        for changes in ({'camera':False},{'ultrasonic_steering':True}):
            factory=Mock()
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                mission.run_mission(dict(self.settings,**changes),car_factory=factory)
            factory.assert_not_called()

    def test_vehicle_is_neutralized_before_waiting_for_capture_thread(self):
        self.car.steps_left=0
        self.run_fake()
        join=self.events.index('join')
        self.assertLess(self.events.index(('throttle',0.)),join)
        self.assertEqual(self.events[join-1],('steer',0.))
        self.assertEqual(self.car.vel_error_integral,0.)
        self.assertLess(join,self.events.index('close'))
        self.car.save.assert_called_once()
        self.car.close.assert_called_once()

    def test_close_still_occurs_if_saving_fails(self):
        self.car.steps_left=0
        self.car.save.side_effect=OSError('disk unavailable')
        with self.assertRaises(OSError):self.run_fake()
        self.car.close.assert_called_once()
        self.assertIn('join',self.events)
        self.assertEqual(self.car.u,0.)

    def test_cleanup_neutralizes_if_starting_the_mission_fails(self):
        self.car.start_mission.side_effect=RuntimeError('startup failed')
        with self.assertRaises(RuntimeError):self.run_fake()
        self.car.close.assert_called_once()
        self.assertEqual(self.car.u,0.)
        self.assertEqual(self.car.st,0.)

    def test_obstacle_stop_remains_latched_after_distance_recovers(self):
        self.car.distances=iter([(.1,True),(10.,True)])
        follower=Mock()
        with patch.object(mission,'RealStanleyFollower',return_value=follower):self.run_fake()
        self.assertEqual(len(follower.apply.call_args_list),2)
        self.assertTrue(all(call.kwargs['obstacle_detected']
                            for call in follower.apply.call_args_list))
        logs=[call.kwargs for call in self.car.log.call_args_list if 'obstacle_stop' in call.kwargs]
        self.assertEqual([log['obstacle_stop'] for log in logs],[1,1])

    def test_full_rc_bypasses_stanley_while_obstacle_keeps_priority(self):
        self.car.sel_direcao = self.car.sel_tracao = 0
        self.car.distances = iter([(10., True), (.1, True)])
        follower = Mock()
        with patch.object(mission, 'RealStanleyFollower', return_value=follower):
            self.run_fake()
        self.assertIn(('speed', 0.), self.events)
        follower.controller.reset.assert_called_once()
        follower.apply.assert_called_once()
        self.assertTrue(follower.apply.call_args.kwargs['obstacle_detected'])

    def test_capture_timestamp_is_taken_before_acquisition_and_snapshot_is_atomic(self):
        stop=threading.Event();data={};events=[]
        def clock():events.append('clock');return 5.
        def capture(gray=False):
            self.assertFalse(gray)
            events.append('capture')
            return np.zeros((10,20,3),np.uint8)
        observation=mission.LineObservation(timestamp=5.,errors=(.1,.2,0.))
        def observe(frame,timestamp):
            events.append('observe')
            self.assertEqual(timestamp,5.)
            stop.set()
            return observation
        car=Mock();car.get_image.side_effect=capture
        detector=Mock();detector.observe_frame.side_effect=observe
        mission.vision_func(car,data,stop,detector,clock)
        self.assertEqual(events,['clock','capture','clock','clock','observe','clock'])
        self.assertEqual(list(data),['observation'])
        self.assertEqual(data['observation'].timestamp, observation.timestamp)
        self.assertEqual(data['observation'].errors, observation.errors)
        self.assertEqual(data['observation'].capture_seconds, 0.)
        self.assertEqual(data['observation'].processing_seconds, 0.)
        car.set_u.assert_not_called()
        car.set_steer.assert_not_called()

    def test_cli_sigterm_handler_uses_interrupt_cleanup_and_restores_previous_handler(self):
        previous=object()
        with patch.object(mission.sys if hasattr(mission,'sys') else sys,'argv',
                          ['main.py','--speed','.4']), \
                patch.object(mission.signal,'signal',return_value=previous) as install:
            def run(*args,**kwargs):
                handler=install.call_args_list[0].args[1]
                with self.assertRaises(KeyboardInterrupt):
                    handler(mission.signal.SIGTERM,None)
            with patch.object(mission,'run_mission',side_effect=run) as run:
                mission.main()
            run.assert_called_once_with({'velocity_reference':.4},display=False)
            self.assertEqual(install.call_args_list[-1].args,
                             (mission.signal.SIGTERM,previous))

    def test_capture_failure_publishes_invalid_observation_without_actuation(self):
        car=Mock();car.get_image.side_effect=RuntimeError('read failed')
        detector=Mock();data={}
        mission.vision_func(car,data,threading.Event(),detector,clock=lambda:5.)
        self.assertIsNone(data['observation'].errors)
        self.assertEqual(data['observation'].timestamp,5.)
        car.set_u.assert_not_called()
        car.set_steer.assert_not_called()

    def test_invalid_encoder_protocol_is_rejected_before_opening_drivers(self):
        factory = Mock()
        with self.assertRaisesRegex(ValueError, 'encoder_protocol'):
            mission.run_mission({'encoder_protocol': 'auto'}, car_factory=factory)
        factory.assert_not_called()

    def test_cli_allows_explicit_legacy_firmware_without_changing_default_rc(self):
        with patch.object(sys, 'argv', ['main.py', '--encoder-protocol', 'rpm', '--speed', '.3']), \
                patch.object(mission, 'run_mission') as run:
            mission.main()
        run.assert_called_once_with({'velocity_reference': .3, 'encoder_protocol': 'rpm'}, display=False)
        self.assertEqual(mission.parameters['encoder_protocol'], 'rc')

    def test_default_mission_passes_rc_protocol_to_car_factory(self):
        factory = Mock(return_value=self.car)
        with patch.object(mission.threading, 'Thread', side_effect=self.fake_thread):
            mission.run_mission(self.settings, car_factory=factory, clock=lambda: 10.)
        self.assertEqual(factory.call_args.args[0]['encoder_protocol'], 'rc')

    def test_invalid_encoder_does_not_enable_manual_bypass(self):
        self.car.sel_direcao = self.car.sel_tracao = 0
        self.car.encoder_valid = False
        follower = Mock()
        with patch.object(mission, 'RealStanleyFollower', return_value=follower):
            self.run_fake()
        follower.controller.reset.assert_not_called()
        self.assertEqual(follower.apply.call_count, 2)


class RealCarLoggingTests(unittest.TestCase):
    def test_late_lateral_columns_are_saved_with_missing_values_as_nan(self):
        # Drivers sao substituidos antes de executar o modulo; Car.__init__ nao roda.
        package=ModuleType('_real_car_test_package')
        package.__path__=[str(REAL/'fva_car')]
        replacements={package.__name__:package}
        for driver in ('encoder','servos','buzzer','ultrasonic','imu'):
            replacements[package.__name__+'.'+driver]=ModuleType(driver)
        name=package.__name__+'.car'
        spec=importlib.util.spec_from_file_location(name,REAL/'fva_car/car.py')
        module=importlib.util.module_from_spec(spec)
        replacements[name]=module
        with patch.dict(sys.modules,replacements):spec.loader.exec_module(module)
        car=module.Car.__new__(module.Car)
        car.traj=[{'t':0.,'v':0.},{'t':.02,'v':.1}]
        car.log(cross_track_m=.12,line_valid=1)
        with tempfile.TemporaryDirectory() as folder:
            car.logfile=folder
            car.save()
            saved=np.genfromtxt(Path(folder)/'car.csv',delimiter=',',names=True)
        self.assertEqual(saved.dtype.names,('t','v','cross_track_m','line_valid'))
        self.assertTrue(np.isnan(saved['cross_track_m'][0]))
        self.assertAlmostEqual(saved['cross_track_m'][1],.12)
        self.assertEqual(saved['line_valid'][1],1.)


if __name__=='__main__':unittest.main()
