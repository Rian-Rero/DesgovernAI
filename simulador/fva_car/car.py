# -*- coding: utf-8 -*-
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075): 
# Fundamentos de Veículos Autônomos - 2026/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
import sys, os
sys.path.append("coppeliasim_zmqremoteapi/")
from coppeliasim_zmqremoteapi_client import *
import numpy as np
import time
import threading
from datetime import datetime
try:
	from . import filter
	from .remote_api import CarRemoteAPI
	from .speed_controller import (
		COAST_DECELERATION, MOTOR_INPUT_GAIN, MOTOR_TORQUE_FACTOR,
		VELOCITY_FILTER_TIME, SpeedPI,
	)
except ImportError:
	import filter
	from remote_api import CarRemoteAPI
	from speed_controller import (
		COAST_DECELERATION, MOTOR_INPUT_GAIN, MOTOR_TORQUE_FACTOR,
		VELOCITY_FILTER_TIME, SpeedPI,
	)

########################################
# GLOBAIS
########################################
# parametros do carro
CAR = {
		'VELMAX'	: 2.5,				# m/s
		'ACCELMAX'	: 1.0, 				# m/s^2
		'STEERMAX'	: np.deg2rad(20.0),	# deg
		'MASS'		: 6.3,				# kg
		'L'			: 0.302,			# distancia entre os eixos das rodas
		'RW' 		: 0.08,				# raio da roda [m]
		'MI' 		: 0.05,				# constante de friccao
		'GRAV'   	: 9.81, 			# gravidade [m/s^2]
	}


########################################
# Carrinho
########################################
class Car:	
	########################################
	# construtor
	def __init__(self, parameters):
		
		self.parameters = parameters
		
		# inicia simulador
		self.init_coppelia_sim()
		
		# tempo
		self.t = 0.0
		# tempo de amostragem
		self.dt = 0.0
		
		# velocidade de comando
		self.vref = 0.0
		self.v = 0.0
		self.v_raw = 0.0
		
		# marcha
		self.gear = 1   # +1 forward, -1 reverse
		
		# comando de aceleracao
		self.u = 0.0
		self.motor_torque = 0.0
		self.motor_pwm_percent = 0.0
		self.coasting = False
		self.speed_controller = SpeedPI()
		self.velocity_filter_time = VELOCITY_FILTER_TIME
		self._steering_command = None
		
		# comando de esterçamento
		self.st = 0.0
		
		# trhead do beep
		self.thread = None

		# filtros dos sinais
		self.v_filt    = filter.AlphaFilter(alpha=0.6)
		self.a_filt    = filter.AlphaFilter(alpha=0.2)
		self.w_filt    = filter.AlphaFilter(alpha=0.5)
		
		# logs de salvamento
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		self.logfile = os.path.join(parameters['logfile'], timestamp)
		# cria a pasta do experimento
		os.makedirs(self.logfile, exist_ok=True)
		
		print("\033[33m##############################\033[0m", flush=True)
		print("\033[33mCarro pronto!\033[0m", flush=True)
		print("\033[33m##############################\033[0m", flush=True)
		
	########################################
	# inicializa interacao com o Coppelia -- Connect to CoppeliaSim
	def init_coppelia_sim(self):
			
		# Cria o cliente
		self.client = RemoteAPIClient(port=self.parameters.get('port', 23000))
		self.sim = self.client.getObject('sim')
		self.sim.stopSimulation()
		while self.sim.getSimulationState() != self.sim.simulation_stopped:
			time.sleep(0.01)
		if 'sample_time' in self.parameters:
			sample_time = self.parameters['sample_time']
			self.sim.setFloatParam(self.sim.floatparam_simulation_time_step, sample_time)
			if not np.isclose(self.sim.getSimulationTimeStep(), sample_time):
				raise RuntimeError('O simulador nao aceitou o intervalo de amostragem.')
		
		car_name = '/Car'
		
		# car
		self.robot = self.sim.getObject(car_name)
		if self.robot == -1:
			print ('Remote API function call returned with error code (robot): ', -1)
			
		# motors
		self.motorL = self.sim.getObject(car_name+'/joint_motor_L')
		if self.motorL == -1:
			print ('Remote API function call returned with error code (motorL): ', -1)
			
		self.motorR = self.sim.getObject(car_name+'/joint_motor_R')
		if self.motorR == -1:
			print ('Remote API function call returned with error code (motorR): ', -1)
			
		# steering
		self.steerL = self.sim.getObject(car_name+'/joint_steer_L')
		if self.steerL == -1:
			print ('Remote API function call returned with error code (steerL): ', -1)
		
		self.steerR = self.sim.getObject(car_name+'/joint_steer_R')
		if self.steerR == -1:
			print ('Remote API function call returned with error code (steerR): ', -1)
		
		# camera
		self.cam = self.sim.getObject(car_name+'/Vision_sensor')
		if self.cam == -1:
			print('Erro: câmera não encontrada')
			
		# depois de self.cam = ...
		self.ultra = self.sim.getObject('/Car/ultra_front')  # ajuste o nome igual ao da cena
		if self.ultra == -1:
			print('Erro: ultrassônico não encontrado')
		self.remote = CarRemoteAPI(self.sim, self.robot)
		self._vision_handling = None
		if self.parameters.get('explicit_vision', False):
			self._vision_handling = self.sim.getExplicitHandling(self.cam)
			self.sim.setExplicitHandling(self.cam, 1)
	
	########################################
	# get states
	def get_states(self):

		# Usa o intervalo atual tambem no calculo da aceleracao.
		state = self.remote.get_state(self.robot) if getattr(self, 'remote', None) else None
		t = (state['time'] if state is not None else self.get_time()) - self.tinit
		self.dt = t - self.t
		self.t = t

		# velocidade 
		self.v_ant = self.v
		self.v, self.w = self.get_vel(state) if state is not None else self.get_vel()

		# aceleracao
		self.a = self.get_accel()

		# orientacao
		self.th = self.get_yaw(state['quaternion']) if state is not None else self.get_yaw()
		
		# posicao
		self.p = np.asarray(state['position'][:2]) if state is not None else self.get_pos()
		
		return self.p, self.v, self.a, self.th, self.w, self.t
	
	########################################
	# comeca a missao
	def start_mission(self):
		self.speed_controller.reset()
		
		# sicronizado com o simulador
		self.client.setStepping(True)
		# Remove comandos dos motores que possam ter sido salvos na cena.
		for motor in [self.motorL, self.motorR]:
			self.sim.setJointTargetVelocity(motor, 0.0)
			self.sim.setJointForce(motor, 0.0)
		for steer in [self.steerL, self.steerR]:
			self.sim.setJointTargetPosition(steer, 0.0)
		
		# comeca a simulacao
		self.sim.startSimulation()
		# O primeiro passo inicializa os scripts antes das chamadas agrupadas.
		self.client.step()
		
		# tempo inicial
		self.tinit = self.get_time()
		self.wall_start = time.perf_counter()
		
		# estados iniciais
		self.get_states()
		
		# comeca na marcha para frente
		self.set_forward()

		# comeca parado
		self.set_u(0.0)
		self.set_steer(0.0)
		
		# aviso sonoro de inicio
		self.beep([0.2] * 3, silence=0.2)
		
		# salva trajetoria
		self.save_traj()
	
	########################################
	def step(self):
		try:
			# passo de simulacao
			self.client.step()

			# atualiza estados
			self.get_states()

			# se esta dando re, avise
			if self.gear == -1:
				self.beep(0.3, silence=1.0)

			# salva trajetoria
			self.save_traj()

			return True

		except KeyboardInterrupt:
			print("\nInterrupcao solicitada pelo usuario.")
			return False
			
	########################################
	# retorna tempo da simulacao no Coppelia
	def get_time(self):
		#while True:
		t = self.sim.getSimulationTime()
		if (t != -1.0): # Em caso de não retornar um erro
			return t
					
	########################################
	# retorna posicao do carro
	def get_pos(self):
		while True:
			pos = self.sim.getObjectPosition(self.robot, -1)
			if (pos != -1):
				return np.array((pos[0], pos[1]))			
				
	########################################
	# retorna yaw
	def get_yaw(self, q=None):
		if q is None:
			q = self.sim.getObjectQuaternion(self.robot, -1)
	
		# quaternion to roll-pitch-yaw
		yaw = self.quaternion_to_yaw(q)
		while yaw < 0.0:
			yaw += 2.0*np.pi
		while yaw > 2.0*np.pi:
			yaw -= 2.0*np.pi
		
		return yaw
		
	########################################
	def quaternion_to_yaw(self, q):
		
		qx, qy, qz, qw = q
		
		# Ensure the quaternion is normalized
		norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
		qx /= norm
		qy /= norm
		qz /= norm
		qw /= norm

		# Calculate the yaw angle (rotation about the Z-axis)
		yaw = np.arctan2(2 * (qx * qy + qw * qz), qw**2 + qx**2 - qy**2 - qz**2)

		return yaw
				
	########################################
	# retorna velocidades linear e angular
	def get_vel(self, state=None):
		if state is None:
			lin, ang = self.sim.getObjectVelocity(self.robot)
			matrix = self.sim.getObjectMatrix(self.robot, -1)
		else:
			lin, ang, matrix = state['linear'], state['angular'], state['matrix']

		# O eixo Z local do modelo Car aponta para a frente do veiculo.
		# A projecao exclui a queda vertical inicial e preserva o sinal real.
		forward = np.array([matrix[2], matrix[6], matrix[10]])
		self.v_raw = float(np.dot(lin, forward))
		v = self.v_raw

		# filtros
		if getattr(self, 'velocity_filter_time', 0) > 0 and self.dt > 0:
			self.v_filt.alpha = -np.expm1(-self.dt / self.velocity_filter_time)
		v = self.v_filt.filter(v)
		w = self.w_filt.filter(ang[2])

		return v, w
	
	########################################
	# retorna aceleracao
	def get_accel(self):
		
		if self.dt == 0.0:
			return 0.0
			
		a = (self.v - self.v_ant)/self.dt
		# filtro
		af = self.a_filt.filter(a)
		
		return af
		
	########################################
	# seta referencia de controle
	def _set_ref(self, vref):

		self.vref = float(np.clip(vref, -CAR['VELMAX'], CAR['VELMAX']))

		# troca para re
		if (self.vref < 0.0) and (self.gear == 1):
			self.set_reverse()

		# troca para frente
		elif (self.vref > 0.0) and (self.gear == -1):
			self.set_forward()

		return self.vref
					
	########################################
	# seta torque do veiculo
	def set_vel(self, vref):
		self._set_ref(vref)
		if self.vref == 0.0 and abs(self.v_raw) < 0.01:
			self.set_coast()
			return

		error = abs(self.vref) - self.gear * self.v
		motor_gain = 2 * MOTOR_TORQUE_FACTOR * MOTOR_INPUT_GAIN
		compensation = CAR['MI'] * CAR['GRAV']
		lower = motor_gain * (-CAR['ACCELMAX'] + compensation) - COAST_DECELERATION
		upper = motor_gain * (CAR['ACCELMAX'] + compensation) - COAST_DECELERATION
		normalized_input = self.speed_controller.update(error, self.dt, lower, upper)

		# Inverte u_normalizado = ganho_motor*(set_u + compensacao) - atrito.
		command = (normalized_input + COAST_DECELERATION) / motor_gain - compensation
		self.set_u(command)
	
	########################################
	# seta torque dos motores do veiculo
	def set_u(self, u, compensate_friction=True):

		# limita aceleracao
		self.u = np.clip(u, -CAR['ACCELMAX'], CAR['ACCELMAX'])
		
		# Compensa o atrito ja modelado pela cena, no sentido da marcha.
		# Na partida, compensa apenas se houver comando de aceleracao.
		F_compensation = 0.0
		if compensate_friction and (abs(self.v) > 0.01 or self.u > 0.0):
			F_compensation = CAR['MASS']*CAR['GRAV']*CAR['MI']

		# força de controle
		F_control = CAR['MASS']*self.u
		
		# forca longitudinal
		F = F_compensation + F_control
		
		# torque
		T = MOTOR_TORQUE_FACTOR*CAR['RW']*F

		# aplica o sentido da marcha
		T = self.gear*T
		self.motor_torque = float(T)
		# PWM equivalente: torque assinado como percentual do torque maximo.
		max_torque = MOTOR_TORQUE_FACTOR * CAR['RW'] * CAR['MASS'] * (
			CAR['ACCELMAX'] + CAR['MI'] * CAR['GRAV']
		)
		self.motor_pwm_percent = 100.0 * self.motor_torque / max_torque
		self.coasting = not compensate_friction and self.u == 0.0

		# atua
		velocity = float(np.sign(T)*CAR['VELMAX']/CAR['RW'])
		if getattr(self, 'remote', None):
			self.remote.set_motors(self.motorL, self.motorR, velocity, float(abs(T)))
			return
		for m in [self.motorL, self.motorR]:
			# A junta recebe rad/s: converte a velocidade linear pelo raio.
			self.sim.setJointTargetVelocity(m, velocity)
			# Apply the desired torques to the joints
			self.sim.setJointForce(m, abs(T))
			
	########################################
	# banguela: sem torque de tracao, frenagem ou compensacao
	def set_coast(self):
		self.set_u(0.0, compensate_friction=False)
		self.vref = 0.0
		self.speed_controller.reset()

	########################################
	# vai para frente
	def set_forward(self):

		# se ja esta para frente, nao faz nada
		if self.gear == 1:
			return

		# freia ate parar
		while abs(self.v) > 0.1:
			self.set_u(-CAR['ACCELMAX'])
			self.step()

		# troca a marcha
		self.gear = 1
		self.speed_controller.reset()
		
	########################################
	# coloca re
	def set_reverse(self):

		# se ja esta de re, nao faz nada
		if self.gear == -1:
			return

		# freia ate parar
		while abs(self.v) > 0.1:
			self.set_u(-CAR['ACCELMAX'])
			self.step()

		# troca a marcha
		self.gear = -1
		self.speed_controller.reset()
		
	########################################
	# seta steer do veiculo
	def set_steer(self, st):
		
		# distancia entre rodas
		width = 0.108
		
		self.st = np.clip(st, -CAR['STEERMAX'], CAR['STEERMAX'])
		st = self.st
		if getattr(self, '_steering_command', None) == st:
			return
		if np.tan(st) == 0:
			stL = stR = 0.0
		else:
			stL = np.arctan(CAR['L'] / ( width + CAR['L'] / np.tan(st)))
			stR = np.arctan(CAR['L'] / (-width + CAR['L'] / np.tan(st)))			
		
		if getattr(self, 'remote', None):
			self.remote.set_steering(self.steerL, self.steerR, float(stL), float(stR))
		else:
			self.sim.setJointTargetPosition(self.steerL, stL)
			self.sim.setJointTargetPosition(self.steerR, stR)
		self._steering_command = st
	
	########################################
	# get image data
	def get_image(self, gray=False):
		if self.parameters.get('explicit_vision', False):
			self.sim.handleVisionSensor(self.cam)
		
		while True:
			image, resolution = self.sim.getVisionSensorImg(self.cam)
			if image != -1:
				break
		
		# trata imagem
		img = np.frombuffer(image, dtype=np.uint8)
		img = img.reshape([resolution[1], resolution[0], 3])

		# corrige orientacao da imagem do CoppeliaSim
		img = np.flipud(img)

		# converte para tons de cinza, se solicitado
		if gray:
			img = np.mean(img, axis=2).astype(np.uint8)

		return img
		
	########################################
	# get ultrasonic distance
	def get_distance(self, max_dist=4.0):
		# CoppeliaSim retorna: hit, dist, point(list3), obj, normal(list3)
		hit, dist, p, obj, _n = self.sim.readProximitySensor(self.ultra)
		if hit:
			dist = min(float(dist), max_dist)
		else:
			dist = max_dist
		valid = True # sempre eh valido
		return dist, valid
	
	########################################
	# salva a trajetoria
	def save_traj(self):
		
		# dados (COLOCAR APENAS ESCALARES)
		data = {	't'     : self.t,
					'wall_time_s': time.perf_counter() - self.wall_start,
					'x'     : self.p[0], 
					'y'     : self.p[1],
					'v'     : self.v,
					'v_raw' : self.v_raw,
					'a'		: self.a,
					'vref'  : self.vref,
					'th'    : self.th,
					'w'     : self.w,
					'u'     : self.u,
					'torque_motor' : self.motor_torque,
					'motor_pwm_percent' : self.motor_pwm_percent,
					'coasting' : int(self.coasting),
				}
				
		# se ja iniciou as trajetorias
		try:
			self.traj.append(data)
		# se for a primeira vez
		except:
			self.traj = [data]
		
	########################################
	# salva trajetoria em csv
	def save(self):

		filename = os.path.join(self.logfile, 'car.csv')

		header = ','.join(self.traj[0].keys())

		data = np.array([list(traj.values()) for traj in self.traj])

		np.savetxt(filename, data, delimiter=',', header=header, comments='')
			
	########################################
	# dispara um ou mais beeps
	def beep(self, durations=0.1, silence=0.1):
		
		if not self.parameters['beep']:
			return
			
		# verifica duracoes
		if isinstance(durations, (int, float)):
			durations = [durations]
		durations = [max(0.0, float(d)) for d in durations]
		
		# dispara beeps
		if self.thread is not None and self.thread.is_alive():
			return
		self.thread = threading.Thread(target=self._beep_pattern, args=(durations, silence,), daemon=True)
		self.thread.start()
			
	########################################
	# executa os beeps
	def _beep_pattern(self, durations, silence=0.1):
		for duration in durations:
			print('\a', end='', flush=True)
			time.sleep(silence)
			
	########################################
	# termina a missao		
	def stop_mission(self):

		# termina parado
		self.set_u(-CAR['ACCELMAX'])
		self.set_steer(0.0)

		# tenta parar por no maximo alguns segundos
		t0 = self.get_time()
		while abs(self.v) > 0.1:
			self.step()
			# nao espera para sempre
			if self.get_time() - t0 > 2.0:
				break

		# aviso sonoro de fim
		self.beep([0.2] * 5, silence=0.2)
		time.sleep(1.0)

		# para simulador
		self.sim.stopSimulation()
		
	########################################
	# fecha tudo
	def close(self):
		try:
			self.stop_mission()

		except KeyboardInterrupt:
			pass

		finally:
			try:
				self.sim.stopSimulation()
				while self.sim.getSimulationState() != self.sim.simulation_stopped:
					time.sleep(0.01)
				if self._vision_handling is not None:
					self.sim.setExplicitHandling(self.cam, self._vision_handling)
				self.remote.close()
			except:
				pass

		print("\033[33m##############################\033[0m", flush=True)
		print("\033[33mMissao terminada!\033[0m", flush=True)
		print("\033[33m##############################\033[0m", flush=True)

########################################
# main teste
########################################
if __name__ == "__main__":
				
	# Globais
	parameters = {	
					'ts'		: 5.0, 			# tempo da simulacao
					'save'		: True,
					'logfile'	: 'logs/',
					'beep'		: True,
				}
	
	# cria comunicacao com o carrinho
	car = Car(parameters)
	
	try:
		car.start_mission()
		
		# testa leitura
		t0 = time.monotonic()
		while (time.monotonic() - t0) <= parameters['ts']:
			t = time.monotonic() - t0
			
			# le sensores
			car.step()
			
			# le ultrasom
			dist, valid = car.get_distance()
			# seta torque do motor
			if valid and dist > 0.10:
				if t < parameters['ts']/2:
					car.set_vel(0.7)
				else:
					car.set_vel(-0.7)
			else:
				car.set_vel(0.0)
			#
			print(
				f"Vel: {car.v:+.2f} m/s | "
				f"Ref: {car.vref:+.2f} m/s "
			)
				
			# seta estercamento junto com ultrasom
			car.set_steer(np.deg2rad(10.0)*np.sin(0.5*t))

		# salva os dados coletados
		if parameters['save']:
			car.save()
		
	finally:
		car.close()
