# -*- coding: utf-8 -*-
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075): 
# Fundamentos de Veículos Autônomos - 2026/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
from fva_car import Car
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,8)

# Globais
parameters = {	
				'ts'		: 18.0, 			# termina antes do fim da pista
				'speed'		: 3.0, 			# velocidade constante [m/s]
				'initial_position': [-15.0, -8.0], # inicio da reta (x, y) [m]
				'save'		: True,
				'logfile'	: 'logs/',
				'beep'		: True,
			}
	
########################################
# thread de controle de velocidade
########################################
def control_func(car):
		
	# mantem as rodas alinhadas para seguir reto na linha da pista
	car.set_steer(0.0)

	# mantém a velocidade longitudinal constante por realimentação
	car.set_vel(parameters['speed'])
		
########################################
# thread de visão
########################################
def vision_func(car):
		
	# pega imagem
	image = car.get_image(gray=False)
	
	# ultrasom
	dist, _ = car.get_distance()
	#print(f'Ultrasonic distance: {dist:.1f}')
	
	return image
				
########################################
# main program
########################################
if __name__ == "__main__":
	
	plt.figure(1)
	plt.ion()
	
	# cria comunicação com o carrinho
	car = Car(parameters)
	
	try:
		# começa a simulação
		car.start_mission()

		# main loop
		while car.t <= parameters['ts']:
			
			# lê senores
			car.step()
			
			# mantem o controlador ativo durante toda a simulacao
			control_func(car)
			
			# funcao de visao
			image = vision_func(car)
			
			########################################
			# plota	
			plt.subplot(211)
			plt.cla()
			plt.gca().imshow(image, cmap='gray')
			plt.axis('off')
			plt.title(f'Telemetria em t={car.t:.1f}s')
			
			plt.subplot(212)
			plt.cla()
			t = [traj['t'] for traj in car.traj]
			v = [traj['v'] for traj in car.traj]
			vref = [traj['vref'] for traj in car.traj]
			plt.plot(t, v, label='velocidade')
			plt.plot(t, vref, '--', label='referencia')
			plt.ylabel('v[m/s]')
			plt.xlabel('t[s]')
			plt.legend()
			
			plt.show()
			plt.pause(0.01)

		# salva
		if parameters['save']:
			car.save()
	
	except KeyboardInterrupt:
		print("\nMissao interrompida pelo usuario.")
		
	finally:
		car.close()
