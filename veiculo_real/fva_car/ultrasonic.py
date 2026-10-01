# -*- coding: utf-8 -*-
########################################
# Disciplina: Topicos em Engenharia de Controle e Automacao IV (ENG075): 
# Fundamentos de Veiculos Autonomos - 2026/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automacao
# DELT - Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
import time
import numpy as np
import threading
import re
import subprocess

########################################
# Globais
########################################
"""
GPIO mode options: BOARD/BCM
# Raspberry Pi 3/4: BOARD pin numbers
# Raspberry Pi 5:   BCM (Broadcom System-On-Chip (SOC)) pin numbers
# trigger_pin = 18 (BOARD) and 24 (BCM)
#    echo_pin = 24 (BOARD) and  8 (BCM)
"""
GAIN = 343.0/2.0
TRIGGER_PIN = 24
ECHO_PIN = 8
SAMPLE_TIME = 0.1  # 100 ms -> 10 Hz
SENSOR_TIMEOUT  = 0.30   	# tempo maximo sem medida [s]

############################################
# Ultrasonic Class for Raspberry Pi 4 or 5
############################################
class Ultrasonic:
	########################################
	# Constructor
	########################################
	def __init__(self, trigger_pin=None, echo_pin=None, min_range=0.0, max_range=4.0):
		
		"""
		Initializes the ultrasonic sensor with given trigger and echo pins.
		Configures GPIO settings and set sensor limits

		Args:
			trigger_pin (int): GPIO pin number for the trigger signal.
			echo_pin (int): GPIO pin number for the echo signal.
			min_range (float): Minimum measurable distance in meters.
			max_range (float): Maximum measurable distance in meters.
		"""
		
		# limites do sensor
		self.min_range = min_range
		self.max_range = max_range
		
		# ultima leitura valida
		self.dist = 0.0
		self.valid = False
		self.last_measurement = None
		self.last_error = "aguardando_primeiro_echo"
		self.last_pulse_s = None
		self.consecutive_failures = 0
		self.total_failures = 0
		self.lock = threading.Lock()
		self.measure_lock = threading.Lock()
		self.echo_condition = threading.Condition()
		self._accepting_echo = False
		self._echo_start_ns = None
		self._echo_duration_ns = None
		self._echo_armed_at_ns = 0
		self._attempt_error = None
		self.echo_callback = None

		# Detectar a versao da Raspberry
		self.rpi_version = self.detect_rpi_version()
		#print(f"[INFO] Detected Raspberry Pi {self.rpi_version}")

		# Raspberry Pi 5
		if self.rpi_version == 5:
			import lgpio as GPIO
			self.GPIO = GPIO
			# detecta o chip
			chip_id = self._find_gpiochip()
			self.handle_chip = self.GPIO.gpiochip_open(chip_id)
			self.trigger_pin = trigger_pin if trigger_pin is not None else TRIGGER_PIN
			self.echo_pin = echo_pin if echo_pin is not None else ECHO_PIN
			self.GPIO.gpio_claim_output(self.handle_chip, self.trigger_pin)
			self.GPIO.gpio_claim_alert(self.handle_chip, self.echo_pin, self.GPIO.BOTH_EDGES)
			self.echo_callback = self.GPIO.callback(
				self.handle_chip, self.echo_pin, self.GPIO.BOTH_EDGES, self._on_lgpio_echo
			)
			
			# funcao de leitura pra raspberry pi 5
			self.read_func = lambda: self.GPIO.gpio_read(self.handle_chip, self.echo_pin)

		# Raspberry Pi 3/4
		else:
			import RPi.GPIO as GPIO
			self.GPIO = GPIO
			self.trigger_pin = trigger_pin if trigger_pin is not None else TRIGGER_PIN
			self.echo_pin = echo_pin if echo_pin is not None else ECHO_PIN
			self.GPIO.setwarnings(False)
			self.GPIO.setmode(GPIO.BCM)
			self.GPIO.setup(self.trigger_pin, GPIO.OUT)
			self.GPIO.setup(self.echo_pin, GPIO.IN)
			
			# funcao de leitura pra raspberry pi 4
			self.read_func = lambda: self.GPIO.input(self.echo_pin)
			self.GPIO.add_event_detect(self.echo_pin, self.GPIO.BOTH, callback=self._on_rpi_echo)
		
		self.stop = threading.Event()
		# thread de leitura
		self.thread = threading.Thread(target=self._read, daemon=True)
		self.thread.start()

	##############################################
	# Raspberry Pi version detector
	##############################################
	def detect_rpi_version(self):
		try:
			with open('/proc/device-tree/model') as f:
				return 5 if 'Raspberry Pi 5' in f.read() else 4
		except OSError:
			return 4  # Default caso nao consiga detectar
	
	##############################################
	# Detecta gpiochip
	##############################################
	def _find_gpiochip(self):
		try:
			out = subprocess.check_output(["gpiodetect"], text=True)
			for line in out.splitlines():
				if "pinctrl-rp1" in line:
					match = re.match(r"gpiochip(\d+)", line)

					if match:
						return int(match.group(1))

		except (OSError, subprocess.SubprocessError):
			pass

		raise RuntimeError("GPIO chip principal da Raspberry Pi 5 nao encontrado.")

	########################################
	# thread de leitura continua
	########################################
	def _read(self):
		# inicializacao
		time.sleep(0.1)

		# loop de leitura
		while not self.stop.is_set():
			started = time.monotonic()
			self._sample_once()
			self.stop.wait(max(0.0, SAMPLE_TIME - (time.monotonic() - started)))

	def _sample_once(self):
		self._attempt_error = None
		try:
			d = self.get_measure()
		except Exception as exc:
			d = None
			self._attempt_error = f"erro_gpio:{type(exc).__name__}:{exc}"
		with self.lock:
			if d is not None:
				self.dist = d
				self.last_measurement = time.monotonic()
				self.valid = True
				self.last_error = "nenhum"
				self.consecutive_failures = 0
			else:
				self.last_error = self._attempt_error or "eco_invalido"
				self.consecutive_failures += 1
				self.total_failures += 1
	
	########################################
	# Funcao para medir distancia
	########################################			
	def get_distance(self):
		status = self.get_status()
		return status["distance"], status["valid"]

	def get_status(self):
		with self.lock:
			age = (
				time.monotonic() - self.last_measurement
				if self.last_measurement is not None else float("inf")
			)
			self.valid = age <= SENSOR_TIMEOUT
			return {
				"distance": self.dist, "valid": self.valid, "age": age,
				"reason": "ok" if self.valid else (
					"leitura_expirada" if self.last_measurement is not None else "sem_leitura"
				),
				"last_error": self.last_error, "pulse_s": self.last_pulse_s,
				"consecutive_failures": self.consecutive_failures,
				"total_failures": self.total_failures,
				"backend": "lgpio_edges" if self.rpi_version == 5 else "rpi_gpio_edges",
			}
	
	########################################
	# escreve no pino de trigger
	def _set_trigger(self, state):
		# Raspberry Pi 5
		if self.rpi_version == 5:
			self.GPIO.gpio_write(self.handle_chip, self.trigger_pin, 1 if state else 0)
		# Raspberry Pi 3/4
		else:
			self.GPIO.output(self.trigger_pin, True if state else False)
			
	########################################
	# Funcao para medir distancia
	########################################
	def _on_lgpio_echo(self, chip, gpio, level, tick):
		# lgpio fornece o instante da borda em ns, nao o instante do callback.
		self._record_echo(level, tick)

	def _on_rpi_echo(self, channel):
		self._record_echo(self.read_func(), time.monotonic_ns())

	def _record_echo(self, level, tick):
		with self.echo_condition:
			if not self._accepting_echo or tick < self._echo_armed_at_ns:
				return
			if level == 1 and self._echo_start_ns is None:
				self._echo_start_ns = tick
			elif level == 0 and self._echo_start_ns is not None and self._echo_duration_ns is None:
				self._echo_duration_ns = tick - self._echo_start_ns
				self.echo_condition.notify_all()

	def get_measure(self):
		with self.measure_lock:
			self._attempt_error = None
			with self.lock:
				self.last_pulse_s = None
			with self.echo_condition:
				self._echo_start_ns = None
				self._echo_duration_ns = None
				self._echo_armed_at_ns = time.monotonic_ns()
				self._accepting_echo = True
			try:
				if self.read_func() != 0:
					self._attempt_error = "echo_alto_antes_trigger"
					return None
				self._set_trigger(True)
				try:
					time.sleep(0.00002)
				finally:
					self._set_trigger(False)
				with self.echo_condition:
					if not self.echo_condition.wait_for(lambda: self._echo_duration_ns is not None, 0.03):
						self._attempt_error = (
							"timeout_subida_echo" if self._echo_start_ns is None else "timeout_descida_echo"
						)
						return None
					pulse_s = self._echo_duration_ns / 1e9
				with self.lock:
					self.last_pulse_s = pulse_s
				distance = GAIN * pulse_s
				if pulse_s <= 0 or not self.min_range <= distance <= self.max_range:
					self._attempt_error = f"eco_fora_faixa:d={distance:.3f}"
					return None
				return distance
			finally:
				with self.echo_condition:
					self._accepting_echo = False
		
	########################################
	# Limpeza dos pinos
	########################################
	def cleanup(self):
		# Closes the GPIO connection.
		if self.rpi_version == 5:
			self.echo_callback.cancel()
			self.GPIO.gpiochip_close(self.handle_chip)
		else:
			self.GPIO.remove_event_detect(self.echo_pin)
			self.GPIO.cleanup(self.trigger_pin)
			self.GPIO.cleanup(self.echo_pin)

	########################################
	# Destrutor
	########################################
	def close(self):
		# termina a thread
		self.stop.set()
		self.thread.join()
		
		#Ensure cleanup is called when object is deleted
		self.cleanup()

########################################################################
# Teste rapido
########################################################################
if __name__ == '__main__':

	import matplotlib.pyplot as plt
	plt.ion()
	fig = plt.figure(figsize=(8, 4))

	# parametros do teste
	TEST_TIME = 30.0
	PLOT_INTERVAL = 2.0
	PLOT_WINDOW = 5.0

	# dados
	ts = []
	dist = []
	valids = []

	# cria o ultrasom
	us = Ultrasonic()

	try:
		t0 = time.monotonic()
		last_plot = t0

		while (time.monotonic() - t0) <= TEST_TIME:

			# tempo atual
			now = time.monotonic()
			t = now - t0

			# le sensor
			distance, valid = us.get_distance()

			# salva dados
			ts.append(t)
			dist.append(distance)
			valids.append(valid)

			# atualiza grafico a cada 5 segundos
			if (now - last_plot) >= PLOT_INTERVAL:

				# converte para arrays
				t_array = np.array(ts)
				d_array = np.array(dist)
				v_array = np.array(valids)

				# seleciona ultimos 10 segundos
				mask = t_array >= (t - PLOT_WINDOW)

				t_plot = t_array[mask]
				d_plot = d_array[mask]
				v_plot = v_array[mask]

				# apaga
				plt.clf()

				# sinal continuo
				plt.plot(t_plot, d_plot, 'b-', label='Distancia')

				# destaca medidas invalidas
				plt.scatter(t_plot[~v_plot], d_plot[~v_plot], color='red', label='Medida invalida', zorder=3)

				plt.xlabel('Time [s]')
				plt.ylabel('Distance [m]')

				plt.ylim([us.min_range - 0.2, us.max_range + 0.2])

				plt.grid()
				plt.legend()
				plt.show(block=False)
				plt.pause(0.1)

				last_plot = now

			time.sleep(SAMPLE_TIME)

	finally:
		print('Terminou...')
		us.close()
