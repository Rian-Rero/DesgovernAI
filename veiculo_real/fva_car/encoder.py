# -*- coding: utf-8 -*-
########################################
# Disciplina: Topicos em Engenharia de Controle e Automacao IV (ENG075):
# Fundamentos de Veiculos Autonomos - 2026/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automacao
# DELT - Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
import numpy as np
from serial.tools import list_ports
import serial
import threading
import time

########################################
# Globais
########################################
BAUDRATE        = 115200
TIMEOUT         = 0.2
REDUCAO_EIXO    = 7.80
RAIO_RODA       = 0.08
SENSOR_TIMEOUT  = 0.30

########################################
# Calibracao do receptor RC [us]
########################################

# Direcao
RC_DIR_MIN      = 1170
RC_DIR_CENTER   = 1490
RC_DIR_MAX      = 1820

# Acelerador / freio
RC_ACEL_MIN     = 1210
RC_ACEL_CENTER  = 1450
RC_ACEL_MAX     = 1780

# Zona morta do comando normalizado
# +/- 5% em torno do centro
RC_DEADZONE     = 0.05
RC_PULSE_MIN    = 800
RC_PULSE_MAX    = 2200

########################################
# Classe para leitura do Arduino Nano
#
# A thread consome continuamente a serial e
# armazena os ultimos valores recebidos.
#
# Formatos explicitos:
# rpm: RPM (firmware original, controle automatico sem receptor/chaves)
# rc (padrao):
# RPM,RC_DIRECAO,RC_ACELERADOR,SEL_DIRECAO,SEL_TRACAO
#
# Selecao:
# 0 = RC
# 1 = AUTO
########################################
class Encoder:

	########################################
	# construtor
	########################################
	def __init__(self, protocol='rc'):

		# A escolha e explicita: uma mensagem incompleta RC nunca habilita AUTO.
		if protocol not in ('rc', 'rpm'):
			raise ValueError("Protocolo do encoder deve ser 'rc' ou 'rpm'.")
		self.protocol = protocol

		port = self.find_arduino()
		if not port:
			raise RuntimeError("Encoder (arduino nano) nao encontrado!")

		self.ser = serial.Serial(port, BAUDRATE, timeout=TIMEOUT)

		# Aguarda reinicializacao do Nano ao abrir a serial
		time.sleep(1.5)
		self.ser.reset_input_buffer()

		# Ultimos valores validos recebidos
		self.vel = 0.0
		self.rc_direcao = 0
		self.rc_acelerador = 0
		self.sel_direcao = 0
		self.sel_tracao = 0

		# Estado da comunicacao
		self.valid = False
		self.last_measurement = 0.0

		# Sincronizacao entre thread e getters
		self.lock = threading.Lock()

		# Controle da thread
		self.running = True
		self.thread = threading.Thread(
			target=self._serial_loop,
			daemon=True
		)
		self.thread.start()

	########################################
	# thread de leitura da serial
	########################################
	def _serial_loop(self):

		while self.running:

			try:
				line = self.ser.readline()

			except (OSError, serial.SerialException):
				with self.lock:
					self.valid = False
				break

			# Timeout normal da serial: nenhuma linha chegou
			if not line:
				with self.lock:
					if (
						self.last_measurement == 0.0 or
						(time.monotonic() - self.last_measurement) > SENSOR_TIMEOUT
					):
						self.valid = False
				continue

			try:
				texto = line.decode("utf-8").strip()
				campos = texto.split(",")

				if len(campos) != (1 if self.protocol == 'rpm' else 5):
					continue

				rpm_motor = float(campos[0])
				if self.protocol == 'rpm':
					# Firmware original: nao publica receptor nem chaves.
					rc_direcao, rc_acelerador = RC_DIR_CENTER, RC_ACEL_CENTER
					sel_direcao = sel_tracao = 1
				else:
					rc_direcao = int(campos[1])
					rc_acelerador = int(campos[2])
					sel_direcao = int(campos[3])
					sel_tracao = int(campos[4])

				if not np.isfinite(rpm_motor):
					continue

				# Garante que os seletores tenham estados validos
				if sel_direcao not in (0, 1) or sel_tracao not in (0, 1):
					continue

				# RPM do motor -> RPM da roda
				rpm_roda = rpm_motor / REDUCAO_EIXO

				# RPM da roda -> velocidade linear [m/s]
				vel = RAIO_RODA * (np.pi / 30.0) * rpm_roda

				# Atualiza o conjunto inteiro de forma atomica
				with self.lock:
					self.vel = vel
					self.rc_direcao = rc_direcao
					self.rc_acelerador = rc_acelerador
					self.sel_direcao = sel_direcao
					self.sel_tracao = sel_tracao
					self.last_measurement = time.monotonic()
					self.valid = True

			except (ValueError, UnicodeDecodeError):
				# Ignora linha incompleta/corrompida e preserva
				# o ultimo conjunto valido recebido
				continue

	########################################
	# retorna velocidade mais recente
	########################################
	def get_vel(self):

		with self.lock:
			vel = self.vel
			valid = self._fresh()

		return vel, valid

	def _fresh(self):
		"""Chamado sob self.lock; rejeita tambem horarios futuros."""
		age = time.monotonic() - self.last_measurement
		return bool(self.valid and self.last_measurement > 0
					and 0 <= age <= SENSOR_TIMEOUT)

	def control_valid(self):
		"""Odometria atual e pulsos presentes nos eixos controlados por RC."""
		with self.lock:
			return bool(self._fresh()
				and (self.sel_direcao == 1 or RC_PULSE_MIN <= self.rc_direcao <= RC_PULSE_MAX)
				and (self.sel_tracao == 1 or RC_PULSE_MIN <= self.rc_acelerador <= RC_PULSE_MAX))

	########################################
	# retorna modo de controle
	#
	# retorno:
	# sel_direcao, sel_tracao
	#
	# 0 = RC
	# 1 = AUTO
	########################################
	def get_control_mode(self):

		with self.lock:
			return self.sel_direcao, self.sel_tracao

	########################################
	# normaliza um canal RC para [-1, +1]
	#
	# minimum -> -1
	# center  ->  0
	# maximum -> +1
	#
	# Aplica zona morta em torno de zero
	# e reescala o restante para manter
	# toda a faixa [-1, +1].
	########################################
	def normalize_rc(self, value, minimum, center, maximum):

		# Pulso ausente/espurio nao representa comando de re ou estercamento.
		if not np.isfinite(value) or not RC_PULSE_MIN <= value <= RC_PULSE_MAX:
			return 0.0

		# Normalizacao em dois trechos
		if value <= center:
			u = (value - center) / (center - minimum)
		else:
			u = (value - center) / (maximum - center)

		# Limita a faixa
		u = float(np.clip(u, -1.0, 1.0))

		# Zona morta
		if abs(u) <= RC_DEADZONE:
			return 0.0

		# Reescala o restante para evitar salto
		# na saida da zona morta
		if u > 0.0:
			u = (u - RC_DEADZONE) / (1.0 - RC_DEADZONE)
		else:
			u = (u + RC_DEADZONE) / (1.0 - RC_DEADZONE)

		return float(np.clip(u, -1.0, 1.0))

	########################################
	# retorna comandos normalizados do radio
	#
	# retorno:
	# direcao, acelerador
	#
	# -1.0 = minimo
	#  0.0 = centro
	# +1.0 = maximo
	########################################
	def get_rc(self):

		with self.lock:
			if not self._fresh():
				return 0.0, 0.0
			rc_direcao = self.rc_direcao
			rc_acelerador = self.rc_acelerador

		direcao = self.normalize_rc(
			rc_direcao,
			RC_DIR_MIN,
			RC_DIR_CENTER,
			RC_DIR_MAX
		)

		acelerador = self.normalize_rc(
			rc_acelerador,
			RC_ACEL_MIN,
			RC_ACEL_CENTER,
			RC_ACEL_MAX
		)

		return direcao, acelerador

	########################################
	# retorna todos os dados crus
	# util para diagnostico/testes
	#
	# RC permanece em microssegundos aqui.
	########################################
	def get_data(self):

		with self.lock:
			vel = self.vel
			rc_direcao = self.rc_direcao
			rc_acelerador = self.rc_acelerador
			sel_direcao = self.sel_direcao
			sel_tracao = self.sel_tracao
			valid = self._fresh()

		return (
			vel,
			rc_direcao,
			rc_acelerador,
			sel_direcao,
			sel_tracao,
			valid
		)

	########################################
	# detecta automaticamente a porta
	########################################
	def find_arduino(self):

		for p in list_ports.comports():

			desc = p.description.lower()
			hwid = p.hwid.lower()

			if (
				"arduino" in desc or
				"wch" in desc or
				"1a86" in hwid or
				"ftdi" in desc or
				"0403" in hwid
			):
				return p.device

		return None

	########################################
	# fecha comunicacao serial e thread
	########################################
	def close(self):

		self.running = False

		try:
			if self.thread.is_alive():
				self.thread.join(timeout=TIMEOUT + 0.1)
		except RuntimeError:
			pass

		try:
			if self.ser and self.ser.is_open:
				self.ser.close()
		except (OSError, serial.SerialException):
			pass

########################################
# main test
########################################
if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser(description='Teste do encoder sem acionar motores.')
	parser.add_argument('--protocol', choices=('rpm', 'rc'), default='rc',
						help='rpm: firmware original; rc: firmware de cinco campos')
	parser.add_argument('--seconds', type=float, help='Duracao do teste; padrao sem limite')
	args = parser.parse_args()
	if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
		parser.error('--seconds deve ser positivo e finito.')
	enc = Encoder(protocol=args.protocol)

	print()
	print("Teste: odometria + receptor RC + chaves")
	print(f"Protocolo: {args.protocol}")
	print("Chaves: 0 = RC | 1 = AUTO")
	print("RC normalizado: -1.0 a +1.0")
	print("Ctrl+C para sair")
	print()

	try:
		started = time.monotonic()
		while args.seconds is None or time.monotonic() - started < args.seconds:

			(
				vel,
				rc_dir_raw,
				rc_acel_raw,
				sel_dir,
				sel_tracao,
				valid
			) = enc.get_data()

			# Valores normalizados
			rc_dir, rc_acel = enc.get_rc()

			modo_dir = "\033[31mAUTO\033[0m" if sel_dir else "\033[33mRC\033[0m"
			modo_tracao = "\033[31mAUTO\033[0m" if sel_tracao else "\033[33mRC\033[0m"
			status = ("\033[31mINVALIDA\033[0m" if not valid else
					  "\033[32mOK\033[0m" if enc.control_valid() else "RC AUSENTE")
			if not valid:
				modo_dir = modo_tracao = "INDEFINIDO"

			print(
				f"Vel = {vel:.1f} m/s | "
				f"RC Dir = {rc_dir:+.2f} | " #({rc_dir_raw:4d} us) | "
				f"RC Acel = {rc_acel:+.2f} | " #({rc_acel_raw:4d} us) | "
				f"Direcao = {modo_dir:4s} | "
				f"Tracao = {modo_tracao:4s} | "
				f"{status}",
				flush=True
			)

			time.sleep(0.1)

	except KeyboardInterrupt:
		print("\nTeste encerrado.")

	finally:
		enc.close()
