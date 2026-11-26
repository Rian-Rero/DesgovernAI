# -*- coding: utf-8 -*-
"""Stanley no veiculo real: linha amarela ou centro de cones pela camera."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import threading
import time

# Algebra do numpy (OpenBLAS) em 1 thread, ANTES de importar o numpy: as
# threads extras ficam em espera ativa depois de cada conta pequena (as do
# estimador) e, com os nucleos fixados, roubavam o nucleo do auxiliar da
# visao. Medido em 26/11 (2 nucleos): o processo da visao gastava ~60 ms de
# CPU por quadro, com 1 thread ~16; o pipeline subiu de ~24 para 30 q/s (o
# teto da camera). Os processos da visao herdam (spawn).
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from fva_car.cone_debug import ConeDebugRecorder
from fva_car.cones import RealConeDetector
from fva_car.obstacle import curve_echo
from fva_car.version import code_version
from fva_car.odometry import PoseHistory, SharedPoseHistory
from fva_car.vision_process import VisionProcess, capture_func, vision_func  # noqa: F401
from fva_car.lateral import (
    GroundCalibration,
    LineObservation,
    RealLineDetector,
    RealStanleyFollower,
)

MAIN_VEL = 0.7  # m/s: velocidade padrao da missao
FAST_VEL = 1.5  # m/s: selecionada com --fast
MAX_VEL = 4.0  # m/s: maior teto aceito em --speed (o Car corta o throttle acima de 4,5)
# Missao em andamento: PID para a GUI encerrar (SIGTERM) ou liberar a parada do
# ultrassom (SIGUSR1) sem reiniciar o programa.
MISSION_PID_FILE = Path("/tmp/fva_missao.pid")
# Canal da GUI (aba Camera): ela escreve "<acao> <codigo>" em MISSION_CMD_FILE e
# espera "<codigo> <resposta>" em MISSION_ACK_FILE. Nao depende de sinais nem de
# permissao para matar processo, e a GUI sabe se o carro recebeu.
MISSION_CMD_FILE = Path("/tmp/fva_missao.cmd")
MISSION_ACK_FILE = Path("/tmp/fva_missao.ack")
RESUME_REQUEST = threading.Event()
STOP_REQUEST = threading.Event()
TELEMETRY_HZ = 10.0
TELEMETRY_PERIOD = 1.0 / TELEMETRY_HZ
OBSTACLE_DISTANCE = 0.20  # folga desejada ate o obstaculo, a partir da FRENTE do pneu
# O ultrassom fica na linha do eixo dianteiro (embaixo da camera) e a frente do
# pneu 7 cm adiante (medido em 19/11): a leitura do sensor tem 7 cm a mais que a
# folga real da frente do carro.
FRONT_OVERHANG_M = 0.07
STOP_REACTION_TIME = 0.30
# Bipe enquanto parado: (duracoes dos bipes em s, periodo de repeticao em s).
# Ultrassom: dois curtos por segundo (como antes). Placa de PARE (ArUco): um
# longo a cada 1,5 s, para a equipe saber de longe por que o carro parou.
STOP_BEEPS = {"ultrassom": ((0.15, 0.15), 1.0), "aruco": ((0.7,), 1.5)}
STOP_DECELERATION = 1.75  # m/s^2: ajuste experimental ja usado no veiculo
# Abaixo disto a leitura do ultrassom e falha do sensor (ensaio de 10/11: 0,00 m
# com nada a frente); um obstaculo real passa antes pela faixa de parada.
ULTRASONIC_MIN_VALID_M = 0.03

parameters = {
    # Duracao maxima da missao (s): a volta acaba sozinha depois disto (ou antes,
    # pelo botao Encerrar da GUI ou Ctrl+C). --tempo / campo "Tempo da missao" da
    # GUI mudam. 50 s, o mesmo padrao da GUI.
    "ts": 50.0,
    "navigation_mode": "cones",  # somente cones; linha requer selecao explicita
    # Largura/espacamento da pista: so PALPITE inicial (a pista muda a cada dia).
    # O estimador do corredor mede a largura com as duas bordas e passa a usar a
    # aprendida; a checagem da calibracao so acusa erro grosseiro (0,4-2,5x).
    # Largura: 1,7-1,9 m em 23/10, 1,0-2,0 m (mediana 1,4) nos 38 ensaios de 24/11,
    # mudando dentro da mesma volta. 1,4 erra menos no pior caso enquanto o carro
    # so ve uma borda e ainda nao mediu; com as duas, vale a medida.
    "cone_settings": {"roi_top": 0, "single_border_recovery": True,
                      "track_width_cm": 140, "cone_spacing_cm": 50},
    "save": True,
    "logfile": "logs/",
    "camera": True,
    # Ajustes opcionais da webcam (vazio: padrao dela). Para fixar a imagem:
    # {"fourcc": "MJPG", "exposure": <valor>, "auto_wb": False, "wb_temperature": <K>}
    # Descubra os valores com: v4l2-ctl -d /dev/video0 --list-ctrls
    "camera_settings": {},
    "ultrasonic_steering": False,  # camera fixa apos calibrada
    "us_buzzer": False,
    "initial_position": [0, 0, 0.0],
    "encoder_protocol": "rc",  # mesmo formato de cinco campos da upstream/rc-input
    "camera_calibration_file": str(
        Path(__file__).resolve().parent / "calibracao" / "camera_lateral.json"
    ),
    "velocity_reference": MAIN_VEL,
    # Acelerador minimo com vref > 0 (zona morta do ESC; fva_car/car.py,
    # THROTTLE_FF_OFFSET). Tranco na partida: baixe; ainda engasga: suba.
    "throttle_ff_offset": 0.32,
    # Throttle de regime aprendido por faixa de velocidade e salvo entre corridas
    # (fva_car/throttle.py, logs/acelerador.json): o offset acima vira so o valor
    # inicial. --throttle-offset fixa o valor e desliga o aprendizado.
    "throttle_learning": True,
    # Partida como antes de 17/11 (comparacao A/B): velocidade de recuperacao
    # pelo sigma no eixo > 0,1 m, alvo do pure pursuit sem minimo e acelerador
    # 0,17*vref sem offset. Tambem --old-start.
    "old_start": False,
    # Stanley: esterco = rumo + atan(k*e/(ks + v)) + atan(L*kappa), k e ks fixos.
    # Pista virtual com a pista de 17/11 (largura 1,3 m, cones a 1 m, visao a 7 e
    # 11 quadros/s, velocidade medida filtrada): k 1,5 -> 7,8 cm RMS, k 2,5 -> 5,8 cm,
    # sem toques; k 3 ainda melhor no simulador, mas o carro real tem mais ruido
    # (poucos cones por quadro): 2,5 deixa margem. --stanley-gain ajusta na pista.
    "stanley_gain": 2.5,
    # Trajetoria de curva (fva_car/lateral.py, _racing_line_offset): com curva a
    # frente, a referencia vai ate racing_line_m para FORA (abre antes da curva,
    # ve mais cones dela) e volta ao centro dentro dela (racing_inside 1: so abre;
    # 1,5 tambem fecha para dentro). Pista virtual, pista de 1,3 m, cones a 1 m,
    # 12 q/s: folga minima ate os cones 20 -> 29 cm e todas as voltas completas
    # (com ruido e erro de calibracao tambem). Em pista larga (>= 1,7 m) piorava:
    # some sozinho acima de racing_full_width_m. 0 desliga.
    "racing_line_m": 0.15,
    "racing_inside": 1.0,
    # Tracado de corrida "apice" (24/11): chega por fora (racing_line_m), converge
    # para o apice (racing_apex_m para dentro) e compensa o arrasto da traseira
    # (L^2*kappa/2, ~5 cm em R = 1,25 m). Pista virtual, curvas fechadas, carro
    # com 85% de resposta e 220 ms: folga mediana aos cones de DENTRO 31,5 ->
    # 35,8 cm (1,3 m) e 46,9 -> 50,4 cm (1,6 m), todas as voltas. Apice para dentro
    # (0,05) com 0,25 para fora tirou folga (25 cm): fica 0. "antigo": formula
    # anterior. --apice M para testar.
    "racing_mode": "apice",
    # Memoria do tracado (s): um quadro novo substitui o caminho ja identificado
    # aos poucos (fva_car/lateral.py _remember_path). Ensaios de 24/11: cada
    # quadro discordava do anterior em ~0,15 rad e o esterco trocava de lado 2x
    # por segundo. Pista virtual (ruido do carro real, 10 q/s): pedaladas 44 ->
    # 13 e 42 -> 19, mesmas voltas e o mesmo erro. 0,4-0,6 s cortava mais, mas
    # perdia voltas. Custo: com objetos falsos AO LADO da pista (iscas), a
    # correcao de um salto do estimador tambem demora: 0,97 -> 0,93 das voltas
    # (15 iscas) e 19 -> 21 toques (6 iscas). Exigir 2 quadros concordando antes
    # de mudar foi pior (pedaladas 48 -> 39 so). 0 desliga (--memoria-tracado 0).
    "path_memory_s": 0.25,
    "racing_apex_m": 0.0,
    "stanley_softening": 0.3,
    # Velocidade do termo de deslocamento do Stanley, atan(k*e/(ks + v)), com k e
    # ks fixos: a MEDIDA pelo encoder, filtrada (1a ordem, stanley_speed_filter_s).
    # A vref nao e a velocidade real (partida, curvas, frenagem); a do encoder
    # sozinha oscila e faria o ganho efetivo oscilar. True volta a vref.
    "stanley_use_vref": False,
    "stanley_speed_filter_s": 0.3,
    "stanley_max_steer_deg": 20.0,
    # Escala do comando do servo por lado (1 = igual). A roda pode virar mais para
    # um lado que para o outro com o mesmo comando: em 19/11 as curvas a esquerda
    # ficaram mais fechadas que as a direita. --esterco-esquerda/--esterco-direita
    # ou os campos da GUI; tools/alinhamento_camera.py mede pelos logs.
    "steering_left_scale": 1.0,
    "steering_right_scale": 1.0,
    # IMU (MPU-9250 no barramento I2C 4): guinada MEDIDA na odometria em vez do
    # modelo da direcao (fva_car/yaw_rate.py). Sem o chip, queimado ou falhando
    # na volta, o Car volta sozinho ao modelo. Na pista virtual (resposta da
    # direcao 0,80-0,89, 10 q/s): erro de 7,6-8,7 para 6,2-6,5 cm e, com teto de
    # 2,8 m/s, 3 toques para 0. --sem-imu desliga.
    "imu": True,
    "imu_bus": 4,
    "imu_gyro_range_dps": 500,  # +-500 graus/s: 3 m/s numa curva de 1 m dao ~170
    # Amortecimento do Stanley pela guinada medida (s): esterco -= k*(guinada -
    # v*curvatura). Nos ensaios de 24/11 o esterco "pedalava" a ~2 Hz, puxado pelo
    # termo de rumo (2-4 graus nessa faixa contra 1-2 do deslocamento): o
    # giroscopio ve a oscilacao na hora, a imagem 270 ms depois. Pista virtual
    # (servo 0,1 s, 220-320 ms, 16 sementes x 6 pistas): pedaladas 14 -> 8 a
    # 0,7 m/s e 12 -> 6 a 1,6 m/s, erro menor, mesmas voltas. Sem IMU valida o
    # termo e zero. --amortecer-guinada 0 desliga.
    "stanley_yaw_damping": 0.2,
    # Guinada real / guinada do modelo cinematico, na odometria (previsao do
    # estimador entre quadros e compensacao do atraso). Medido nos logs de 19/11
    # pelos cones parados (tools/alinhamento_camera.py): 0,88 a esquerda e 0,90
    # a direita; com 1,0 a odometria girava ~11% a mais que o carro.
    "steering_response": 0.89,
    # Atraso da direcao (s): comando -> guinada no giroscopio. Ensaio de 26/11
    # (13 s de oscilacao, atraso puro + 1a ordem ajustados): 0,16-0,20 s. O
    # Stanley mira o estado em que o carro estara quando o esterco de agora
    # chegar a roda (fva_car/lateral.py _preview_actuator). Pista virtual com
    # esse atraso e a calibracao atual (5 pistas x 5 sementes, 1,5 m/s):
    # antes 96% das voltas, 10,5 cm, 1040 pedaladas; com o historico de poses
    # 96%, 5,4 cm, 4; com os dois 100%, 4,6 cm, 2. 0 desliga.
    "steer_delay_s": 0.16,
    # 60 graus/s levava 0,33 s (0,5 m a 1,5 m/s) para chegar ao batente:
    # quase todo o trecho visivel antes da curva.
    "stanley_max_rate_deg_s": 120.0,
    # Em cones, libera o esterco progressivamente nos primeiros 24 cm em movimento.
    "cone_steering_start_distance_m": 0.24,
    # Esterco de regime atan(L*kappa) somado ao Stanley; 0 desliga.
    "stanley_curvature_ff": 1.0,
    "wheelbase": 0.36,  # m: mesmo CAR["L"] de fva_car/car.py
    # Passa-baixas (s) do limite de velocidade em curva; 0 desliga. Evitam os
    # trancos de acelerador ("engasgos"); o esterco nao passa por eles.
    "curvature_filter_s": 0.2,
    # 0,3 s deixava a vref variar 0,19-0,47 m/s em ~1 s (log 20261023_080437).
    "speed_filter_s": 0.6,
    # Leva os erros da imagem ao instante do comando por odometria (ate 0,25 s).
    "vision_latency_compensation": True,
    "steering_sign": 1.0,  # convencao do Car real: positivo para esquerda
    "steering_filter_size": 3,  # Stanley ja limita a taxa; reduz atraso extra do servo
    "curve_min_speed": 0.15,
    "curve_lateral_accel": 0.4,
    "vision_timeout": 1.0,  # segundos desde o inicio da captura
    # Perdendo a faixa, segue a ultima trajetoria vista (odometria) por ate
    # este numero de quadros seguidos sem deteccao (~16-20 quadros/s) antes de
    # parar; 0 volta a parar no primeiro quadro sem faixa.
    "line_loss_frames": 8,
    # Teto em segundos da mesma tolerancia, caso a visao pare de entregar quadros.
    "line_loss_tolerance": 1.0,
    # Durante a perda, soma esterco para o lado da curva (ou da faixa) a esta
    # taxa, ate o batente: ~18 graus apos 8 quadros. 0 so segue a trajetoria.
    "line_search_rate_deg_s": 40.0,
    # Cones: perdendo o corredor, segue a trajetoria prevista por odometria por
    # ate cone_loss_frames quadros (e cone_loss_tolerance s), reduzindo a vref
    # em rampa ate curve_min_speed; depois neutraliza. 0 para no 1o quadro.
    "cone_loss_frames": 6,
    "cone_loss_tolerance": 0.6,
    # 0,12 m ficava abaixo do ruido metrico dos cones a 2-3 m: 31 "saltos"
    # descartados em 14 s, cada um forcando hold e rampa de vref.
    "cone_max_lateral_jump_m": 0.25,
    "cone_max_heading_jump_rad": 0.35,
    # Debug por quadro em logs/<ensaio>/cones_debug.jsonl + frames JPG so em eventos.
    "cone_debug": True,
    "cone_max_event_frames": 300,
    # Coleta para rotular/avaliar (tools/rotular_cones.py): quadros originais em
    # logs/<ensaio>/frames/ a esta taxa (Hz). 0 desliga; nao usar na prova.
    "record_frames_hz": 0.0,
    # Rede que confirma cada candidato a cone (tools/cone_classifier.py):
    # caminho do .onnx treinado por tools/treinar_classificador_cones.py, com o
    # .json ao lado. None: so cor, forma e tamanho (como antes).
    "cone_classifier": None,
    # Estimador do corredor (fva_car/corridor.py): posicao, rumo, curvatura e
    # largura filtrados pela odometria, com as duas cores num so ajuste. True
    # liga com os ajustes padrao; dicionario muda ajustes de CorridorSettings
    # (ex.: {"memory_m": 0.6}); False volta ao ajuste quadro a quadro.
    "cone_estimator": True,
    # Fita amarela no centro do corredor de cones (pista de 10/11): entra no
    # estimador como linha central. DESLIGADA: em 17/11 a pista nao tinha fita, mas
    # o corredor tem uma linha amarela pintada no chao que nao e o centro da pista
    # (estado LINHA em 1-16% dos quadros), e a busca custava ~105 ms no Raspberry
    # a cada 2 quadros (geometria 124 ms contra 19 ms sem ela). --fita-amarela liga.
    "cone_line_fusion": False,
    # Lei de direcao com o estimador: Stanley no centro estimado (padrao, pedido do
    # professor) ou pure pursuit (alvo a pursuit_min_lookahead_m +
    # pursuit_lookahead_s*v; --pure-pursuit). Linha amarela: sempre Stanley.
    "steering_law": "stanley",
    "pursuit_min_lookahead_m": 0.4,
    "pursuit_lookahead_s": 0.2,
    "pursuit_max_lookahead_m": 1.5,
    # Previa de curva: o carro nao freia (rola a ~0,5-0,7 m/s^2, logs de 20/10),
    # entao reduz a vref ANTES da curva vista a ate speed_preview_m. A vref sobe
    # no maximo speed_rise_m_s2 e desce na hora (substitui speed_filter_s nos
    # cones). Pista virtual a 1,5 m/s: aceleracao lateral max 0,7 -> 0,4 m/s^2,
    # volta 3-7% mais lenta. 0 desliga cada um.
    "speed_preview_m": 2.5,
    "speed_rise_m_s2": 1.0,
    "coast_deceleration": 0.5,
    # Limite de visibilidade: a camera ve ~2,5 m e o carro so rola para reduzir;
    # depois do trecho visto supoe uma curva de raio min_curve_radius_m (a mais
    # fechada da pista). Assim velocity_reference vira a velocidade MAXIMA NAS
    # RETAS: o carro acelera ate o que ainda deixa rolar ate essa curva (~1,6 m/s
    # com R = 1,5 m) e reduz antes das curvas vistas. Pista virtual, vref 2,0:
    # volta 26-46% mais rapida que a 0,7 m/s, sem toques (tools/pista_virtual.py).
    # Pista dos ensaios de 10/11: curvas de raio ~1,2 m (kappa 0,8 no estimador).
    "min_curve_radius_m": 1.2,
    # Curva vista por uma cor so (a borda de dentro saiu da imagem): esterco x1,15
    # e volante 1,5x mais rapido. Pista virtual: 1,1-1,2 neutro a melhor, 1,3/2,0
    # ja passava do ponto. 1 desliga.
    "one_border_curve_steer_gain": 1.15,
    "one_border_curve_rate_gain": 1.5,
    # Parada pelo ultrassom em curva: ignora o eco quando a visao mostra que ele e
    # o cone da borda e o carro ja esta virando para longe dele (fva_car/obstacle.py).
    # Obstaculo no meio da faixa, reta ou sem corredor: para como antes.
    "ultrasonic_curve_filter": True,
    # Eco curto precisa durar isto (s; ultrassom le a cada ~0,12 s) para travar a
    # parada: 9 das 13 rodadas de 10/11 pararam por uma leitura isolada falsa.
    # Entra tambem na distancia de parada. 0 volta a parar na primeira leitura.
    "obstacle_confirm_s": 0.15,
    # Velocidade limitada pelo atraso da imagem: v <= isto / idade media (m). Com
    # 0,3 s de atraso (ensaios de 10/11) da ~1 m/s; com a visao rapida, nao limita.
    "latency_speed_budget_m": 0.5,
    # Nucleos do Pi 4: controle (e gravacao/transmissao) em 0-1, processo de visao
    # em 2-3, sem um roubar o nucleo do outro (fva_car/system_health.py). So com
    # a visao em processo e 4+ nucleos. False: o sistema escolhe (--sem-nucleos).
    "cpu_cores": {"control": [0, 1], "vision": [2, 3]},
    # Temperatura, clock e limitacao do Pi a cada 2 s, no resumo da volta.
    "health_monitor": True,
    # Placas ArUco (DICT_5X5_100, como camera.py; fva_car/aruco.py). Acao por ID:
    # "pare" (ID 1, placa de PARE AO LADO da pista): reduz a approach_speed e
    # para com a frente do carro na altura da placa (stop_at_m; negativo = antes).
    # A posicao vem da imagem: direcao pela homografia e distancia pelo tamanho
    # do marcador. marker_m = LADO DO QUADRADO PRETO impresso (medir com regua!);
    # a altura estimada do centro da placa sai no terminal e confere a medida.
    # Placa na frente: a visao antecipa a parada em stop_gap_m (do pneu ao
    # cone). O ultrassom mantem a distancia de seguranca normal.
    # Fica parado hold_s segundos (5) e segue sozinho, ignorando a mesma placa
    # por ignore_s; o botao "Liberar parada" da GUI libera antes. hold_s None:
    # so pelo botao. Bipe proprio (um bipe LONGO, STOP_BEEPS), diferente do
    # ultrassom (dois curtos).
    # "ignorar": so registra. every: busca a cada N quadros. False desliga.
    "aruco": {"every": 2, "actions": {"1": "pare"}, "approach_speed": 0.3, "stop_gap_m": 0.08,
              # marker_height_m: altura do CENTRO do marcador (m). Com ela, a posicao
              # nao depende do tamanho (melhor para placa baixa). None: pelo tamanho.
              "marker_m": 0.10, "marker_height_m": None, "stop_at_m": 0.0, "front_half_width_m": 0.25,
              "hold_s": 5.0, "ignore_s": 4.0},
    # Depois de "Liberar parada", o ultrassom fica ignorado por isto (s): o carro
    # quase sempre parou encostado no cone, e sem a carencia travava de novo em
    # 0,15 s (o bipe nao parava e o botao parecia nao funcionar).
    "obstacle_resume_grace_s": 2.0,
    # Teto pela distancia de parada no alcance do ultrassom (4 m): ~2,8 m/s.
    # False tira o teto (pista fechada, sem pessoas/paredes na frente).
    "ultrasonic_speed_cap": True,
    # Centralizacao: em reta (duas bordas) o carro aprende o centro do SERVO
    # (trim), igual para Stanley e pure pursuit; a odometria continua com o
    # esterco pretendido. Nos ensaios de 10/11 o carro andava 2-12 cm a direita.
    # Aprende pelo esterco MEDIO na reta (seguindo reta, a roda real fica em 0):
    # sem IMU, o vies da direcao entorta tambem a odometria e o estimador, que
    # ve o carro "no centro" com ele a 11 cm (pista virtual, 2 graus de vies).
    # Pista virtual, 4 pistas, vies de +-2 graus, visao a 7 q/s: 3,8-8,7 cm RMS
    # pelo esterco (0,3) contra 5,2-12,3 pelo erro lateral (centering_gain 0,15).
    # O trim aprendido e salvo em logs/direcao_trim.json e a corrida seguinte
    # comeca dele.
    "centering_steer_gain": 0.3,     # 1/s: trim += 0,3 * esterco comandado * dt, so em reta
    "centering_gain": 0.0,           # rad por (m*s) do erro lateral estimado (antigo: 0,15)
    # Trim inicial se ainda nao houver o arquivo. Nos ensaios de 17/11 a roda virava
    # ~3,5 graus mais para a direita que o comando (tools/alinhamento_camera.py,
    # "curva nao vista"); o trim aprendido la saiu com o sinal errado (-2 graus)
    # porque a percepcao estava girada 11 graus. Comeca em +3 e o aprendizado ajusta.
    "steering_trim_deg": 3.0,
    # Teto do trim aprendido (graus): 4 nao alcancava o vies medido.
    "centering_limit_deg": 6.0,
    "steering_trim_learning": True,  # False: trim fixo (steering_trim_deg), sem aprender
    # Camera e detector em processo proprio (outro nucleo, sem disputar o GIL
    # com o controle a 50 Hz). False volta as threads no mesmo processo.
    "vision_process": True,
    # Processos auxiliares da visao (so cones, so com vision_process): a etapa
    # SEM estado do quadro (reduzir, achar e classificar candidatos, refinar
    # bases; ~57% do tempo) roda num processo enquanto o processo de visao faz
    # a etapa com estado (rastreio, estimador, mapa, ArUco) do quadro anterior.
    # Mesmo resultado quadro a quadro, ~1,7x mais quadros por segundo nos 2
    # nucleos da visao. 0 volta ao sequencial (--sem-pipeline).
    "vision_workers": 1,
    # Largura do processamento dos cones (0 = 640 original). 320: deteccao ~2,5x mais
    # rapida (no Pi ~95 -> ~35 ms; ensaios de 10/11 rodaram a 6-9 quadros/s com
    # 260-290 ms de atraso). Nos 617 quadros reais de 10/11, em relacao a 640:
    # 98% dos cones ate 1,5 m, 94% ate 2,2 m e 86% ate 3 m. 480: 99/97/91%, so 28%
    # mais rapido. A linha amarela usa a vista de cima e ignora isto.
    "vision_width": 320,
    # Base dos cones a mais de 1,3 m refeita na imagem cheia (so uma janela em volta
    # de cada um; fva_car/cones.py refine_bases): nos quadros reais de 10/11 a
    # distancia a 1,7-2,6 m passou a bater com a de 640 px (p90 da diferenca
    # 8-14 cm -> 4-5 cm) por +0,7 ms/quadro no notebook. False desliga.
    "cone_refine_bases": True,
    # Mapa de cones (EKF-SLAM, fva_car/cone_slam.py) + tracado de margem maxima
    # (SVM, fva_car/svm_path.py): com o estimador perdido (sem cones aceitos alem
    # da memoria), o carro segue devagar pelos cones ja mapeados, inclusive os ao
    # lado e atras, em vez de parar. "substituir": o tracado do mapa vale sempre
    # (na pista virtual piorou: 6,7 -> 9-13 cm rms). False desliga.
    "cone_map": True,
    # Mesmo modo de reconstrucao do detector atualizado em tools/yellow_path.py.
    # ROI 0: usa a imagem inteira, como no detector HSV de referencia.
    "path_settings": {"s_min": 25, "dashed": True, "dash_gap": 120, "roi_top": 50},
    "path_anchor": 0.5,  # preferencia horizontal da faixa, entre 0 e 1
    # Linha amarela com a camera ALTA (a dos cones): a faixa e detectada na vista
    # de cima do chao (fva_car/lateral.py BirdsEyeView, 0,5-3 m, 1 cm/px), onde
    # fica como na camera inclinada para baixo, e seguida pelo estimador do
    # corredor como linha central (clotoide + odometria), com pure pursuit,
    # previa de curva e limite de visibilidade como nos cones. Na curva fechada
    # a esquerda a linha sai da imagem (camera 8 graus a direita): memoria maior
    # e curvatura mais estavel que nos cones. Pista virtual (--linha): RMS de
    # 1,4-3,8 cm com pure pursuit. False nos dois volta ao modo antigo (camera
    # inclinada para baixo: --line-classic).
    "line_birdseye": True,
    "line_estimator": {"memory_m": 2.5, "memory_s": 5.0, "kappa_noise": 0.05, "dkappa_noise": 0.1},
    # Porta HTTP com a visao ao vivo (http://<ip-do-carro>:porta/); 0 desliga.
    "stream_port": 0,
}


def corridor_log(corridor):
    """Colunas do car.csv com o estado do estimador (NaN quando desligado)."""
    from fva_car.corridor import STATES

    nan = float("nan")
    if not corridor:
        return dict(corridor_state=nan, corridor_sigma_e=nan, corridor_speed_factor=nan)
    sigma = corridor.get("sigma") or [None]
    return dict(corridor_state=float(STATES.index(corridor["state"])),
                corridor_sigma_e=nan if sigma[0] is None else float(sigma[0]),
                corridor_speed_factor=float(corridor.get("speed_factor", nan)))


def line_estimator_settings(config):
    """line_estimator do config -> argumento estimator do RealLineDetector (None desliga)."""
    value = config.get("line_estimator", None)
    if value in (None, False):
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("line_estimator deve ser True, False ou um dicionario de ajustes.")
    return {"pose_offset_m": float(config.get("wheelbase", 0.36)), **value}


def corridor_settings(config):
    """cone_estimator do config -> argumento corridor do RealConeDetector (None desliga)."""
    value = config.get("cone_estimator", True)
    if value in (None, False):
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("cone_estimator deve ser True, False ou um dicionario de ajustes.")
    # A pose do Car e do eixo traseiro; os cones tem origem no eixo dianteiro.
    # A largura NAO e fixa: --largura-pista so muda o palpite inicial
    # (cone_settings.track_width_cm); o estimador mede e acompanha a largura com
    # as duas bordas. known_width_m (trava +-15%) so se posto aqui de proposito.
    settings = {"pose_offset_m": float(config.get("wheelbase", 0.36)), **value}
    if config.get("old_start", False):
        # Criterio antigo: devagar com sigma no eixo > 0,1 m e alvo sem minimo.
        settings.update(slow_sigma_e=0.1, slow_sigma_lateral=99.0, near_sigma_e=99.0)
    return settings


def cone_calibration_warnings(calibration, detector):
    """Avisos de cobertura da homografia para a faixa em que os cones sao usados."""
    from tools.cone_geometry import calibration_report

    settings = getattr(detector, "settings", None)
    max_range = getattr(settings, "max_range_cm", 300)
    if isinstance(max_range, bool) or not isinstance(max_range, (int, float)):
        max_range = 300
    try:
        return calibration_report(calibration, max_range_m=max_range / 100)["warnings"]
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        return [f"nao foi possivel verificar a calibracao: {error}"]


# Trim da direcao aprendido entre corridas (fva_car/lateral.py, centralizacao).
# Em logs/: a limpeza do envio da GUI nao apaga e o Coletar Dados traz junto.
# v2: o direcao_trim.json antigo foi aprendido com a percepcao girada (calibracao
# de 17/11 antes da correcao de 19/11) e esta com o sinal errado; nao e mais lido.
TRIM_FILE = Path(__file__).resolve().parent / "logs" / "direcao_trim_v2.json"
TRIM_MIN_STRAIGHT_S = 3.0  # so salva se aprendeu em pelo menos isto de reta
TRIM_BLEND = 0.5           # cada corrida move o trim salvo so metade do que aprendeu


def load_steering_trim(config, path=TRIM_FILE):
    """Trim inicial (graus): o salvo pelas corridas anteriores, senao steering_trim_deg."""
    base = float(config.get("steering_trim_deg", 0.0) or 0.0)
    if not config.get("steering_trim_learning", True):
        return base
    try:
        value = float(json.loads(Path(path).read_text(encoding="utf-8"))["steering_trim_deg"])
    except (OSError, ValueError, KeyError, TypeError):
        return base
    return value if math.isfinite(value) and abs(value) <= 10.0 else base


def save_steering_trim(follower, config, path=TRIM_FILE):
    """Grava o trim para a proxima corrida; devolve o valor salvo ou None."""
    learned_time = float(getattr(follower, "centering_time", 0.0) or 0.0)
    if not config.get("steering_trim_learning", True) or learned_time < TRIM_MIN_STRAIGHT_S:
        return None
    base = float(config.get("steering_trim_deg", 0.0) or 0.0)
    learned = math.degrees(float(follower.centering))
    value = round(base + TRIM_BLEND * learned, 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "steering_trim_deg": value, "anterior_deg": round(base, 2),
        "aprendido_nesta_corrida_deg": round(learned, 2), "reta_s": round(learned_time, 1),
        "salvo_em": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n", encoding="utf-8")
    return value


# Throttle de regime aprendido (fva_car/throttle.py): em logs/, como o trim.
THROTTLE_FILE = Path(__file__).resolve().parent / "logs" / "acelerador.json"
THROTTLE_MIN_STEADY_S = 3.0  # so salva com pelo menos isto de regime na corrida


def load_throttle_map(path=THROTTLE_FILE):
    """Mapa salvo pelas corridas anteriores (dict) ou None."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_throttle_map(car, config, path=THROTTLE_FILE):
    """Grava o mapa do throttle para a proxima corrida; devolve o mapa ou None."""
    throttle_map = getattr(car, "throttle_map", None)
    if not config.get("throttle_learning", False) or throttle_map is None:
        return None
    if throttle_map.learned_s < THROTTLE_MIN_STEADY_S:
        return None
    data = dict(throttle_map.to_dict(), regime_nesta_corrida_s=round(throttle_map.learned_s, 1),
                salvo_em=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return throttle_map


def now_print_ok(last, now, period=1.0):
    """No maximo uma mensagem por periodo (last: lista de um elemento, mutavel)."""
    if now - last[0] >= period:
        last[0] = now
        return True
    return False


def stopping_distance(velocity, confirm_s=0.0):
    """Distancia minima do obstaculo para parar a tempo, contando a espera da confirmacao."""
    speed = abs(velocity)
    return (
        OBSTACLE_DISTANCE
        + FRONT_OVERHANG_M
        + speed * (STOP_REACTION_TIME + confirm_s)
        + speed**2 / (2.0 * STOP_DECELERATION)
    )


def ultrasonic_speed_cap(confirm_s=0.0, sensor_range=4.0, margin=0.3):
    """Maior velocidade cuja distancia de parada cabe no alcance do ultrassom.

    Mais rapido que isto, um obstaculo so e visto quando ja nao da para parar
    antes dele. Com 4 m de alcance, 0,3 s de reacao + 0,15 s de confirmacao e
    1,75 m/s^2 de frenagem: ~2,8 m/s."""
    usable = sensor_range - margin
    low, high = 0.0, 10.0
    for _ in range(50):
        middle = (low + high) / 2
        low, high = (middle, high) if stopping_distance(middle, confirm_s) <= usable else (low, middle)
    return low


def save_health(summary, recorder=None, folder=None):
    """Saude do Pi na volta: resumo de cones, saude_raspberry.json e terminal."""
    if recorder is not None:
        recorder.notes["raspberry"] = summary
    if isinstance(folder, (str, Path)):
        try:
            Path(folder, "saude_raspberry.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    temp, low = summary.get("temp_max_c"), summary.get("freq_min_mhz")
    print("Raspberry na volta:"
          + (f" ate {temp:.0f} C" if temp is not None else "")
          + (f" | clock minimo {low:.0f} MHz" if low is not None else "")
          + (" | limitado: " + ", ".join(summary["problems_during_run"]) if summary.get("problems_during_run")
             else " | sem limitacao"), flush=True)
    for warning in summary.get("warnings", []):
        print(f"AVISO Raspberry: {warning}", flush=True)
    return summary


def save_imu(car, recorder=None, folder=None, config=None):
    """Resumo da IMU na volta (resposta da direcao medida) no log e no terminal."""
    gyro = getattr(car, "gyro", None)
    status = getattr(car, "imu_status", None)
    if gyro is None and status is None:
        return None  # carro simulado/injetado
    summary = gyro.summary() if gyro is not None else {"ativa": False, "motivo": status}
    if recorder is not None:
        recorder.notes["imu"] = summary
    if isinstance(folder, (str, Path)):
        try:
            Path(folder, "imu.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    if gyro is None:
        print(f"IMU: nao usada nesta volta ({status})", flush=True)
        return summary
    response = summary["resposta_direcao"]
    text = ", ".join(f"{side} {value:.2f}" for side, value in response.items() if value is not None)
    print(f"IMU: {'ativa' if summary['ativa'] else 'DESLIGADA no meio da volta (' + summary['motivo'] + ')'}"
          f" | {summary['leituras_usadas']} leituras | {summary['falhas_i2c']} falhas I2C"
          + (f" | resposta da direcao: {text}" if text else ""), flush=True)
    values = [v for v in response.values() if v is not None]
    current = float((config or {}).get("steering_response", 1.0))
    if values and abs(sum(values)/len(values) - current) > 0.05:
        print(f"IMU: a roda fez {sum(values)/len(values):.2f} da curva pedida (steering_response hoje"
              f" {current:.2f}); so importa sem a IMU.", flush=True)
    left, right = response["esquerda"], response["direita"]
    if left and right and abs(left - right) > 0.08:
        # A resposta medida ja inclui as escalas em uso: igualar os lados na media deles.
        target = (left + right) / 2
        scales = [float(np.clip(100 * float((config or {}).get(key, 1.0)) * target / value, 50, 150))
                  for key, value in (("steering_left_scale", left), ("steering_right_scale", right))]
        print(f"IMU: direcao assimetrica (esq {left:.2f}, dir {right:.2f}): --esterco-esquerda {scales[0]:.0f}"
              f" --esterco-direita {scales[1]:.0f} igualaria os lados.", flush=True)
    return summary


def aruco_every(config):
    """Busca de placas a cada N quadros (0: desligado)."""
    value = config.get("aruco")
    if not value:
        return 0
    return max(1, int((value if isinstance(value, dict) else {}).get("every", 2)))


def aruco_marker_m(config):
    """Lado do quadrado preto dos marcadores (m); 0: sem posicao das placas."""
    value = config.get("aruco")
    if not value:
        return 0.0
    return float(value.get("marker_m", 0.10) or 0.0) if isinstance(value, dict) else 0.10


def aruco_marker_height_m(config):
    """Altura do centro dos marcadores (m), ou None (posicao pelo tamanho)."""
    value = config.get("aruco")
    height = value.get("marker_height_m") if isinstance(value, dict) else None
    return None if height is None else float(height)


def sign_controller(config, mode):
    """SignController das placas ArUco pela configuracao, ou None."""
    value = config.get("aruco")
    if mode != "cones" or not value:
        return None
    from fva_car.aruco import SignController

    value = value if isinstance(value, dict) else {}
    return SignController(
        actions={int(k): v for k, v in (value.get("actions") or {"1": "pare"}).items()},
        approach_speed=float(value.get("approach_speed", 0.3)),
        stop_gap_m=float(value.get("stop_gap_m", 0.08)),
        front_overhang_m=FRONT_OVERHANG_M,
        reaction_s=STOP_REACTION_TIME + float(config.get("obstacle_confirm_s", 0.0) or 0.0),
        deceleration=float(config.get("coast_deceleration", 0.5)),
        hold_s=value.get("hold_s"),
        ignore_s=float(value.get("ignore_s", 4.0)),
        stop_at_m=float(value.get("stop_at_m", 0.0)),
        wheelbase=float(config.get("wheelbase", 0.36)),
        front_half_width_m=float(value.get("front_half_width_m", 0.25)),
        max_frame_age=float(config.get("vision_timeout", 1.0)),
    )


def read_mission_command(path=None):
    """(acao, codigo) pedida pela GUI, ou None. O arquivo e consumido."""
    path = MISSION_CMD_FILE if path is None else path
    try:
        text = Path(path).read_text(encoding="utf-8").split()
        Path(path).unlink()
    except (OSError, UnicodeDecodeError):
        return None
    if not text:
        return None
    return text[0], (text[1] if len(text) > 1 else "")


def answer_mission_command(code, message, path=None):
    """Resposta para a GUI (troca atomica: ela nunca le pela metade)."""
    path = MISSION_ACK_FILE if path is None else path
    try:
        temporary = Path(str(path) + ".tmp")
        temporary.write_text(f"{code} {message}\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass


def write_mission_pid(path=MISSION_PID_FILE):
    try:
        Path(path).write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError as error:
        print(f"Aviso: PID da missao nao salvo ({error}); botoes da GUI nao alcancam esta missao.", flush=True)


def remove_mission_pid(path=MISSION_PID_FILE):
    """Apaga o arquivo so se ainda e desta missao (outra pode ter comecado)."""
    try:
        if Path(path).read_text(encoding="utf-8").split()[0] == str(os.getpid()):
            Path(path).unlink()
    except (OSError, IndexError):
        pass


def stop_vehicle(car):
    """Neutro imediato: set_vel(0) sozinho poderia conservar integral do PI."""
    car._set_ref(0.0)
    car.vel_error_integral = 0.0
    car.set_u(0.0)
    car.set_steer(0.0)


def save_vision_snapshot(car, observation, status, now):
    """Salva o ultimo frame somente apos neutralizar os motores."""
    folder = getattr(car, "logfile", None)
    if not isinstance(folder, (str, Path)) or observation is None:
        return
    folder = Path(folder)

    def finite(value):
        return float(value) if value is not None and np.isfinite(value) else None

    try:
        for name, frame in (
            ("vision_camera.jpg", observation.source_frame),
            ("vision_detection.jpg", observation.frame),
        ):
            if frame is not None:
                if name == "vision_detection.jpg":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(folder / name), frame):
                    raise OSError(f"Nao foi possivel salvar {name}")
        data = {
            "code_version": code_version(),
            "control_status": status,
            "vision_reason": observation.reason,
            "metric_line_valid": bool(observation.valid),
            "image_size": (
                None
                if observation.source_frame is None
                else list(observation.source_frame.shape[1::-1])
            ),
            "processing_size": observation.processing_size,
            "vision_frame_id": observation.sequence,
            "vision_age_s": finite(now - observation.timestamp),
            "capture_s": finite(observation.capture_seconds),
            "capture_clock": getattr(observation, "capture_clock", ""),
            "processing_s": finite(observation.processing_seconds),
            "detection_s": finite(observation.detection_seconds),
            "geometry_s": finite(observation.geometry_seconds),
            "errors": None if observation.errors is None else list(observation.errors),
            "color_support": finite(observation.color_support),
            "cone_debug": getattr(observation, "debug", None),
            "encoder_valid": bool(getattr(car, "encoder_valid", False)),
            "sel_direcao": getattr(car, "sel_direcao", None),
            "sel_tracao": getattr(car, "sel_tracao", None),
        }
        from fva_car.cone_debug import clean

        (folder / "vision_status.json").write_text(
            json.dumps(clean(data), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Diagnostico de visao salvo em {folder}", flush=True)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Nao foi possivel salvar diagnostico de visao: {error}", flush=True)


def stream_text(car, status, observation):
    """Legenda do quadro transmitido: o que o controle decidiu neste passo."""
    lines = [status[:48]]
    if len(status) > 48:  # tolerancia de perda: quadros, tempo e fator da vref
        lines.append(status[48:110])
    if observation is not None:
        lines.append("Referencia: " + ("cones" if observation.reference_source == "cones" else "linha amarela"))
    if observation is not None and observation.reference_source == "cones":
        lines.append(f"Cones: verdes={len(observation.green_pixels)} vermelhos={len(observation.red_pixels)}")
        debug = getattr(observation, "debug", None) or {}
        sides = debug.get("sides") or {}
        if sides:
            lines.append(f"Lados: verde={'E' if sides.get('green_left') else 'D'}"
                         f" ({'auto' if sides.get('auto') else 'manual'} {sides.get('score') or 0:+.2f})")
        trajectory = debug.get("trajectory") or {}
        if trajectory:
            text = {"reta": "reta", "curva_esq": "curva esq", "curva_dir": "curva dir"}.get(
                trajectory.get("segment"), "?")
            if trajectory.get("radius_m") is not None and trajectory.get("segment") != "reta":
                text += f" R={abs(trajectory['radius_m']):.1f}m"
            if trajectory.get("curve_ahead_m"):
                text += f" | curva em {trajectory['curve_ahead_m']:.1f}m"
            if debug.get("branch"):
                text += " | BIFURCACAO"
            lines.append("Trecho: " + text)
    if observation is not None and observation.valid:
        lateral, heading, curvature = observation.errors
        lines.append(f"e={lateral:+.3f}m psi={np.degrees(heading):+.1f} k={curvature:+.2f}")
    lines.append(f"v={car.v:.2f} vref={car.vref:.2f} st={np.degrees(car.st):+.1f}")
    return lines


def render_stream_frame(detector, observation, lines):
    """Quadro transmitido: so a regiao que o detector usa (abaixo do ROI), com a
    legenda numa faixa preta acima. O horizonte nao e transmitido: JPEG menor,
    menos CPU no Pi e nada escondendo os cones."""
    image = detector.preview_bgr(observation)
    if image is None:
        width, height = detector.processing_size
        image = np.zeros((height, width, 3), np.uint8)
    roi_row = getattr(detector, "roi_row", None)
    top = 0
    if callable(roi_row):
        try:
            top = int(np.clip(roi_row(), 0, image.shape[0] - 2))
        except (TypeError, ValueError):
            top = 0
    view = image[top:]
    band = np.zeros((6 + 14 * len(lines), image.shape[1], 3), np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(band, line, (4, 14 + 14 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, view]) if lines else view.copy()


def follower_config(config, mode, corridor):
    """Ajustes do RealStanleyFollower para o modo (os de cones sobrepoem os da linha)."""
    config = dict(config)
    if mode != "cones" and corridor is None:
        # Linha no modo antigo (ajuste quadro a quadro): Stanley e vref antiga.
        config["steering_law"] = "stanley"
        config["speed_preview_m"] = 0.0
        config["speed_rise_m_s2"] = 0.0
        return config
    if mode != "cones":
        # Linha com o estimador: mesmo caminho dos cones (pure pursuit, previa,
        # limite de visibilidade); a memoria do estimador cobre as perdas.
        radius = float(config.get("min_curve_radius_m", 0) or 0)
        config["speed_sight_curvature"] = 1.0 / radius if radius > 0 else 0.0
        config["line_loss_frames"] = min(config.get("line_loss_frames", 8), 2)
        config["line_loss_tolerance"] = min(config.get("line_loss_tolerance", 1.0), 0.3)
        return config
    # Sem chaveamento para linha nem esterco de busca: durante a tolerancia
    # apenas continua a trajetoria ja identificada, com a vref em rampa.
    config["line_loss_frames"] = config.get("cone_loss_frames", 6)
    config["line_loss_tolerance"] = config.get("cone_loss_tolerance", 0.6)
    if corridor is not None:
        # O estimador ja segue a memoria por ate memory_m; a tolerancia do
        # Stanley so cobre quadros sem visao, sem somar outro trecho as cegas.
        config["line_loss_frames"] = min(config["line_loss_frames"], 2)
        config["line_loss_tolerance"] = min(config["line_loss_tolerance"], 0.3)
    else:
        # Pure pursuit e previa de curva precisam do caminho do estimador; o
        # ajuste quadro a quadro continua com Stanley e o filtro de vref antigo.
        config["steering_law"] = "stanley"
        config["speed_preview_m"] = 0.0
        config["speed_rise_m_s2"] = 0.0
    radius = float(config.get("min_curve_radius_m", 0) or 0)
    config["speed_sight_curvature"] = 1.0 / radius if radius > 0 and corridor is not None else 0.0
    config["line_search_rate_deg_s"] = 0.
    config["steering_start_distance_m"] = config.get("cone_steering_start_distance_m", 0.)
    config["loss_speed_ramp"] = True
    config["max_lateral_jump_m"] = config.get("cone_max_lateral_jump_m", 0.12)
    config["max_heading_jump_rad"] = config.get("cone_max_heading_jump_rad", 0.35)
    return config


def with_safety_speed(config):
    """safe_speed_m_s do seguidor pelo ultrassom (ultrasonic_speed_cap)."""
    config = dict(config)
    if config.get("ultrasonic_speed_cap", True):
        config["safe_speed_m_s"] = ultrasonic_speed_cap(float(config.get("obstacle_confirm_s", 0.0) or 0.0))
        if config.get("velocity_reference", 0.0) > config["safe_speed_m_s"]:
            print(f"Velocidade: teto {config['velocity_reference']:.1f} m/s pedido; o ultrassom (4 m) so"
                  f" para a tempo ate {config['safe_speed_m_s']:.1f} m/s, que vira o teto efetivo"
                  " (--sem-teto-ultrassom tira)", flush=True)
    return config


def run_mission(settings=None, display=False, car_factory=None, clock=time.monotonic):
    config = dict(parameters)
    if settings is not None:
        config.update(settings)
    if config.get("old_start", False):
        config["throttle_ff_offset"], config["throttle_ff_gain"] = 0.0, 0.17  # acelerador de antes de 17/11
        config["throttle_learning"] = False
    if config.get("throttle_learning", False) and car_factory is None and "throttle_map" not in config:
        config["throttle_map"] = load_throttle_map()
    if not config["camera"] or config["ultrasonic_steering"]:
        raise ValueError(
            "Stanley requer camera ligada e fixa (ultrasonic_steering=False)."
        )
    if config["encoder_protocol"] not in ("rpm", "rc"):
        raise ValueError("encoder_protocol deve ser 'rpm' ou 'rc'.")
    if (
        isinstance(config["steering_filter_size"], bool)
        or not isinstance(config["steering_filter_size"], int)
        or config["steering_filter_size"] < 1
    ):
        raise ValueError("steering_filter_size deve ser inteiro positivo.")
    # Valida geometria e ajustes antes de abrir sensores ou atuar nos motores.
    calibration = GroundCalibration.from_file(config["camera_calibration_file"])
    version = code_version()
    print(f"Codigo: versao {version}", flush=True)
    mode = config.get("navigation_mode", "cones")
    if mode == "cones":
        corridor = corridor_settings(config)
        detector = RealConeDetector(
            calibration, settings=config.get("cone_settings"), render_preview=False,
            processing_width=config.get("vision_width", 0),
            classifier=config.get("cone_classifier"), corridor=corridor,
            line_fusion=bool(config.get("cone_line_fusion", False)),
            refine_bases=bool(config.get("cone_refine_bases", True)),
            cone_map=config.get("cone_map", False),
            aruco_every=aruco_every(config),
            aruco_marker_m=aruco_marker_m(config),
            aruco_marker_height_m=aruco_marker_height_m(config),
        )
        if detector.classifier is not None:
            print(f"Cones: classificador {detector.classifier.path}", flush=True)
        calibration_warnings = cone_calibration_warnings(calibration, detector)
        for warning in calibration_warnings:
            print(f"AVISO calibracao: {warning}", flush=True)
        config = follower_config(config, mode, corridor)
    elif mode == "line":
        detector = RealLineDetector(
            calibration,
            path_settings=config.get("path_settings", {}),
            anchor=config.get("path_anchor", 0.5),
            render_preview=False,
            processing_width=config.get("vision_width", 0),
            birdseye=bool(config.get("line_birdseye", False)),
            estimator=line_estimator_settings(config),
        )
        config = follower_config(config, mode, line_estimator_settings(config))
    else:
        raise ValueError("navigation_mode deve ser line ou cones")
    if car_factory is None:  # carro real: comeca com o trim aprendido nas corridas anteriores
        config["steering_trim_deg"] = load_steering_trim(config)
        print(f"Direcao: trim {config['steering_trim_deg']:+.1f} graus"
              + (" (aprendendo nas retas)" if config.get("steering_trim_learning", True) else ""), flush=True)
    if not config.get("steering_trim_learning", True):
        config["centering_gain"] = config["centering_steer_gain"] = 0.0
    config = with_safety_speed(config)
    follower = RealStanleyFollower(config)
    stream = None
    if config.get("stream_port"):
        from fva_car.stream import VisionStream

        # Abre a porta antes dos motores: porta ocupada falha sem mover o carro.
        stream = VisionStream(
            lambda observation, lines: render_stream_frame(detector, observation, lines),
            port=config["stream_port"],
        )
    vision_data = {"observation": None}
    # Carro real: camera e detector em outro processo, fora do GIL do laco de
    # controle. Um carro injetado (testes, simulacao) fornece as proprias
    # imagens por get_image(), entao a visao continua em threads.
    use_process = bool(config.get("vision_process", True)) and car_factory is None
    control_cores = vision_cores = None
    if use_process:
        from fva_car.system_health import core_layout, pin_process

        control_cores, vision_cores = core_layout(config)
        # Antes de criar as threads de transmissao, gravacao e recepcao: elas herdam.
        control_cores = pin_process(control_cores) if control_cores else None
    poses = None
    if mode == "cones" or line_estimator_settings(config) is not None:
        # A visao leva o mapa de cones (ou a linha) pela odometria registrada pelo controle.
        poses = SharedPoseHistory() if use_process else PoseHistory()
        # O seguidor leva a foto ao instante do comando pelo movimento REAL
        # registrado (historico de poses, com a guinada do giroscopio). Antes
        # usava a guinada de agora x idade da foto: com a direcao ~0,18 s
        # atrasada (ensaio de 26/11, guinada do giroscopio contra o comando) o
        # carro balancava a ~1,7 Hz e a guinada de agora nao representava o giro
        # desde a foto, realimentando a oscilacao (esterco de lado a lado).
        # Pista virtual com esse atraso: 1007-1341 pedaladas -> 3-8, erro
        # 8,8-17 cm -> 5 cm, em todas as latencias de imagem testadas.
        follower.odometry = poses
        if not use_process:
            detector.odometry = poses
    vision = None
    if use_process:
        if mode == "cones":
            detector_kwargs = dict(settings=config.get("cone_settings"),
                                   processing_width=config.get("vision_width", 0),
                                   classifier=config.get("cone_classifier"),
                                   corridor=corridor_settings(config),
                                   line_fusion=bool(config.get("cone_line_fusion", False)),
                                   refine_bases=bool(config.get("cone_refine_bases", True)),
                                   cone_map=config.get("cone_map", False),
                                   aruco_every=aruco_every(config),
                                   aruco_marker_m=aruco_marker_m(config),
                                   aruco_marker_height_m=aruco_marker_height_m(config))
        else:
            detector_kwargs = dict(path_settings=config.get("path_settings", {}),
                                   anchor=config.get("path_anchor", 0.5),
                                   processing_width=config.get("vision_width", 0),
                                   birdseye=bool(config.get("line_birdseye", False)),
                                   estimator=line_estimator_settings(config))
        vision = VisionProcess(vision_data, mode, config["camera_calibration_file"], detector_kwargs,
                               camera_settings=config.get("camera_settings") or {}, poses=poses,
                               cpu_cores=vision_cores,
                               workers=int(config.get("vision_workers", 0) or 0) if mode == "cones" else 0)
        try:
            # Abre a camera antes dos motores: falha cedo sem mover o carro.
            vision.start()
        except BaseException:
            if stream is not None:
                stream.close()
            raise
        print(f"Visao: processo separado (pid {vision.process.pid})"
              + (f" | nucleos: controle {control_cores}, visao {getattr(vision, 'cpu_cores', None)}"
                 if control_cores or getattr(vision, "cpu_cores", None) else ""), flush=True)
    if car_factory is None:
        from fva_car import Car

        car_factory = Car
    try:
        # Com a visao em processo, a camera pertence a ele, nao ao Car.
        car = car_factory(dict(config, camera=False) if use_process else config)
    except BaseException:
        if vision is not None:
            vision.stop()
        if stream is not None:
            stream.close()
        raise
    recorder = None
    folder = getattr(car, "logfile", None)
    record_hz = float(config.get("record_frames_hz", 0.0) or 0.0)
    if (mode == "cones" and (config.get("cone_debug", True) or record_hz > 0) and config["save"]
            and isinstance(folder, (str, Path))):
        try:
            recorder = ConeDebugRecorder(folder, max_event_frames=config.get("cone_max_event_frames", 300),
                                         record_hz=record_hz)
            recorder.notes["calibration_warnings"] = calibration_warnings
            recorder.notes["code_version"] = version
            # Opcoes desta volta, para o relatorio comparar voltas (A/B).
            recorder.notes["options"] = {
                "classifier": bool(config.get("cone_classifier")),
                "refine_bases": bool(config.get("cone_refine_bases", True)),
                "cone_map": config.get("cone_map", False),
                "old_start": bool(config.get("old_start", False)),
                "throttle_ff_offset": float(config.get("throttle_ff_offset", 0.32)),
                "throttle_learning": bool(config.get("throttle_learning", False)),
                "velocity_reference": float(config.get("velocity_reference", 0.0)),
                "cpu_cores": {"control": control_cores,
                              "vision": getattr(vision, "cpu_cores", None)},
            }
        except OSError as error:
            print(f"Debug de cones desativado: {error}", flush=True)
    stop_event = threading.Event()
    frames = queue.Queue(maxsize=1)
    thread_vision = None
    thread_capture = None
    plot = None
    observation = None
    control_status = "Missao ainda nao iniciada"
    last_control_at = None
    health = None
    if config.get("health_monitor", True) and car_factory is None:
        from fva_car.system_health import HealthMonitor

        health = HealthMonitor().start()
        first = health.samples[0]
        print("Raspberry: " + " | ".join(part for part in (
            f"{first['temp_c']:.0f} C" if first.get("temp_c") is not None else "",
            f"governor {first['governor']}" if first.get("governor") else "",
            ("LIMITADO desde que ligou: " + ", ".join(first["problems_since_boot"]))
            if first.get("problems_since_boot") else "sem limitacao") if part), flush=True)
    try:
        car.start_mission()
        print(
            f"{'Pure pursuit' if config.get('steering_law') == 'pure_pursuit' else 'Stanley'} {mode}"
            f" | {config['ts']:.0f} s | teto {config['velocity_reference']:.2f} m/s"
            + (f" | previa de curva {float(config['speed_preview_m']):.1f} m"
               if config.get("speed_preview_m") else "")
            + (f" | curva mais fechada R={1/float(config['speed_sight_curvature']):.1f} m"
               if config.get("speed_sight_curvature") else ""),
            flush=True,
        )
        print(f"Encoder: protocolo {config['encoder_protocol']}", flush=True)
        print(
            (
                "Stanley: velocidade de referencia (vref)"
                if config["stanley_use_vref"]
                else f"Stanley: velocidade medida filtrada (1a ordem, {config.get('stanley_speed_filter_s', 0):g} s)"
            ),
            flush=True,
        )
        print(
            f"Direcao: filtro de {config['steering_filter_size']} amostras", flush=True
        )
        if vision is None:
            thread_capture = threading.Thread(
                target=capture_func,
                args=(car, frames, stop_event, clock),
                daemon=True,
                name="captura-camera",
            )
            thread_capture.start()
            thread_vision = threading.Thread(
                target=vision_func,
                args=(car, vision_data, stop_event, detector, clock),
                kwargs={"frames": frames},
                daemon=True,
                name="linha-hsv",
            )
            thread_vision.start()
        if stream is not None:
            stream.start()
            print(
                f"Camera ao vivo: http://{socket.gethostname()}.local:{stream.port}/"
                f" (ou http://<ip-do-carro>:{stream.port}/)",
                flush=True,
            )
        if display:
            import matplotlib.pyplot as plt

            plt.ion()
            figure, axis = plt.subplots()
            plot = (plt, figure, axis)
        t_plot = clock()
        t_telemetry = t_plot - TELEMETRY_PERIOD
        obstacle_detected = False
        echo_prints = [-math.inf]
        echo_since = None  # inicio do eco curto ainda nao confirmado (tempo da missao)
        next_stop_beep = 0.0
        last_encoder_valid = None
        t_diagnostic = t_plot - 1.0
        last_diagnostic_sequence = 0
        commands_enabled = bool(config.get("mission_commands", car_factory is None))
        if car_factory is None:
            write_mission_pid()
        if commands_enabled:
            for stale in (MISSION_CMD_FILE, MISSION_ACK_FILE):
                try:
                    stale.unlink()  # comando de uma missao anterior nao vale para esta
                except OSError:
                    pass
        signs = sign_controller(config, mode)
        command_check = -math.inf
        resume_until = -math.inf
        sign = None
        previous_sign_state, previous_sign_in_path = "LIVRE", False
        while car.t < config["ts"]:
            if not car.step():
                break
            if commands_enabled and clock() - command_check >= 0.1:
                command_check = clock()
                request = read_mission_command()
                if request is not None:
                    action, code = request
                    if action == "liberar":
                        RESUME_REQUEST.set()
                        answer_mission_command(code, "liberar recebido pelo carro")
                    elif action == "encerrar":
                        STOP_REQUEST.set()
                        answer_mission_command(code, "encerrar recebido pelo carro: salvando logs")
                    else:
                        answer_mission_command(code, f"acao desconhecida: {action}")
            if STOP_REQUEST.is_set():
                STOP_REQUEST.clear()
                print("Missao encerrada pela interface.", flush=True)
                break
            if poses is not None:
                try:
                    poses.append(clock(), float(car.p[0]), float(car.p[1]), float(car.th))
                except (TypeError, ValueError, IndexError):
                    pass
            encoder_valid = bool(getattr(car, "encoder_valid", True))
            if encoder_valid != last_encoder_valid:
                print(
                    (
                        "Encoder: leitura OK."
                        if encoder_valid
                        else "Encoder sem leitura valida: comandos em neutro. Confira o protocolo e a serial."
                    ),
                    flush=True,
                )
                last_encoder_valid = encoder_valid
            dist, valid = car.get_distance()
            confirm_s = float(config.get("obstacle_confirm_s", 0.0) or 0.0)
            stop_distance = stopping_distance(car.v, confirm_s)
            observation = vision_data["observation"]
            if signs is not None:
                try:
                    pose_now = (float(car.p[0]), float(car.p[1]), float(car.th))
                except (TypeError, ValueError, IndexError):
                    pose_now = None
                sign = signs.update(observation, clock(), pose_now,
                                    poses.pose_at if poses is not None else None, float(car.v))
                if sign.message:
                    print(sign.message, flush=True)
                if previous_sign_state == "PARADO" and sign.state == "SAINDO":
                    # Liberou sozinha (hold_s). Placa ao lado: a mesma carencia do
                    # botao, para o ultrassom nao travar no cone da borda. Placa
                    # num cone A FRENTE: sem carencia (seguir bateria nele): o
                    # ultrassom trava a parada, com o bipe dele.
                    grace = 0.0 if previous_sign_in_path else float(config.get("obstacle_resume_grace_s", 2.0) or 0.0)
                    resume_until = float(car.t) + grace
                    echo_since = None
                    print(f"PARE (ArUco {sign.marker_id}): {signs.hold_s:.0f} s parado, seguindo"
                          + (f"; ultrassom ignorado por {grace:.1f} s" if grace > 0 else "") + ".", flush=True)
                previous_sign_state = sign.state
                if sign.state in ("APROXIMANDO", "PARADO"):
                    previous_sign_in_path = bool(sign.in_path)
                if sign.stop:
                    if not math.isfinite(sign.remaining_m):
                        reason = "placa confirmada sem posicao confiavel"
                    else:
                        location = "cone a frente" if sign.in_path else "placa lateral"
                        reason = f"{location}, faltam {100*sign.remaining_m:.0f} cm ate o ponto de parada"
                    print(f"PARE (ArUco {sign.marker_id}) acionado pela visao ({reason}).", flush=True)
            echo_ignored = False
            close = (valid and np.isfinite(dist) and ULTRASONIC_MIN_VALID_M <= dist <= stop_distance)
            if RESUME_REQUEST.is_set():
                # Botao "Liberar parada" da GUI (arquivo de comando ou SIGUSR1):
                # destrava a parada e ignora o ultrassom por obstacle_resume_grace_s,
                # para o carro sair de perto do cone em que parou.
                RESUME_REQUEST.clear()
                sign_stopped = sign is not None and sign.state == "PARADO"
                if sign_stopped:
                    print(signs.release(clock()), flush=True)
                    sign = replace(sign, state="SAINDO", speed_limit=math.inf, stop=False)
                if obstacle_detected or sign_stopped:
                    grace = float(config.get("obstacle_resume_grace_s", 2.0) or 0.0)
                    resume_until = float(car.t) + grace
                    obstacle_detected = False
                    echo_since = None
                    print("Parada liberada pela interface"
                          + (f"; ultrassom ignorado por {grace:.1f} s para sair de perto"
                             + (f" (le {dist:.2f} m)" if close else "") if grace > 0 else "")
                          + ".", flush=True)
                else:
                    print("Liberar parada: nenhuma parada estava travada.", flush=True)
            if float(car.t) < resume_until:
                close = False  # carencia depois de liberar (obstacle_resume_grace_s)
            sign_stopped = sign is not None and sign.state == "PARADO"
            if not close or sign_stopped:
                echo_since = None
            if not obstacle_detected and not sign_stopped and close:
                reason = ""
                frontal_sign = sign is not None and sign.state == "APROXIMANDO" and sign.in_path
                if mode == "cones" and config.get("ultrasonic_curve_filter", True) and not frontal_sign:
                    # Em curva, ja virando, o eco pode ser o cone da borda de fora:
                    # a visao diz se ele esta fora do caminho (fva_car/obstacle.py).
                    echo_ignored, reason = curve_echo(
                        float(dist), observation,
                        float(config["steering_sign"]) * float(getattr(car, "st", 0.0)),
                        float(car.v), clock())
                if echo_ignored:
                    echo_since = None
                    if now_print_ok(echo_prints, clock()):
                        print(f"Ultrassom a {dist:.2f} m ignorado: {reason}.", flush=True)
                elif echo_since is None and confirm_s > 0:
                    # Leitura isolada curta e ruido do sensor (ensaios de 10/11: 0,1-0,2 m
                    # do nada, com o carro parado oscilando 0,16/0,78/4,0): so para se
                    # o eco continuar por confirm_s (~2 leituras).
                    echo_since = float(car.t)
                if not echo_ignored and (confirm_s <= 0 or float(car.t) - echo_since >= confirm_s):
                    if frontal_sign and signs.stopped(clock()):
                        sign = replace(sign, state="PARADO", speed_limit=0.)
                        sign_stopped = True
                        print(f"PARE (ArUco {sign.marker_id}) acionado pelo ultrassom a {dist:.2f} m do cone.",
                              flush=True)
                    else:
                        obstacle_detected = True
                        print(
                            f"Obstaculo a {dist:.2f} m: parada travada (botao Liberar parada na GUI)"
                            + (f" ({reason})." if reason else "."),
                            flush=True,
                        )
            stop_required = obstacle_detected or sign_stopped
            if sign is not None:
                follower.external_speed_limit = sign.speed_limit
            manual = (
                getattr(car, "sel_direcao", 1) == 0
                and getattr(car, "sel_tracao", 1) == 0
            )
            if (
                manual
                and encoder_valid
                and not car.emergencia
                and not stop_required
                and np.isfinite(car.v)
                and np.isfinite(car.dt)
                and car.dt > 0
            ):
                # Car aplica os dois canais RC; a perda da faixa nao impede
                # controle manual. Obstaculos e emergencia conservam prioridade.
                follower.reset()
                car.set_steer(0.0)
                car.set_vel(0.0)
                car.log(lateral_active=0)
                control_status = "Controle manual RC"
            else:
                follower.apply(
                    car, observation, clock(), obstacle_detected=stop_required
                )
                control_status = f"Parada ArUco {sign.marker_id} travada" if sign_stopped else follower.status
            if mode == "cones" and observation is not None:
                debug = getattr(observation, "debug", None) or {}
                sides = debug.get("sides") or {}
                trajectory = debug.get("trajectory") or {}
                segment = {"reta": 0, "curva_esq": 1, "curva_dir": -1}.get(trajectory.get("segment"))
                car.log(navigation_cones=int(observation.reference_source == "cones"),
                        cone_red_count=len(getattr(observation, "red_pixels", ())),
                        cone_green_count=len(getattr(observation, "green_pixels", ())),
                        corridor_width_m=getattr(observation, "corridor_width_m", float("nan")),
                        cone_green_left=float(sides["green_left"]) if "green_left" in sides else float("nan"),
                        cone_estimated=float(bool(debug.get("estimated"))),
                        cone_branch=float(bool(debug.get("branch"))),
                        cone_segment=float("nan") if segment is None else float(segment),
                        **corridor_log(debug.get("corridor")))
            if recorder is not None and observation is not None:
                recorder.record(observation, dict(
                    {"active": False} if manual else follower.last_state, status=control_status, manual=bool(manual),
                    obstacle=bool(stop_required), aruco_state=None if sign is None else sign.state,
                    aruco_id=None if sign is None else sign.marker_id,
                    aruco_remaining_m=None if sign is None else sign.remaining_m,
                    t=float(car.t), v=float(car.v),
                    vref=float(car.vref), steer_cmd_rad=float(car.st)))
            car.log(
                obstacle_ignored=int(echo_ignored),
                obstacle_stop=int(stop_required),
                aruco_state=-1 if sign is None else ("LIVRE", "APROXIMANDO", "PARADO", "SAINDO").index(sign.state),
                aruco_id=-1 if sign is None else sign.marker_id,
                aruco_remaining_m=math.nan if sign is None else sign.remaining_m,
                distance_m=dist,
                stopping_distance_m=stop_distance,
            )
            if stream is not None:
                stream.publish(observation, stream_text(car, control_status, observation))
            now = clock()
            last_control_at = now
            if not (vision.is_alive() if vision is not None else thread_vision.is_alive()):
                stop_vehicle(car)
                print("Visao encerrada; missao interrompida.", flush=True)
                break
            if now - t_diagnostic >= 1.0:
                timing = ""
                sequence = 0 if observation is None else observation.sequence
                vision_fps = (sequence - last_diagnostic_sequence) / (
                    now - t_diagnostic
                )
                if observation is not None:
                    timing = (
                        f" | imagem={now-observation.timestamp:.3f}s"
                        f" captura={observation.capture_seconds:.3f}s"
                        f" processamento={observation.processing_seconds:.3f}s"
                        f" deteccao={observation.detection_seconds:.3f}s"
                        f" geometria={observation.geometry_seconds:.3f}s"
                    )
                    if observation.processing_size:
                        timing += f" grade={observation.processing_size[0]}x{observation.processing_size[1]}"
                print(
                    f"CONTROLE: {control_status}{timing} visao={vision_fps:.1f}fps"
                    f" | vref={car.vref:.2f} u={car.u:.2f}",
                    flush=True,
                )
                t_diagnostic = now
                last_diagnostic_sequence = sequence
            if stop_required and now >= next_stop_beep:
                # Bipe diz POR QUE parou: placa (um longo) ou ultrassom (dois curtos).
                pattern, period = STOP_BEEPS["ultrassom" if obstacle_detected else "aruco"]
                if car.bz.beep(list(pattern), silence=0.15):
                    next_stop_beep = now + period
            # O controle roda em todo passo; somente a telemetria usa 10 Hz.
            if now - t_telemetry >= TELEMETRY_PERIOD:
                print(
                    f"DATA,{car.t:.3f},{car.p[0]:.3f},{car.p[1]:.3f},"
                    f"{car.v:.3f},{car.vref:.3f},{car.a:.3f},{car.u:.3f},"
                    f"{car.w:.3f},{car.th:.3f}",
                    flush=True,
                )
                t_telemetry = now
            if plot is not None and now - t_plot >= 1.0:
                plt, _, axis = plot
                preview = detector.with_preview(observation)
                if preview is not None and preview.frame is not None:
                    axis.clear()
                    axis.imshow(preview.frame)
                    axis.set_title(observation.reason or "Linha amarela HSV")
                    axis.axis("off")
                    plt.pause(0.001)
                t_plot = now
    except KeyboardInterrupt:
        print("\nMissao interrompida pelo usuario.", flush=True)
    finally:
        # Neutraliza antes de esperar a thread, inclusive se cap.read estiver bloqueado.
        try:
            stop_vehicle(car)
        finally:
            if car_factory is None:
                remove_mission_pid()
            stop_event.set()
            if stream is not None:
                stream.close()
            # Diagnosticos nunca impedem o resto do encerramento (gravacao, visao).
            try:
                if health is not None:
                    save_health(health.stop(), recorder, folder)
                save_imu(car, recorder, folder, config)
            except Exception as error:  # noqa: BLE001
                print(f"Diagnostico do fim da volta falhou: {error}", flush=True)
            if recorder is not None:
                recorder.close()
            if vision is not None:
                vision.stop()
            if thread_vision is not None:
                thread_vision.join(timeout=1.0)
            if thread_capture is not None:
                thread_capture.join(timeout=1.0)
            try:
                if config["save"]:
                    save_vision_snapshot(
                        car,
                        detector.with_preview(observation),
                        control_status,
                        last_control_at,
                    )
                if config["save"] and getattr(car, "traj", None):
                    car.save()
            finally:
                if car_factory is None:
                    try:
                        saved = save_steering_trim(follower, config)
                        if saved is not None:
                            print(f"Direcao: trim aprendido nesta corrida"
                                  f" {math.degrees(follower.centering):+.1f} graus; proxima comeca em"
                                  f" {saved:+.1f} graus ({TRIM_FILE.name}).", flush=True)
                    except (OSError, TypeError, ValueError) as error:
                        print(f"Trim da direcao nao salvo: {error}", flush=True)
                    try:
                        learned = save_throttle_map(car, config)
                        if learned is not None:
                            print(f"Acelerador: regime aprendido ({learned.learned_s:.0f} s nesta corrida):"
                                  f" {learned.summary()} ({THROTTLE_FILE.name}).", flush=True)
                    except (OSError, TypeError, ValueError) as error:
                        print(f"Mapa do acelerador nao salvo: {error}", flush=True)
                car.close()
                if plot is not None:
                    plot[0].close(plot[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", help="JSON da camera calibrada em metros")
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument(
        "--speed",
        type=float,
        default=MAIN_VEL,
        help=f"Teto de velocidade em m/s (padrao {MAIN_VEL})",
    )
    speed.add_argument(
        "--fast",
        action="store_const",
        dest="speed",
        const=FAST_VEL,
        help=f"Atalho para --speed {FAST_VEL}",
    )
    parser.add_argument(
        "--encoder-protocol",
        choices=("rpm", "rc"),
        help="rpm: Arduino original; rc: Arduino com receptor e chaves",
    )
    parser.add_argument(
        "--display", action="store_true", help="Mostrar a faixa detectada"
    )
    parser.add_argument(
        "--vision-width",
        type=int,
        help="Largura do processamento: 0 original, 320 ou 480 para reduzir custo",
    )
    parser.add_argument(
        "--steering-filter",
        type=int,
        help="Amostras do filtro da direcao: padrao 3; 1 desativa a media movel",
    )
    parser.add_argument(
        "--loss-frames",
        type=int,
        metavar="N",
        help="Quadros seguidos sem faixa/cones em que segue a ultima trajetoria (padrao 8 linha, 6 cones)",
    )
    parser.add_argument(
        "--stream",
        type=int,
        nargs="?",
        const=8080,
        metavar="PORTA",
        help="Transmitir a visao ao vivo por HTTP (padrao 8080), para o navegador do notebook",
    )
    parser.add_argument(
        "--record-frames",
        type=float,
        metavar="HZ",
        help="Coleta: grava quadros originais em logs/<ensaio>/frames/ a HZ (ex.: 5)",
    )
    parser.add_argument(
        "--no-cone-debug",
        action="store_true",
        help="Nao gravar cones_debug.jsonl nem frames de eventos",
    )
    parser.add_argument("--classifier", metavar="MODELO.onnx",
                        help="Rede que confirma os cones (com o .json ao lado); 'none' desliga")
    parser.add_argument("--no-estimator", action="store_true",
                        help="Cones: erros pelo ajuste quadro a quadro, sem o estimador do corredor")
    parser.add_argument("--stanley", action="store_true",
                        help="Cones: Stanley no centro estimado (o padrao; mantido por compatibilidade)")
    parser.add_argument("--stanley-gain", type=float, metavar="K",
                        help="Ganho k do termo de deslocamento do Stanley (padrao 2.5): oscila, baixe;"
                             " passa reto da borda, suba")
    parser.add_argument("--esterco-esquerda", type=float, metavar="PCT",
                        help="Comando do servo nas curvas a ESQUERDA, em %% do pedido (padrao 100; vira demais: baixe)")
    parser.add_argument("--esterco-direita", type=float, metavar="PCT",
                        help="Comando do servo nas curvas a DIREITA, em %% do pedido (padrao 100)")
    parser.add_argument("--abrir-curva", type=float, metavar="M",
                        help="Quanto a referencia vai para fora antes de uma curva (padrao 0.10 m; 0 desliga;"
                             " so age em pista estreita)")
    parser.add_argument("--pure-pursuit", action="store_true",
                        help="Cones: pure pursuit no centro estimado em vez do Stanley")
    parser.add_argument("--fita-amarela", action="store_true",
                        help="Cones: usa a fita amarela do centro da pista junto com os cones")
    parser.add_argument("--no-preview", action="store_true",
                        help="Cones: sem previa de curva (vref so pela curvatura no eixo)")
    parser.add_argument("--line-classic", action="store_true",
                        help="Linha amarela no modo antigo (camera inclinada para baixo): sem vista de cima nem estimador")
    parser.add_argument("--no-trim-learning", action="store_true",
                        help="Direcao: nao aprende nem usa o trim salvo (usa steering_trim_deg)")
    parser.add_argument("--strict-ultrasonic", action="store_true",
                        help="Para por qualquer eco do ultrassom, inclusive o cone da borda na curva")
    parser.add_argument("--min-radius", type=float, metavar="R",
                        help="Cones: raio (m) da curva mais fechada da pista; limita a velocidade nas retas"
                             " (padrao 1.2; 0 desliga)")
    parser.add_argument("--throttle-offset", type=float, metavar="U",
                        help="Acelerador minimo fixo com o carro andando (zona morta do ESC). Sem esta"
                             " opcao, o throttle e aprendido e salvo entre corridas (logs/acelerador.json)."
                             " Tranco na partida: baixe; engasga: suba")
    parser.add_argument("--no-refine", action="store_true",
                        help="Cones: sem o refino da base dos cones longe na imagem cheia (comparacao)")
    parser.add_argument("--sem-teto-ultrassom", action="store_true",
                        help="Sem o teto de ~2,8 m/s pela distancia de parada no alcance do ultrassom")
    parser.add_argument("--largura-pista", type=float, metavar="M",
                        help="Palpite inicial da largura (m). Nao trava: o carro mede e acompanha a largura")
    parser.add_argument("--apice", type=float, metavar="M",
                        help="Na curva, ir ate M metros para dentro (apice); padrao 0")
    parser.add_argument("--tempo", type=float, metavar="S",
                        help=f"Duracao maxima da missao em s (padrao {parameters['ts']:.0f}; encerre pela GUI)")
    parser.add_argument("--memoria-tracado", type=float, metavar="S",
                        help="Memoria do tracado contra o esterco 'pedalando' (s; 0 desliga; padrao 0.25)")
    parser.add_argument("--sem-aruco", action="store_true",
                        help="Nao procura placas ArUco (PARE)")
    parser.add_argument("--placa-altura", type=float, metavar="M",
                        help="Altura do CENTRO do ArUco acima do chao, em m: posicao da placa sem depender do tamanho")
    parser.add_argument("--placa-lado", type=float, metavar="M",
                        help="Lado do quadrado preto do ArUco impresso, em m (padrao 0.10): da a distancia da placa")
    parser.add_argument("--pare-segundos", type=float, metavar="S",
                        help="Na placa de PARE, segue sozinho depois de S segundos (padrao 5; 0 = so pelo botao)")
    parser.add_argument("--sem-imu", action="store_true",
                        help="Guinada pelo modelo da direcao, sem o giroscopio da IMU")
    parser.add_argument("--amortecer-guinada", type=float, metavar="K",
                        help="Amortecimento do Stanley pela guinada medida, em s (0 = desligado; teste 0.1-0.3)")
    parser.add_argument("--atraso-direcao", type=float, metavar="S",
                        help="Atraso da direcao previsto pelo Stanley, s (padrao 0.16; 0 desliga o preditor)")
    parser.add_argument("--sem-pipeline", action="store_true",
                        help="Visao sequencial, sem o processo auxiliar (vision_workers = 0)")
    parser.add_argument("--sem-nucleos", action="store_true",
                        help="Nao fixa controle e visao em nucleos separados (o sistema escolhe)")
    parser.add_argument("--sem-mapa", action="store_true",
                        help="Cones: sem o mapa SLAM + tracado SVM de reserva (comparacao)")
    parser.add_argument("--mapa-substitui", action="store_true",
                        help="Cones: tracado do mapa SLAM + SVM no lugar do estimador sempre (experimento)")
    parser.add_argument("--old-start", action="store_true",
                        help="Partida como antes de 17/11, para comparar (ignora --throttle-offset)")
    parser.add_argument("--navigation", choices=("line", "cones"),
                        help="cones: somente cones (padrao); line: somente linha amarela")
    args = parser.parse_args()
    if not 0 < args.speed <= MAX_VEL:
        parser.error(f"--speed deve estar entre 0 e {MAX_VEL} m/s")
    settings = {"velocity_reference": args.speed}
    if args.navigation is not None:
        settings["navigation_mode"] = args.navigation
    if args.vision_width is not None:
        settings["vision_width"] = args.vision_width
    if args.steering_filter is not None:
        settings["steering_filter_size"] = args.steering_filter
    if args.stream is not None:
        settings["stream_port"] = args.stream
    if args.loss_frames is not None:
        settings["line_loss_frames"] = args.loss_frames
        settings["cone_loss_frames"] = args.loss_frames
    if args.no_cone_debug:
        settings["cone_debug"] = False
    if args.record_frames is not None:
        if args.record_frames < 0:
            parser.error("--record-frames deve ser >= 0")
        settings["record_frames_hz"] = args.record_frames
    if args.encoder_protocol:
        settings["encoder_protocol"] = args.encoder_protocol
    if args.calibration:
        settings["camera_calibration_file"] = args.calibration
    if args.classifier:
        settings["cone_classifier"] = None if args.classifier.lower() == "none" else args.classifier
    if args.no_estimator:
        settings["cone_estimator"] = False
    if args.stanley:
        settings["steering_law"] = "stanley"
    if args.pure_pursuit:
        settings["steering_law"] = "pure_pursuit"
    for value, key, flag in ((args.esterco_esquerda, "steering_left_scale", "--esterco-esquerda"),
                             (args.esterco_direita, "steering_right_scale", "--esterco-direita")):
        if value is not None:
            if not 50 <= value <= 150:
                parser.error(f"{flag} deve estar entre 50 e 150 (%)")
            settings[key] = value/100.0
    if args.abrir_curva is not None:
        if not 0 <= args.abrir_curva <= .5:
            parser.error("--abrir-curva deve estar entre 0 e 0.5 m")
        settings["racing_line_m"] = args.abrir_curva
    if args.stanley_gain is not None:
        if not 0 < args.stanley_gain <= 10:
            parser.error("--stanley-gain deve estar entre 0 e 10")
        settings["stanley_gain"] = args.stanley_gain
    if args.fita_amarela:
        settings["cone_line_fusion"] = True
    if args.line_classic:
        settings["line_birdseye"] = False
        settings["line_estimator"] = False
    if args.no_trim_learning:
        settings["steering_trim_learning"] = False
    if args.strict_ultrasonic:
        settings["ultrasonic_curve_filter"] = False
    if args.min_radius is not None:
        if args.min_radius < 0:
            parser.error("--min-radius deve ser >= 0")
        settings["min_curve_radius_m"] = args.min_radius
    if args.throttle_offset is not None:
        if not 0. <= args.throttle_offset <= .6:
            parser.error("--throttle-offset deve estar entre 0 e 0.6")
        settings["throttle_ff_offset"] = args.throttle_offset
        settings["throttle_learning"] = False  # valor fixo para comparar
    if args.no_refine:
        settings["cone_refine_bases"] = False
    if args.sem_teto_ultrassom:
        settings["ultrasonic_speed_cap"] = False
    if args.largura_pista is not None:
        if not .5 <= args.largura_pista <= 3.:
            parser.error("--largura-pista deve estar entre 0.5 e 3 m")
        settings["cone_settings"] = {**parameters["cone_settings"], "track_width_cm": round(100*args.largura_pista)}
    if args.apice is not None:
        if not 0 <= args.apice <= .3:
            parser.error("--apice deve estar entre 0 e 0.3 m")
        settings["racing_apex_m"] = args.apice
    if args.tempo is not None:
        if args.tempo <= 0:
            parser.error("--tempo deve ser positivo (s)")
        settings["ts"] = float(args.tempo)
    if args.memoria_tracado is not None:
        if not 0 <= args.memoria_tracado <= 1:
            parser.error("--memoria-tracado deve estar entre 0 e 1 s")
        settings["path_memory_s"] = args.memoria_tracado
    if args.sem_aruco:
        settings["aruco"] = False
    if args.pare_segundos is not None and settings.get("aruco", True) is not False:
        if args.pare_segundos < 0:
            parser.error("--pare-segundos deve ser >= 0")
        settings["aruco"] = {**settings.get("aruco", parameters["aruco"]),
                             "hold_s": args.pare_segundos if args.pare_segundos > 0 else None}
    if args.placa_altura is not None and settings.get("aruco", True) is not False:
        if not 0 <= args.placa_altura <= .2:
            parser.error("--placa-altura deve estar entre 0 e 0.2 m (abaixo da camera)")
        settings["aruco"] = {**settings.get("aruco", parameters["aruco"]), "marker_height_m": args.placa_altura}
    if args.placa_lado is not None and settings.get("aruco", True) is not False:
        if not .02 <= args.placa_lado <= .5:
            parser.error("--placa-lado deve estar entre 0.02 e 0.5 m")
        settings["aruco"] = {**settings.get("aruco", parameters["aruco"]), "marker_m": args.placa_lado}
    if args.sem_imu:
        settings["imu"] = False
    if args.amortecer_guinada is not None:
        if not 0 <= args.amortecer_guinada <= 1:
            parser.error("--amortecer-guinada deve estar entre 0 e 1 s")
        settings["stanley_yaw_damping"] = args.amortecer_guinada
    if args.atraso_direcao is not None:
        if not 0 <= args.atraso_direcao <= .5:
            parser.error("--atraso-direcao deve estar entre 0 e 0.5 s")
        settings["steer_delay_s"] = args.atraso_direcao
    if args.sem_pipeline:
        settings["vision_workers"] = 0
    if args.sem_nucleos:
        settings["cpu_cores"] = False
    if args.sem_mapa:
        settings["cone_map"] = False
    if args.mapa_substitui:
        settings["cone_map"] = "substituir"
    if args.old_start:
        settings["old_start"] = True
    if args.no_preview:
        settings["speed_preview_m"] = 0.0
        settings["speed_rise_m_s2"] = 0.0

    def interrupt_mission(_signum, _frame):
        # A GUI remota encerra com SIGTERM. Usa o mesmo finally do Ctrl+C.
        raise KeyboardInterrupt

    def resume_mission(_signum, _frame):
        RESUME_REQUEST.set()  # o laco de controle libera a parada no proximo passo

    previous_handler = signal.signal(signal.SIGTERM, interrupt_mission)
    previous_resume = signal.signal(signal.SIGUSR1, resume_mission) if hasattr(signal, "SIGUSR1") else None
    try:
        try:
            run_mission(settings, display=args.display)
        except (FileNotFoundError, ValueError) as error:
            parser.exit(
                2,
                f"{error}\nCalibre com veiculo_real/tools/calibrar_camera_lateral.py antes de rodar.\n",
            )
    finally:
        if previous_resume is not None:
            signal.signal(signal.SIGUSR1, previous_resume)
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
