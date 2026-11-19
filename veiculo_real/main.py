# -*- coding: utf-8 -*-
"""Stanley no veiculo real: linha amarela ou centro de cones pela camera."""

import argparse
import json
import math
from pathlib import Path
import queue
import signal
import socket
import threading
import time

import numpy as np
import cv2

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
TELEMETRY_HZ = 10.0
TELEMETRY_PERIOD = 1.0 / TELEMETRY_HZ
OBSTACLE_DISTANCE = 0.20  # folga desejada ate o obstaculo, a partir da FRENTE do pneu
# O ultrassom fica na linha do eixo dianteiro (embaixo da camera) e a frente do
# pneu 7 cm adiante (medido em 19/11): a leitura do sensor tem 7 cm a mais que a
# folga real da frente do carro.
FRONT_OVERHANG_M = 0.07
STOP_REACTION_TIME = 0.30
STOP_DECELERATION = 1.75  # m/s^2: ajuste experimental ja usado no veiculo
# Abaixo disto a leitura do ultrassom e falha do sensor (ensaio de 10/11: 0,00 m
# com nada a frente); um obstaculo real passa antes pela faixa de parada.
ULTRASONIC_MIN_VALID_M = 0.03

parameters = {
    "ts": 30.0,
    "navigation_mode": "cones",  # somente cones; linha requer selecao explicita
    # Largura/espacamento da pista: so PALPITE inicial (a pista muda a cada dia).
    # O estimador do corredor mede a largura com as duas bordas e passa a usar a
    # aprendida; a checagem da calibracao so acusa erro grosseiro (0,4-2,5x).
    # Valores dos logs 20261023_0803*/0804*: corredor 1,7-1,9 m, cones a ~0,5 m.
    "cone_settings": {"roi_top": 0, "single_border_recovery": True,
                      "track_width_cm": 180, "cone_spacing_cm": 50},
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
    "racing_line_m": 0.10,
    "racing_inside": 1.0,
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
    # 60 graus/s levava 0,33 s (0,5 m a 1,5 m/s) para chegar ao batente:
    # quase todo o trecho visivel antes da curva.
    "stanley_max_rate_deg_s": 120.0,
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
    "latency_speed_budget_m": 0.3,
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
    config["loss_speed_ramp"] = True
    config["max_lateral_jump_m"] = config.get("cone_max_lateral_jump_m", 0.12)
    config["max_heading_jump_rad"] = config.get("cone_max_heading_jump_rad", 0.35)
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
    poses = None
    if mode == "cones" or line_estimator_settings(config) is not None:
        # A visao leva o mapa de cones (ou a linha) pela odometria registrada pelo controle.
        poses = SharedPoseHistory() if use_process else PoseHistory()
        # O seguidor leva a foto ao instante do comando pela velocidade e guinada
        # atuais (pista virtual, atraso de 0,3-0,4 s: melhor que pelo historico).
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
                                   refine_bases=bool(config.get("cone_refine_bases", True)))
        else:
            detector_kwargs = dict(path_settings=config.get("path_settings", {}),
                                   anchor=config.get("path_anchor", 0.5),
                                   processing_width=config.get("vision_width", 0),
                                   birdseye=bool(config.get("line_birdseye", False)),
                                   estimator=line_estimator_settings(config))
        vision = VisionProcess(vision_data, mode, config["camera_calibration_file"], detector_kwargs,
                               camera_settings=config.get("camera_settings") or {}, poses=poses)
        try:
            # Abre a camera antes dos motores: falha cedo sem mover o carro.
            vision.start()
        except BaseException:
            if stream is not None:
                stream.close()
            raise
        print(f"Visao: processo separado (pid {vision.process.pid})", flush=True)
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
                "old_start": bool(config.get("old_start", False)),
                "throttle_ff_offset": float(config.get("throttle_ff_offset", 0.32)),
                "throttle_learning": bool(config.get("throttle_learning", False)),
                "velocity_reference": float(config.get("velocity_reference", 0.0)),
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
        while car.t < config["ts"]:
            if not car.step():
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
            echo_ignored = False
            close = (valid and np.isfinite(dist) and ULTRASONIC_MIN_VALID_M <= dist <= stop_distance)
            if not close:
                echo_since = None
            if not obstacle_detected and close:
                reason = ""
                if mode == "cones" and config.get("ultrasonic_curve_filter", True):
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
                    obstacle_detected = True
                    print(
                        f"Obstaculo a {dist:.2f} m: parada travada ate o fim do ensaio"
                        + (f" ({reason})." if reason else "."),
                        flush=True,
                    )
            manual = (
                getattr(car, "sel_direcao", 1) == 0
                and getattr(car, "sel_tracao", 1) == 0
            )
            if (
                manual
                and encoder_valid
                and not car.emergencia
                and not obstacle_detected
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
                    car, observation, clock(), obstacle_detected=obstacle_detected
                )
                control_status = follower.status
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
                    obstacle=bool(obstacle_detected), t=float(car.t), v=float(car.v),
                    vref=float(car.vref), steer_cmd_rad=float(car.st)))
            car.log(
                obstacle_ignored=int(echo_ignored),
                obstacle_stop=int(obstacle_detected),
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
            if obstacle_detected and now >= next_stop_beep:
                if car.bz.beep([0.15, 0.15], silence=0.15):
                    next_stop_beep = now + 1.0
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
            stop_event.set()
            if stream is not None:
                stream.close()
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
    parser.add_argument("--old-start", action="store_true",
                        help="Partida como antes de 17/11, para comparar (ignora --throttle-offset)")
    parser.add_argument("--navigation", choices=("line", "cones"),
                        help="cones: somente cones (padrao); line: somente linha amarela")
    args = parser.parse_args()
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
    if args.old_start:
        settings["old_start"] = True
    if args.no_preview:
        settings["speed_preview_m"] = 0.0
        settings["speed_rise_m_s2"] = 0.0

    def interrupt_mission(_signum, _frame):
        # A GUI remota encerra com SIGTERM. Usa o mesmo finally do Ctrl+C.
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt_mission)
    try:
        try:
            run_mission(settings, display=args.display)
        except (FileNotFoundError, ValueError) as error:
            parser.exit(
                2,
                f"{error}\nCalibre com veiculo_real/tools/calibrar_camera_lateral.py antes de rodar.\n",
            )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
