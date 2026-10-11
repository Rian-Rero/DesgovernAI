# -*- coding: utf-8 -*-
# Disciplina: Fundamentos de Veiculos Autonomos - 2026/2
# Etapa 6: alerta de colisao frontal (FCW) e frenagem autonoma de emergencia (AEB)
# com o ultrassom do simulador (car.get_distance, o car.getDistance() do enunciado).
#
# AEB: dispara na primeira leitura com d <= d_stop(v) e fica travada ate o fim:
#   d_stop = D_MARGEM + |v|*T_REACAO + v^2/(2*A_STOP)
#   vref vai a 0 e o carro freia com u = -U_FREIO ate |v| < V_PARADO; depois, banguela.
# FCW: alerta (beep + mensagem) enquanto o tempo ate colisao d/|v| <= TTC_FCW.
#   Com A_STOP = 1,0 m/s^2, a AEB dispara com d/|v| entre 0,95 e 1,4 s para
#   0,5 <= v <= 2,0 m/s, entao o FCW (2,0 s) vem antes da frenagem nessa faixa.
#
# O obstaculo (caixa estatica) e criado pelo script, pois a cena nao tem um a frente:
#   fixo:       existe desde t = 0, a --dist m a frente do ultrassom;
#   inesperado: surge em t = --t-obst, a --dist m a frente do ultrassom naquele instante.
# O CoppeliaSim remove o obstaculo ao parar a simulacao.
#
# Uso (cena simulador_cones.ttt aberta, simulacao parada, na raiz do repositorio):
#   venv/bin/python simulador/fcw_aeb.py --vref 1.0 --obstaculo fixo
#   venv/bin/python simulador/fcw_aeb.py --vref 1.5 --obstaculo inesperado
# Gera logs/<data>/car.csv com as colunas padrao + dist, d_stop, ttc, fcw, aeb.
# dist, d_stop, ttc, fcw e aeb sao da propria amostra; u e vref (convencao do
# car.csv) sao o comando aplicado ate aquela amostra, decidido na anterior.
########################################
import argparse
import math

import numpy as np
from main import parameters, run_mission

VELOCITY_REFERENCE = 1.0  # m/s

# AEB: mesma forma da Eq. de d_stop do relatorio
D_MARGEM = 0.20  # m: folga desejada entre o ultrassom e o obstaculo
T_REACAO = 0.30  # s: atraso de leitura e atuacao
A_STOP = 1.0  # m/s^2: medido 1,10 m/s^2 com u = -1 no simulador (1,0 e 1,5 m/s); usa ~9% a menos
U_FREIO = 1.0  # comando de frenagem (car.set_u(-U_FREIO))
V_PARADO = 0.05  # m/s: abaixo disso a frenagem e liberada (banguela)

# FCW
TTC_FCW = 2.0  # s
V_MIN_TTC = 0.05  # m/s: abaixo disso o tempo ate colisao nao e calculado

# ultrassom do simulador: get_distance satura em 4,0 m quando nao ha deteccao
DIST_MAX = 4.0  # m

# obstaculo criado pelo script (profundidade, largura, altura) [m]
OBSTACLE_SIZE = (0.10, 0.60, 0.40)
OBSTACLE_DIST = {"fixo": 8.0, "inesperado": 3.0}  # m a frente do ultrassom
OBSTACLE_TIME = {"fixo": 0.0, "inesperado": 5.0}  # s


def stopping_distance(velocity):
    """Distancia de acionamento da AEB [m] para a velocidade [m/s]."""
    speed = abs(velocity)
    return D_MARGEM + speed * T_REACAO + speed ** 2 / (2.0 * A_STOP)


class FcwAeb:
    def __init__(self, vref, obstacle, distance, spawn_time):
        self.vref = vref
        self.obstacle = obstacle
        self.distance = distance
        self.spawn_time = spawn_time
        self.obstacle_handle = None
        self.fcw = False
        self.aeb = False
        self.braking = False

    def spawn_obstacle(self, car):
        """Cria uma caixa estatica a self.distance m a frente do ultrassom."""
        sim = car.sim
        m = sim.getObjectMatrix(car.robot, -1)
        # o eixo Z local do Car aponta para a frente (ver Car.get_vel)
        forward = np.array([m[2], m[6], 0.0])
        forward /= np.linalg.norm(forward)
        sensor = np.array(sim.getObjectPosition(car.ultra, -1))
        depth, width, height = OBSTACLE_SIZE
        center = sensor + forward * (self.distance + depth / 2)
        center[2] = height / 2
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, list(OBSTACLE_SIZE), 0)
        sim.setObjectAlias(h, "obstaculo_aeb")
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        sim.setObjectPosition(h, -1, center.tolist())
        sim.setObjectOrientation(h, -1, [0.0, 0.0, math.atan2(forward[1], forward[0])])
        self.obstacle_handle = h
        print(f"t = {car.t:5.2f} s | obstaculo {self.obstacle} a {self.distance:.2f} m", flush=True)

    def __call__(self, car):
        car.set_steer(0.0)
        if self.obstacle_handle is None and car.t >= self.spawn_time:
            self.spawn_obstacle(car)

        dist, valid = car.get_distance(max_dist=DIST_MAX)
        speed = abs(car.v)
        d_stop = stopping_distance(car.v)
        seen = valid and dist < DIST_MAX
        ttc = dist / speed if seen and speed > V_MIN_TTC else math.nan

        # AEB: trava na primeira leitura dentro de d_stop
        if not self.aeb and seen and dist <= d_stop:
            self.aeb = self.braking = True
            print(f"t = {car.t:5.2f} s | AEB: d = {dist:.2f} m <= d_stop = {d_stop:.2f} m "
                  f"(v = {speed:.2f} m/s)", flush=True)

        # FCW: alerta por tempo ate colisao (mantido durante a AEB)
        fcw = self.aeb or (not math.isnan(ttc) and ttc <= TTC_FCW)
        if fcw and not self.fcw:
            print(f"t = {car.t:5.2f} s | FCW: d = {dist:.2f} m, TTC = {ttc:.2f} s", flush=True)
        self.fcw = fcw
        if fcw:
            car.beep([0.1, 0.1], silence=0.1)

        if self.aeb:
            car.speed_controller.reset()
            if self.braking and speed > V_PARADO:
                car.set_u(-U_FREIO)
            else:
                # parado: sem torque, para nao engatar re
                self.braking = False
                car.set_coast()
            car.vref = 0.0
        else:
            car.set_vel(self.vref)

        car.log(d_stop=d_stop, ttc=ttc, fcw=int(self.fcw), aeb=int(self.aeb))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FCW e AEB com o ultrassom no CoppeliaSim.")
    p.add_argument("--vref", type=float, default=VELOCITY_REFERENCE, help="velocidade de cruzeiro [m/s]")
    p.add_argument("--obstaculo", choices=["fixo", "inesperado"], default="fixo")
    p.add_argument("--dist", type=float, help="distancia do obstaculo ao ultrassom quando criado [m]")
    p.add_argument("--t-obst", type=float, help="instante em que o obstaculo e criado [s]")
    p.add_argument("--ts", type=float, default=parameters["ts"], help="duracao [s]")
    p.add_argument("--sem-grafico", action="store_true", help="nao abre a telemetria")
    a = p.parse_args()

    dist = OBSTACLE_DIST[a.obstaculo] if a.dist is None else a.dist
    t_obst = OBSTACLE_TIME[a.obstaculo] if a.t_obst is None else a.t_obst
    print(f"FCW/AEB: vref = {a.vref:.2f} m/s, obstaculo {a.obstaculo} a {dist:.2f} m "
          f"em t = {t_obst:.1f} s | d_stop(vref) = {stopping_distance(a.vref):.2f} m", flush=True)
    run_mission(
        dict(parameters, ts=a.ts),
        display=not a.sem_grafico,
        control=FcwAeb(a.vref, a.obstaculo, dist, t_obst),
    )
