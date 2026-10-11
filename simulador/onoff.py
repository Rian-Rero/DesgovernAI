# -*- coding: utf-8 -*-
# Disciplina: Fundamentos de Veiculos Autonomos - 2026/2
# Etapa 2: controlador on-off de velocidade longitudinal com histerese.
#
# Usa somente a velocidade medida (car.v) e atua na aceleracao (car.set_u,
# o car.setU() do enunciado). Lei de controle, com e = vref - v:
#   e >  +H  -> u = +U_MAX  (liga: acelera)
#   e <  -H  -> u = -U_MAX  (desliga: freia)
#   |e| <= H -> mantem o ultimo u (histerese)
#
# Uso (cena aberta no CoppeliaSim, simulacao parada, na raiz do repositorio):
#   venv/bin/python simulador/onoff.py --vref 1.0
#   venv/bin/python simulador/onoff.py --vref 1.5 --h 0.05 --ts 20
# Gera logs/<data>/car.csv com as colunas padrao + onoff (1 ligado, 0 desligado).
########################################
import argparse

from main import parameters, run_mission

VELOCITY_REFERENCE = 1.0  # m/s
HYSTERESIS = 0.05  # m/s: meia largura da faixa morta em torno de vref
U_MAX = 1.0  # m/s^2: comando normalizado maximo do car.set_u (ACCELMAX)


class OnOffController:
    def __init__(self, vref=VELOCITY_REFERENCE, hysteresis=HYSTERESIS, u_max=U_MAX):
        if vref <= hysteresis:
            # com vref <= H o estado desligado (-U_MAX) poria o carro em re
            raise ValueError("A referencia deve ser maior que a histerese.")
        self.vref = vref
        self.hysteresis = hysteresis
        self.u_max = u_max
        self.on = True  # parte acelerando

    def __call__(self, car):
        car.set_steer(0.0)
        # registra a referencia no car.csv (set_u nao altera car.vref)
        car.vref = self.vref

        error = self.vref - car.v
        if error > self.hysteresis:
            self.on = True
        elif error < -self.hysteresis:
            self.on = False
        # dentro da faixa: mantem o estado anterior

        car.set_u(self.u_max if self.on else -self.u_max)
        car.log(onoff=int(self.on))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Controle on-off de velocidade no CoppeliaSim.")
    p.add_argument("--vref", type=float, default=VELOCITY_REFERENCE, help="referencia [m/s]")
    p.add_argument("--h", type=float, default=HYSTERESIS, help="histerese [m/s]")
    p.add_argument("--umax", type=float, default=U_MAX, help="|u| nos dois estados")
    p.add_argument("--ts", type=float, default=parameters["ts"], help="duracao [s]")
    p.add_argument("--sem-grafico", action="store_true", help="nao abre a telemetria")
    a = p.parse_args()

    print(f"On-off: vref = {a.vref:.2f} m/s, H = {a.h:.3f} m/s, u = +/-{a.umax:.2f}", flush=True)
    run_mission(
        dict(parameters, ts=a.ts),
        display=not a.sem_grafico,
        control=OnOffController(a.vref, a.h, a.umax),
    )
