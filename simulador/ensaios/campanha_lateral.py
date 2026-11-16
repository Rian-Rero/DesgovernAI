# -*- coding: utf-8 -*-
"""Campanha de ensaios da Parte 2 (controle lateral) no CoppeliaSim.

Cada ensaio chama simulador/controle_lateral.py com os argumentos da tabela
ENSAIOS e grava em <saida>/<id>/<data>/{car.csv,resumo.json}.

Uso (na raiz; CoppeliaSim aberto com simulador_cones.ttt, simulacao parada):
  venv/bin/python simulador/ensaios/campanha_lateral.py --grupo geometrico
  venv/bin/python simulador/ensaios/campanha_lateral.py --grupo visao --port 23010
  venv/bin/python simulador/ensaios/campanha_lateral.py --somente visao_onoff_1.0
Ensaios ja concluidos (com resumo.json) sao pulados; use --refazer para repetir.
A camera so renderiza com a janela do CoppeliaSim oculta (-h), nao em -H:
  QT_QPA_PLATFORM=xcb /opt/CoppeliaSim/coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23010 cena.ttt
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controle_lateral import parse, run  # noqa: E402

SAIDA = Path('simulador/ensaios/dados/lateral_20261116')


def _ensaios():
    e = {}
    # 1) Referencias geometricas (pose verdadeira), dois controladores, duas velocidades.
    for ref in ('reta', 'circulo', 'oito'):
        for ctl in ('onoff', 'stanley'):
            for v in (1.0, 2.0):
                e[f'geo_{ref}_{ctl}_{v}'] = ('geometrico', [
                    '--referencia', ref, '--controlador', ctl, '--vref', str(v), '--ts', '80'])
    # 2) Sensibilidade: amplitude do on-off no oito e ganho do Stanley na reta.
    for d in (8., 20.):
        e[f'sens_onoff_delta{d:g}'] = ('sensibilidade', [
            '--referencia', 'oito', '--controlador', 'onoff', '--vref', '1.0',
            '--delta-on', str(d), '--ts', '80'])
    # Amplitude menor que o esterco de regime das curvas mais fechadas da pista.
    e['sens_visao_onoff_delta8'] = ('sensibilidade', [
        '--referencia', 'linha', '--controlador', 'onoff', '--vref', '0.5',
        '--delta-on', '8', '--ts', '700'])
    e['sens_visao_onoff_delta8_0.75'] = ('sensibilidade', [
        '--referencia', 'linha', '--controlador', 'onoff', '--vref', '0.75',
        '--delta-on', '8', '--ts', '700'])
    for k in (0.5, 5.0):
        e[f'sens_stanley_k{k:g}'] = ('sensibilidade', [
            '--referencia', 'reta', '--controlador', 'stanley', '--vref', '1.0',
            '--ganho', str(k), '--ts', '80'])
    # 3) Linha amarela pela camera: uma volta completa em varias velocidades.
    for ctl in ('onoff', 'stanley'):
        for v in (1.0, 1.5, 2.0, 2.5):
            e[f'visao_{ctl}_{v}'] = ('visao', [
                '--referencia', 'linha', '--controlador', ctl, '--vref', str(v), '--ts', '400'])
    # O on-off perde a linha a 1,0 m/s: velocidades menores.
    for v in (0.5, 0.75):
        e[f'visao_onoff_{v}'] = ('visao', [
            '--referencia', 'linha', '--controlador', 'onoff', '--vref', str(v), '--ts', '700'])
    # 4) Velocidade maxima com reducao nas curvas (v <= sqrt(a_lat/|kappa|)).
    for ctl in ('onoff', 'stanley'):
        e[f'visao_{ctl}_2.5_curvas'] = ('curvas', [
            '--referencia', 'linha', '--controlador', ctl, '--vref', '2.5',
            '--limitar-curvas', '1.5', '--ts', '400'])
    return e


ENSAIOS = _ensaios()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grupo', choices=('geometrico', 'sensibilidade', 'visao', 'curvas', 'todos'),
                   default='todos')
    p.add_argument('--somente', nargs='*', help='ids de ensaios especificos')
    p.add_argument('--port', type=int, default=23000)
    p.add_argument('--saida', type=Path, default=SAIDA)
    p.add_argument('--refazer', action='store_true')
    a = p.parse_args()
    for name, (group, argv) in ENSAIOS.items():
        if a.somente and name not in a.somente:
            continue
        if not a.somente and a.grupo != 'todos' and group != a.grupo:
            continue
        folder = a.saida / name
        if not a.refazer and list(folder.glob('*/resumo.json')):
            print(f'[pulado] {name}', flush=True)
            continue
        print(f'[ensaio] {name}: {" ".join(argv)}', flush=True)
        run(parse(argv + ['--saida', str(folder) + '/', '--port', str(a.port)]))


if __name__ == '__main__':
    main()
