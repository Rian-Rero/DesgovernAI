# -*- coding: utf-8 -*-
"""Figuras e tabela de metricas da Parte 2 a partir dos logs da campanha.

Uso (na raiz):
  venv/bin/python simulador/ensaios/figuras_lateral.py \
      --dados simulador/ensaios/dados/lateral_20261116 --saida relatorio/imgs
Gera PNGs e metricas.csv (uma linha por ensaio, a partir dos resumo.json).
Nao precisa do CoppeliaSim: a textura da pista vem do .jpg da cena.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEXTURA = ROOT / 'simulador/coppeliasim/autodromo_linha_amarela.jpg'
# Mapeamento UV -> mundo lido de /Pista (getShapeViz): x = 72u - 36.125, y = 27v - 13.05.
EXTENT = (-36.125, 35.875, -13.05, 13.95)
CORES = {'onoff': 'tab:red', 'stanley': 'tab:blue'}
NOMES = {'onoff': 'On-off', 'stanley': 'Stanley'}
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def carregar(dados):
    ensaios = {}
    for resumo in sorted(Path(dados).glob('*/*/resumo.json')):
        nome = resumo.parent.parent.name
        log = np.genfromtxt(resumo.parent / 'car.csv', delimiter=',', names=True)
        ensaios[nome] = dict(resumo=json.loads(resumo.read_text(encoding='utf-8')), log=log)
    return ensaios


def caminho_geometrico(ref, v=1.0):
    """Reconstroi a referencia como em controle_lateral.py (para os graficos)."""
    import sys
    sys.path.insert(0, str(ROOT / 'simulador'))
    from controle_lateral import GEOMETRIC_POSES
    from fva_car.trajetorias import circle, figure_eight, straight
    x, y, yaw = GEOMETRIC_POSES[ref]
    yaw = math.radians(yaw)
    # Eixo dianteiro 0,157 m a frente da origem de /Car (medido nos logs).
    front = np.array([x, y]) + .157 * np.array([math.cos(yaw), math.sin(yaw)])
    return {'reta': lambda: straight(front, yaw), 'circulo': lambda: circle(front, yaw),
            'oito': lambda: figure_eight(front, yaw)}[ref]().points


def fig_geometrico(ens, saida, v=1.0):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), gridspec_kw=dict(width_ratios=(1.6, 1, 1)))
    titulos = {'reta': 'Reta (offset inicial 0,5 m)', 'circulo': 'Círculo (R = 3 m)',
               'oito': 'Oito (R = 3 m)'}
    for ax, ref in zip(axes, ('reta', 'circulo', 'oito')):
        p = caminho_geometrico(ref)
        ax.plot(p[:, 0], p[:, 1], 'k--', lw=1.2, label='Referência')
        for ctl in ('onoff', 'stanley'):
            e = ens.get(f'geo_{ref}_{ctl}_{v}')
            if e:
                ax.plot(e['log']['x_front'], e['log']['y_front'], color=CORES[ctl],
                        lw=1, label=NOMES[ctl])
        ax.set(title=titulos[ref], xlabel='x (m)', ylabel='y (m)')
        ax.set_aspect('equal', adjustable='datalim')
    axes[0].set_ylim(-.6, 1.2)
    axes[0].set_aspect('auto')
    axes[0].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(saida / 'geo_trajetorias.png')
    plt.close(fig)


def fig_oito_tempo(ens, saida, v=1.0):
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.4), sharex=True)
    for ctl in ('onoff', 'stanley'):
        e = ens.get(f'geo_oito_{ctl}_{v}')
        if not e:
            continue
        d = e['log']
        axes[0].plot(d['t'], 100 * d['cross_track_m'], color=CORES[ctl], lw=1, label=NOMES[ctl])
        axes[1].plot(d['t'], np.degrees(d['delta_geo_rad']), color=CORES[ctl], lw=1,
                     drawstyle='steps-post', label=NOMES[ctl])
    axes[0].set(ylabel='Erro lateral $e_y$ (cm)')
    axes[1].set(ylabel=r'Esterçamento $\delta$ (°)', xlabel='Tempo (s)')
    axes[0].legend(ncol=2, fontsize=8)
    fig.align_ylabels()
    fig.savefig(saida / 'geo_oito_tempo.png')
    plt.close(fig)


def fig_sensibilidade(ens, saida):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for d, estilo in ((8, ':'), (15, '-'), (20, '--')):
        nome = 'geo_oito_onoff_1.0' if d == 15 else f'sens_onoff_delta{d}'
        e = ens.get(nome)
        if e:
            axes[0].plot(e['log']['t'], 100 * e['log']['cross_track_m'], estilo, lw=1,
                         label=fr'$\delta_{{on}}$ = {d}°')
    axes[0].set(title='On-off no oito, 1,0 m/s', xlabel='Tempo (s)', ylabel='$e_y$ (cm)')
    axes[0].legend(fontsize=8)
    for k, estilo in ((0.5, ':'), (1.5, '-'), (5, '--')):
        nome = 'geo_reta_stanley_1.0' if k == 1.5 else f'sens_stanley_k{k:g}'
        e = ens.get(nome)
        if e:
            axes[1].plot(e['log']['t'], 100 * e['log']['cross_track_m'], estilo, lw=1,
                         label=f'k = {k:g} 1/s')
    axes[1].set(title='Stanley na reta, 1,0 m/s', xlabel='Tempo (s)', ylabel='$e_y$ (cm)',
                xlim=(0, 12))
    axes[1].legend(fontsize=8)
    fig.savefig(saida / 'geo_sensibilidade.png')
    plt.close(fig)


def fig_pista(ens, saida, v):
    img = plt.imread(TEXTURA)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6))
    for ax, ctl in zip(axes, ('onoff', 'stanley')):
        ax.imshow(img, extent=EXTENT, origin='upper')
        e = ens.get(f'visao_{ctl}_{v}')
        if e:
            d = e['log']
            ok = np.isfinite(d['gt_error_m'])
            sc = ax.scatter(d['x_front'][ok], d['y_front'][ok], c=100 * d['gt_error_m'][ok],
                            s=2, cmap='viridis', vmin=0, vmax=15)
            r = e['resumo']
            estado = 'volta completa' if r['concluido'] else f"falha: {r['falha']}"
            ax.set_title(f"{NOMES[ctl]}, $v_{{ref}}$ = {v:g} m/s ({estado}, "
                         f"{r['tempo_s']:.0f} s)")
        ax.set(xlim=(-34, 34), ylim=(-12, 12.5), ylabel='y (m)')
        ax.grid(False)
    axes[1].set_xlabel('x (m)')
    axes[0].tick_params(labelbottom=False)
    cb = fig.colorbar(sc, ax=axes, shrink=.8, pad=.01)
    cb.set_label('Erro lateral verdadeiro (cm)')
    fig.savefig(saida / f'pista_{v:g}.png'.replace('.', '_', 1).replace('_png', '.png'))
    plt.close(fig)


def fig_velocidades(ens, saida, volta=245.6):
    """(a) erro verdadeiro RMS; (b) fracao da volta percorrida. Vazado = falhou."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ctl, marker in (('onoff', 'o'), ('stanley', 's')):
        pts = []
        for v in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
            e = ens.get(f'visao_{ctl}_{v}')
            if e:
                r = e['resumo']
                pts.append((v, 100 * (r['erro_rms_m'] or np.nan),
                            min(100., 100 * r['distancia_m'] / volta), r['concluido']))
        if not pts:
            continue
        v, rms, frac, ok = map(np.array, zip(*pts))
        for ax, y in zip(axes, (rms, frac)):
            ax.plot(v, y, '-', color=CORES[ctl], lw=1, label=NOMES[ctl])
            ax.plot(v[ok], y[ok], marker, color=CORES[ctl])
            ax.plot(v[~ok], y[~ok], marker, mfc='w', color=CORES[ctl])
    axes[0].set(xlabel='$v_{ref}$ (m/s)', ylabel='RMS do erro verdadeiro (cm)')
    axes[1].set(xlabel='$v_{ref}$ (m/s)', ylabel='Volta percorrida (%)', ylim=(0, 105))
    axes[0].legend(fontsize=8, title='marcador vazado: falhou', title_fontsize=7)
    fig.savefig(saida / 'visao_velocidades.png')
    plt.close(fig)


def fig_visao_tempo(ens, saida, v):
    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    for ctl in ('onoff', 'stanley'):
        e = ens.get(f'visao_{ctl}_{v}')
        if not e:
            continue
        d = e['log']
        axes[0].plot(d['distance_m'], 100 * d['cross_track_m'], color=CORES[ctl], lw=.8,
                     label=f'{NOMES[ctl]} (estimado pela câmera)')
        axes[1].plot(d['distance_m'], np.degrees(d['delta_geo_rad']), color=CORES[ctl], lw=.8,
                     label=NOMES[ctl])
        axes[2].plot(d['distance_m'], d['v'], color=CORES[ctl], lw=.8, label=NOMES[ctl])
        if ctl == 'stanley':
            axes[2].plot(d['distance_m'], d['vref_cmd'], 'k:', lw=.8, label='Referência')
    axes[0].set(ylabel='$e_y$ visual (cm)')
    axes[1].set(ylabel=r'$\delta$ (°)')
    axes[2].set(ylabel='$v$ (m/s)', xlabel='Distância percorrida (m)')
    for ax in axes:
        ax.legend(fontsize=7, ncol=3, loc='upper right')
    fig.align_ylabels()
    fig.savefig(saida / f'visao_tempo_{str(v).replace(".", "_")}.png')
    plt.close(fig)


def tabela(ens, saida):
    campos = ['ensaio', 'referencia', 'controlador', 'vref', 'limitar_curvas', 'concluido',
              'falha', 'tempo_s', 'distancia_m', 'velocidade_media_m_s', 'erro_rms_m',
              'erro_p95_m', 'erro_max_m', 'erro_visual_rms_m', 'taxa_estercamento_rms_deg_s',
              'fracao_saturado', 'comutacoes_por_s', 'inclinacao_max_deg', 'fracao_linha_valida']
    with open(saida / 'metricas.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, campos, extrasaction='ignore')
        w.writeheader()
        for nome, e in ens.items():
            w.writerow(dict(e['resumo'], ensaio=nome))


REAL = ROOT / 'veiculo_real/ensaios/stanley_20261023'


def metricas_reais(pasta=REAL):
    """Uma linha por ensaio do carrinho real (erros estimados pela camera)."""
    linhas = []
    for f in sorted(Path(pasta).glob('*/car.csv')):
        d = np.genfromtxt(f, delimiter=',', names=True)
        t = d['t'] - d['t'][0]
        mov = d['vref'] > .05
        ok = mov & (d['line_valid'] == 1)
        ey, ep = d['cross_track_m'][ok], d['heading_error_rad'][ok]
        status = json.loads((f.parent / 'vision_status.json').read_text(encoding='utf-8'))
        linhas.append(dict(
            ensaio=f.parent.name, duracao_auto_s=float(t[mov][-1] - t[mov][0]),
            vref_mediana=float(np.median(d['vref'][mov])), v_media=float(np.mean(d['v'][mov])),
            linha_valida=float(np.mean(d['line_valid'][mov] == 1)),
            ey_rms_cm=float(100 * np.sqrt(np.mean(ey ** 2))), ey_max_cm=float(100 * np.max(np.abs(ey))),
            ey_media_cm=float(100 * np.mean(ey)), epsi_media_deg=float(np.degrees(np.mean(ep))),
            epsi_rms_deg=float(np.degrees(np.sqrt(np.mean(ep ** 2)))),
            delta_media_deg=float(np.degrees(np.mean(d['steer_cmd_rad'][ok]))),
            idade_visao_s=float(np.nanmedian(d['vision_age_s'][ok])),
            parada_obstaculo=int(np.nanmax(d['obstacle_stop']) > 0),
            encoder_valido=float(np.mean(d['encoder_valid'])),
            status_final=status.get('control_status')))
    return linhas


def fig_real(saida, pasta=REAL, ensaio='20261023_045917'):
    d = np.genfromtxt(Path(pasta) / ensaio / 'car.csv', delimiter=',', names=True)
    t = d['t'] - d['t'][0]
    img = plt.imread(Path(pasta) / ensaio / 'vision_detection.jpg')
    fig = plt.figure(figsize=(10, 3.6))
    g = fig.add_gridspec(2, 2, width_ratios=(1, 1.6))
    ax = fig.add_subplot(g[:, 0])
    ax.imshow(img)
    ax.set(xticks=[], yticks=[], title='(a) Último quadro: faixa detectada')
    ax.grid(False)
    ax1 = fig.add_subplot(g[0, 1])
    ax1.plot(t, 100 * d['cross_track_m'], lw=.8, label='$e_y$ (cm)')
    ax1.plot(t, np.degrees(d['heading_error_rad']), lw=.8, label=r'$e_\psi$ (°)')
    ax1.plot(t, np.degrees(d['steer_cmd_rad']), lw=.8, label=r'$\delta$ (°)')
    ax1.set(title='(b) Erros estimados pela câmera e esterçamento')
    ax1.legend(fontsize=7, ncol=3)
    ax2 = fig.add_subplot(g[1, 1], sharex=ax1)
    ax2.plot(t, d['v'], lw=.8, label='$v$ medida')
    ax2.plot(t, d['vref'], 'k:', lw=.8, label='$v_{ref}$')
    ax2.set(xlabel='Tempo (s)', ylabel='m/s')
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(saida / 'real_stanley.png')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados', type=Path, default=Path('simulador/ensaios/dados/lateral_20261116'))
    p.add_argument('--saida', type=Path, default=Path('simulador/ensaios/dados/lateral_20261116/figuras'))
    a = p.parse_args()
    a.saida.mkdir(parents=True, exist_ok=True)
    ens = carregar(a.dados)
    tabela(ens, a.saida)
    fig_geometrico(ens, a.saida)
    fig_oito_tempo(ens, a.saida)
    fig_sensibilidade(ens, a.saida)
    fig_velocidades(ens, a.saida)
    for v in (1.0, 1.5, 2.0, 2.5):
        if f'visao_stanley_{v}' in ens or f'visao_onoff_{v}' in ens:
            fig_pista(ens, a.saida, v)
            fig_visao_tempo(ens, a.saida, v)
    if REAL.exists():
        reais = metricas_reais()
        with open(a.saida / 'metricas_real.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, list(reais[0]))
            w.writeheader()
            w.writerows(reais)
        fig_real(a.saida)
    print(f'{len(ens)} ensaios -> {a.saida}')


if __name__ == '__main__':
    main()
