"""Relatorio offline: erros visuais estimados, comandos e trajetoria medida."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_report(logfile, settle_time=5.0):
    folder = Path(logfile)
    data = np.atleast_1d(np.genfromtxt(folder / 'car.csv', delimiter=',', names=True))
    evaluated = data[data['t'] >= settle_time]
    valid = evaluated[evaluated['line_valid'] == 1]

    def metric(values, function):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        return float(function(values)) if len(values) else None

    metrics = {
        'error_source': 'camera estimate; not independent ground truth',
        'evaluated_after_s': settle_time,
        'duration_s': float(data['t'][-1]),
        'distance_travelled_m': float(np.sum(np.hypot(np.diff(data['x']), np.diff(data['y'])))),
        'line_valid_fraction': metric(evaluated['line_valid'], np.mean),
        'line_stop_samples': int(np.count_nonzero(evaluated['line_stop'] == 1)),
        'cross_track_rms_m': metric(valid['cross_track_m'], lambda x: np.sqrt(np.mean(x*x))),
        'cross_track_peak_m': metric(valid['cross_track_m'], lambda x: np.max(np.abs(x))),
        'heading_rms_deg': metric(np.rad2deg(valid['heading_error_rad']), lambda x: np.sqrt(np.mean(x*x))),
        'mean_speed_m_s': metric(evaluated['v'], np.mean),
        'speed_tracking_rms_m_s': metric(evaluated['v'] - evaluated['commanded_speed_m_s'],
                                        lambda x: np.sqrt(np.mean(x*x))),
    }
    (folder / 'stanley_metrics.json').write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].plot(data['x'], data['y'])
    axes[0, 0].set(xlabel='x mundial (m)', ylabel='y mundial (m)',
                   title='Trajetoria medida no simulador', aspect='equal')
    axes[0, 1].plot(data['t'], data['cross_track_m'], color='tab:red')
    axes[0, 1].set(xlabel='Tempo (s)', ylabel='Erro lateral visual (m)',
                   title='Estimativa no eixo dianteiro')
    axes[1, 0].plot(data['t'], data['v'], label='Medida')
    axes[1, 0].plot(data['t'], data['commanded_speed_m_s'], '--', label='Referencia')
    axes[1, 0].set(xlabel='Tempo (s)', ylabel='Velocidade (m/s)', title='PI longitudinal')
    axes[1, 0].legend()
    axes[1, 1].plot(data['t'], np.rad2deg(data['stanley_delta_rad']), color='tab:purple')
    axes[1, 1].set(xlabel='Tempo (s)', ylabel='Estercamento (graus, esquerda +)',
                   title='Stanley', ylim=(-22, 22))
    for axis in axes.flat:
        axis.grid(True)
    fig.savefig(folder / 'stanley_report.png', dpi=150)
    plt.close(fig)
    return metrics
