"""Compara ganhos do Stanley no mesmo trecho, com avaliacao independente.

Abra simulador_cones.ttt e pare a simulacao antes de executar na raiz:
venv/bin/python simulador/ensaios/sintonizar_stanley.py --ganhos 1.5 2.5 3.5 5 7
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avaliar_stanley import evaluate
from main import parameters, run_mission
from seguir_linha import VELOCITY_REFERENCE, YellowLineMission, place_car


def summarize(logfile, gain, reference, settle_time=5.0):
    d = np.atleast_1d(np.genfromtxt(Path(logfile) / 'car.csv', delimiter=',', names=True))
    evaluated = d['t'] >= settle_time
    valid = evaluated & (d['line_valid'] == 1)
    curved = valid & (d['line_curvature_inv_m'] > .12)

    def rms(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(values**2))) if len(values) else None

    e = d['cross_track_m'][curved]
    delta = d['stanley_delta_rad'][valid]
    indices = np.flatnonzero(valid)
    adjacent = np.diff(indices) == 1
    dt = np.diff(d['t'][valid])[adjacent]
    rates = np.rad2deg(np.diff(delta)[adjacent]) / dt
    lost = int(np.count_nonzero(evaluated & (d['line_valid'] == 0)))
    stops = int(np.count_nonzero(evaluated & (d['line_stop'] == 1)))
    return {
        'gain': gain, 'logfile': str(logfile),
        'duration_s': float(d['t'][-1]),
        'distance_m': float(np.hypot(np.diff(d['x']), np.diff(d['y'])).sum()),
        'curve_rms_m': rms(e),
        'curve_p95_m': float(np.percentile(abs(e), 95)) if len(e) else None,
        'curve_peak_m': float(np.max(abs(e))) if len(e) else None,
        'reference_rms_m': reference['cross_track_rms_m'],
        'reference_p95_m': reference['cross_track_p95_m'],
        'reference_peak_m': reference['cross_track_peak_m'],
        'reference_pixel_size_m': reference['reference_pixel_size_m'],
        'steering_rate_rms_deg_s': rms(rates),
        'saturated_fraction': float(np.mean(abs(delta) > np.deg2rad(19.8))) if len(delta) else None,
        'lost_frames': lost, 'stop_samples': stops,
        'mean_speed_m_s': float(np.mean(d['v'][evaluated])) if np.any(evaluated) else None,
        'feasible': lost == 0 and stops == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ganhos', type=float, nargs='+', default=[1.5, 2.5, 3.5, 5.0, 7.0])
    parser.add_argument('--vref', type=float, default=VELOCITY_REFERENCE)
    parser.add_argument('--ts', type=float, default=60)
    parser.add_argument('--port', type=int, default=23000)
    parser.add_argument('--initial-pose', type=float, nargs=3, default=[11.025, -9.124, 0])
    args = parser.parse_args()
    if not np.isfinite(args.ts) or args.ts <= 5:
        parser.error('--ts deve ser maior que 5 s e finito.')
    # Valida todos os ganhos antes de reposicionar ou abrir um ensaio.
    for gain in args.ganhos:
        YellowLineMission(args.vref, gain)
    sim = RemoteAPIClient(port=args.port).require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        parser.error('Pare a simulacao antes de iniciar a comparacao.')
    pista = sim.getObject('/Pista')
    shape = sim.getShapeViz(pista, 0)
    matrix = sim.getObjectMatrix(pista, -1)
    folder = Path('logs') / datetime.now().strftime('stanley_tuning_%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for gain in args.ganhos:
        robot, original = place_car(sim, args.initial_pose)
        mission = YellowLineMission(args.vref, gain)
        try:
            run_mission(dict(parameters, ts=args.ts, port=args.port, real_time=False,
                             beep=False, logfile=str(folder / f'gain_{gain:g}')),
                        display=False, control=mission)
        finally:
            try:
                mission.restore_camera()
            finally:
                sim.setObjectMatrix(robot, original, -1)
        mission.save_diagnostics()
        if mission.logfile is None:
            print('Ensaio interrompido antes da aquisicao da linha.', flush=True)
            break
        reference = evaluate(mission.logfile, shape, matrix)
        result = summarize(mission.logfile, gain, reference)
        result['completed'] = result['duration_s'] >= args.ts - 1e-6
        result['feasible'] = result['feasible'] and result['completed']
        results.append(result)
        (folder / 'comparison.json').write_text(
            json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps(results[-1], indent=2), flush=True)
        if not result['completed']:
            print('Comparacao interrompida; os ganhos restantes nao serao executados.', flush=True)
            break
    print(f'Comparacao salva em {folder / "comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
