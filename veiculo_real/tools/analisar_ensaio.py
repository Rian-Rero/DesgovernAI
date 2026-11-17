#!/usr/bin/env python3
"""Relatorio de um ensaio do carro (cones), com os criterios do PLANO_CORRECAO.md.

    python3 tools/analisar_ensaio.py logs/<ensaio>
    python3 tools/analisar_ensaio.py experimentos/vermelho/2026*      # varios
    python3 tools/analisar_ensaio.py logs/<ensaio> --linha             # + posicao real pela fita

Le car.csv, cones_debug.jsonl, cones_summary.json e vision_status.json. Com
--linha e quadros gravados (frames/), mede onde o carro estava em relacao a fita
amarela do centro, a unica "verdade" da pista. Grava <ensaio>/relatorio.txt.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOALS = {  # PLANO_CORRECAO.md
    'quadros_por_s': 15., 'idade_ms': 150., 'erro_rms_cm': 5., 'rumo_rms_graus': 6.,
    'partida_s': 1.5,  # ate 80% da velocidade de cruzeiro (README_cones.md, "Partida")
}
RECOVERY_SPEED = .35  # fva_car/cones.py CONE_RECOVERY_SPEED


def _read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stops(car):
    """Paradas pelo ultrassom: (instante, leitura, leituras curtas seguidas antes, leitura depois)."""
    if 'obstacle_stop' not in car or not car['obstacle_stop'].max() == 1:
        return []
    index = int(np.flatnonzero(car['obstacle_stop'].to_numpy() == 1)[0])
    distance = car['distance_m'].to_numpy()
    limit = car['stopping_distance_m'].to_numpy()
    changes = np.flatnonzero(np.diff(distance[:index+1]) != 0) + 1
    readings = distance[np.r_[0, changes]] if len(changes) else distance[:1]
    short = 0
    for value in readings[::-1]:
        if value <= limit[index]:
            short += 1
        else:
            break
    after = distance[index:index+150]
    after_changes = after[np.r_[0, np.flatnonzero(np.diff(after) != 0) + 1]]
    return [dict(t=float(car['t'].iloc[index]), leitura=float(distance[index]), seguidas=int(short),
                 depois=[round(float(v), 2) for v in after_changes[:6]])]


def tracking(records):
    """Erros do corredor com o controle ativo: reta e curvas, trocas de lado, estados."""
    rows = {'e': [], 'psi': [], 'reta_e': [], 'esq_e': [], 'dir_e': [], 'reta_esterco': []}
    states, flips, line_points = {}, [], 0
    t0 = records[0]['t_capture'] if records else 0.
    for record in records:
        control = record.get('control') or {}
        corridor = (record.get('vision') or {}).get('corridor') or {}
        if not control.get('active') or control.get('obstacle'):
            continue
        state = corridor.get('state')
        if state:
            states[state] = states.get(state, 0) + 1
        sides = corridor.get('sides') or {}
        if sides.get('flipped'):
            flips.append(round(record['t_capture'] - t0, 1))
        line_points += bool(corridor.get('line_points'))
        if not corridor.get('valid') or not corridor.get('errors'):
            continue
        e, psi, kappa = corridor['errors']
        if abs(e) > .6:
            continue
        rows['e'].append(e)
        rows['psi'].append(psi)
        if abs(kappa) < .12:
            rows['reta_e'].append(e)
            rows['reta_esterco'].append(control.get('steer_cmd_rad') or 0.)
        elif kappa >= .25:
            rows['esq_e'].append(e)
        elif kappa <= -.25:
            rows['dir_e'].append(e)
    total = sum(states.values())
    return rows, ({k: round(v/total, 2) for k, v in states.items()} if total else {}), flips, line_points


def line_truth(folder, records):
    """Posicao do carro em relacao a fita amarela (quadros gravados, controle ativo)."""
    import cv2
    from fva_car.lateral import (BirdsEyeView, GroundCalibration, LINE_MIN_SATURATION, line_saturation,
                                 mask_cones)
    from tools.yellow_path import PathSettings, YellowPathDetector
    calibration = GroundCalibration.from_file(ROOT/'calibracao'/'camera_lateral.json')
    view = BirdsEyeView(calibration)
    settings = PathSettings(roi_top=0, min_span=15, h_max=35, s_min=25, dashed=True, dash_gap=120)
    by_frame = {record['frame']: record for record in records}
    offsets = []
    for path in sorted((folder/'frames').glob('f*.jpg')):
        record = by_frame.get(int(path.stem[1:7]))
        if record is None or not (record.get('control') or {}).get('active'):
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        top = view.warp(mask_cones(image))
        detector = YellowPathDetector()
        detector.reset(anchor=.5)
        result = detector.process(top, settings, timestamp=0.)
        if not result.valid or line_saturation(top, result) < LINE_MIN_SATURATION:
            continue
        line = view.to_ground(result.center)
        line = line[(line[:, 0] > .6) & (line[:, 0] < 1.6)]
        if len(line) >= 5:
            fit = np.polyfit(line[:, 0], line[:, 1], 1)
            tape, tape_axle = float(np.polyval(fit, 1.)), float(fit[1])
            errors = ((record.get('vision') or {}).get('corridor') or {}).get('errors')
            # Centro dos cones a 1 m (aproximacao pequena: e + psi*x + kappa*x^2/2).
            cones = errors[0] + errors[1] + errors[2]/2 if errors else math.nan
            offsets.append((tape_axle, tape, cones))
    return offsets


def startup(car, records):
    """Partida: primeiros segundos depois que o controle lateral ligou.

    Devolve dict: alvo (vref de cruzeiro), ate_80_s (tempo ate v >= 80% do alvo;
    nan se nao chegou em 10 s), motor_parou (vezes que v caiu < 0,08 m/s com vref
    > 0,25 nos primeiros 5 s, depois de ter andado), lenta_s (tempo a velocidade de
    recuperacao nos primeiros 10 s), esterco_parado_graus (maior |esterco| com o
    carro parado no primeiro 1,5 s); None sem controle ativo.
    """
    if 'lateral_active' not in car or not (car['lateral_active'] == 1).any():
        return None
    t0 = float(car.loc[car['lateral_active'] == 1, 't'].iloc[0])
    window = car[(car['t'] >= t0) & (car['t'] <= t0 + 10.)]
    active = car[car['lateral_active'] == 1]
    target = float(active['vref'].quantile(.75)) if len(active) else math.nan
    reached = window[window['v'] >= .8*target] if math.isfinite(target) and target > 0 else window.iloc[:0]
    first5 = window[window['t'] <= t0 + 5.]
    moving = (first5['v'] > .15).cumsum() > 0  # depois de ter andado
    stalled = ((first5['v'] < .08) & (first5['vref'] > .25) & moving).astype(int)
    dt = float(np.median(np.diff(car['t']))) if len(car) > 1 else .02
    slow = window['vref'].between(RECOVERY_SPEED - .005, RECOVERY_SPEED + .005)
    steer = 'steer_cmd_rad' if 'steer_cmd_rad' in car else 'st' if 'st' in car else None
    still = window[(window['t'] <= t0 + 1.5) & (window['v'].abs() < .05)]
    return {
        'alvo': target,
        'ate_80_s': float(reached['t'].iloc[0] - t0) if len(reached) else math.nan,
        'motor_parou': int(np.sum(np.diff(np.r_[0, stalled.to_numpy()]) > 0)),
        'lenta_s': float(slow.sum()*dt) if target > RECOVERY_SPEED + .05 else math.nan,
        'esterco_parado_graus': (math.degrees(float(still[steer].abs().max())) if steer and len(still)
                                 else math.nan),
    }


def report(folder, with_line=False):
    import pandas as pd
    folder = Path(folder)
    car = pd.read_csv(folder/'car.csv')
    records = _read_jsonl(folder/'cones_debug.jsonl')
    summary = json.loads((folder/'cones_summary.json').read_text()) if (folder/'cones_summary.json').exists() else {}
    status = json.loads((folder/'vision_status.json').read_text()) if (folder/'vision_status.json').exists() else {}
    lines, warnings = [], []
    ok = lambda good, text: (lines if good else warnings).append(('OK      ' if good else 'ATENCAO ') + text)

    version = summary.get('code_version') or status.get('code_version')
    lines.append(f'== {folder.name}')
    lines.append(f'versao do codigo: {version or "DESCONHECIDA (codigo anterior ao carimbo de versao)"}')
    active = car[car['lateral_active'] == 1] if 'lateral_active' in car else car
    laws = sorted({(r.get('control') or {}).get('steering_law') for r in records if (r.get('control') or {}).get('active')} - {None})
    distance = float(np.sum(np.hypot(np.diff(car['x']), np.diff(car['y'])))) if len(car) > 1 else 0.
    lines.append(f'controle: {", ".join(laws) or "?"} | ativo {len(active)*np.median(np.diff(car["t"])):.1f} s'
                 f' | {distance:.1f} m | v max {car["v"].max():.2f} m/s | vref media {active["vref"].mean():.2f}'
                 f' | visao {status.get("processing_size")}')
    if laws and laws != ['pure_pursuit']:
        warnings.append('ATENCAO controle com Stanley: o plano usa pure pursuit (metade do erro na pista virtual)')

    options = summary.get('options') or {}
    refined = [((r.get('vision') or {}).get('detection') or {}).get('refined_bases') for r in records]
    refined = [v for v in refined if v is not None]
    classified = any(((r.get('vision') or {}).get('detection') or {}).get('classifier') for r in records)
    flag = lambda value: '?' if value is None else 'sim' if value else 'nao'
    lines.append('opcoes: classificador ' + flag(options.get('classifier', classified if records else None))
                 + ' | refino da base ' + flag(options.get('refine_bases', bool(refined) if records else None))
                 + (f' ({np.mean(refined):.1f} cones/quadro)' if refined else '')
                 + ' | partida ' + {True: 'ANTIGA', False: 'nova'}.get(options.get('old_start'), '?')
                 + ('' if 'throttle_ff_offset' not in options or options.get('old_start') else
                    ' | acelerador aprendido' if options.get('throttle_learning') else
                    f' | acelerador minimo fixo {options["throttle_ff_offset"]:.2f}'))
    start = startup(car, records)
    if start is not None:
        reached = start['ate_80_s']
        ok(math.isfinite(reached) and reached <= GOALS['partida_s'],
           f'partida: 80% de {start["alvo"]:.2f} m/s em '
           + (f'{reached:.1f} s' if math.isfinite(reached) else 'mais de 10 s (nao chegou)')
           + f' (meta <= {GOALS["partida_s"]:.1f})'
           + (f' | {start["lenta_s"]:.1f} s presa a {RECOVERY_SPEED:.2f} m/s' if math.isfinite(start['lenta_s']) else ''))
        ok(start['motor_parou'] == 0, f'partida: motor parou {start["motor_parou"]} vez(es) nos primeiros 5 s'
           ' (acelerador minimo baixo? suba na GUI)' if start['motor_parou'] else
           'partida: motor nao parou nos primeiros 5 s')
        if math.isfinite(start['esterco_parado_graus']):
            ok(start['esterco_parado_graus'] < 12., f'partida: maior esterco com o carro parado'
               f' {start["esterco_parado_graus"]:.0f} graus (fim de curso ~20)')

    frames = active.drop_duplicates('vision_frame_id') if 'vision_frame_id' in active else active
    span = max(float(active['t'].max() - active['t'].min()), 1e-3) if len(active) else 1.
    rate = len(frames)/span
    age = 1000*active['vision_age_s'].median() if 'vision_age_s' in active else math.nan
    ok(rate >= GOALS['quadros_por_s'], f'visao: {rate:.1f} quadros/s (meta >= {GOALS["quadros_por_s"]:.0f})')
    ok(age <= GOALS['idade_ms'], f'atraso da imagem: {age:.0f} ms mediana, p90 {1000*active["vision_age_s"].quantile(.9):.0f} ms'
       f' (meta <= {GOALS["idade_ms"]:.0f}) | captura {1000*frames["vision_capture_s"].median():.0f}'
       f' deteccao {1000*frames["vision_detection_s"].median():.0f} geometria {1000*frames["vision_geometry_s"].median():.0f} ms')
    if (folder/'frames').exists() and any((folder/'frames').iterdir()):
        warnings.append('ATENCAO quadros gravados nesta corrida: a visao fica ~20% mais lenta (so em coleta)')

    for stop in stops(car):
        kind = ('uma leitura isolada (eco falso)' if stop['seguidas'] <= 1 and max(stop['depois'][1:] or [0]) > 1.5
                else 'leituras seguidas (obstaculo real?)')
        warnings.append(f'ATENCAO parada pelo ultrassom em t={stop["t"]:.1f} s: {stop["leitura"]:.2f} m, {kind};'
                        f' depois: {stop["depois"]}')
    if 'obstacle_ignored' in car:
        lines.append(f'ultrassom: {int(car["obstacle_ignored"].sum())} passos com eco de cone de borda ignorado em curva')

    rows, states, flips, line_points = tracking(records)
    if rows['e']:
        rms_e = 100*np.sqrt(np.mean(np.square(rows['e'])))
        rms_psi = math.degrees(np.sqrt(np.mean(np.square(rows['psi']))))
        ok(rms_e <= GOALS['erro_rms_cm'], f'erro lateral (estimador): {rms_e:.1f} cm RMS (meta <= {GOALS["erro_rms_cm"]:.0f})')
        ok(rms_psi <= GOALS['rumo_rms_graus'], f'rumo: {rms_psi:.1f} graus RMS (meta <= {GOALS["rumo_rms_graus"]:.0f}; acima disso: oscilacao)')
        mean = lambda values: 100*np.mean(values) if values else math.nan
        lines.append(f'centro estimado em relacao ao carro (+ = carro a direita): reta {mean(rows["reta_e"]):+.1f} cm'
                     f' | curva esq {mean(rows["esq_e"]):+.1f} | curva dir {mean(rows["dir_e"]):+.1f}'
                     f' | esterco medio na reta {math.degrees(np.mean(rows["reta_esterco"])) if rows["reta_esterco"] else math.nan:+.1f} graus')
    lines.append(f'estados do corredor: {states}')
    ok(not flips, f'trocas de lado das cores com o carro andando: {flips or "nenhuma"}')
    if line_points:
        lines.append(f'fita amarela usada no estimador em {line_points} quadros')
    if 'steer_trim_rad' in car and car['steer_trim_rad'].notna().any():
        lines.append(f'trim da direcao no fim: {math.degrees(car["steer_trim_rad"].dropna().iloc[-1]):+.2f} graus')
    if summary.get('learned_width_m'):
        lines.append(f'largura aprendida: {summary["learned_width_m"]:.2f} m')

    if with_line:
        pairs = np.asarray(line_truth(folder, records)).reshape(-1, 3)
        offsets = pairs[:, 0]  # fita no eixo do carro: a posicao real, sem o efeito do rumo
        if len(offsets) >= 5:
            rms = 100*np.sqrt(np.mean(offsets**2))
            ok(rms <= GOALS['erro_rms_cm'], f'POSICAO REAL (fita amarela no eixo, {len(offsets)} quadros): carro {100*np.median(offsets):+.1f} cm'
               f' (+ = carro a direita da fita), RMS {rms:.1f} cm, p90 |.| {100*np.percentile(np.abs(offsets), 90):.1f} cm')
            bias = pairs[:, 2] - pairs[:, 1]
            bias = bias[np.isfinite(bias)]
            if len(bias) >= 5:
                # Calibracao boa: o centro achado pelos cones cai em cima da fita.
                ok(abs(np.median(bias)) <= .03, f'CALIBRACAO (centro dos cones - fita, a 1 m): {100*np.median(bias):+.1f} cm'
                   f' mediana (+ = cones poem o centro a esquerda da fita; meta +-3 cm)')
        else:
            lines.append('posicao real: fita amarela em menos de 5 quadros gravados (sem --record-frames?)')
    text = '\n'.join(lines + warnings)
    (folder/'relatorio.txt').write_text(text + '\n', encoding='utf-8')
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ensaios', nargs='+', type=Path)
    parser.add_argument('--linha', action='store_true', help='mede a posicao real pela fita amarela (quadros gravados)')
    args = parser.parse_args(argv)
    for folder in args.ensaios:
        if (folder/'car.csv').exists():
            print(report(folder, args.linha), end='\n\n', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
