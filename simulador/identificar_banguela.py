"""Identifica perdas em banguela e compara uma dinamica longitudinal nos dados."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def load_trials(folder):
    manifest = json.loads((folder / 'serie.json').read_text(encoding='utf-8'))
    trials = []
    for info in manifest['trials']:
        if not info['complete']:
            raise ValueError(f'Ensaio incompleto: {info["file"]}')
        with (folder / info['file']).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        data = {name: np.array([float(row[name]) for row in rows])
                for name in ['t', 'v_raw', 'u_motor', 'torque_L', 'torque_R']}
        data['coast'] = np.array([row['phase'] == 'coast' for row in rows])
        if len(data['t']) < 8 or np.any(np.diff(data['t']) <= 0):
            raise ValueError(f'Amostragem invalida: {info["file"]}')
        coast = data['coast']
        if not np.any(coast) or np.any(data['torque_L'][coast] != 0) or np.any(data['torque_R'][coast] != 0):
            raise ValueError(f'Banguela deve ter torque zero: {info["file"]}')
        trials.append({'info': info, 'data': data})
    if len(trials) != len(manifest['speeds']) * manifest['repetitions']:
        raise ValueError('A serie nao contem todas as repeticoes planejadas.')
    return manifest, trials


def nnls_small(matrix, response):
    """Minimos quadrados nao negativos por conjuntos ativos (ate 3 termos)."""
    best = np.zeros(matrix.shape[1])
    error = float(response @ response)
    for count in range(1, matrix.shape[1] + 1):
        for active in itertools.combinations(range(matrix.shape[1]), count):
            solution, _, rank, _ = np.linalg.lstsq(matrix[:, active], response, rcond=None)
            if rank != count or np.any(solution < 0):
                continue
            candidate = np.zeros(matrix.shape[1])
            candidate[list(active)] = solution
            residual = response - matrix @ candidate
            if residual @ residual < error:
                best, error = candidate, float(residual @ residual)
    return best


def integral_equations(trials):
    matrix, response = [], []
    for trial in trials:
        data = trial['data']
        t, v = data['t'], data['v_raw']
        release_t = t[np.flatnonzero(data['coast'])[0]]
        # Exclui a comutacao inicial e o regime de aderencia proximo de zero.
        indices = np.flatnonzero(data['coast'] & (t >= release_t + 0.10) & (v >= 0.15))
        for start in range(0, len(indices) - 4, 4):
            ids = indices[start:start + 5]
            if np.any(np.diff(ids) != 1):
                continue
            ti, vi = t[ids], v[ids]
            dt = np.diff(ti)
            matrix.append([ti[-1] - ti[0],
                           np.sum(0.5 * (vi[1:] + vi[:-1]) * dt),
                           np.sum(0.5 * (vi[1:]**2 + vi[:-1]**2) * dt)])
            response.append(vi[0] - vi[-1])
    if len(matrix) < 6:
        raise ValueError('Poucos dados uteis de banguela para identificar as perdas.')
    return np.asarray(matrix), np.asarray(response)


def simulate(t, inputs, initial_v, loss, gains, tau):
    """q e a aceleracao motriz; v_dot=q-(c0+c1*v+c2*v^2)."""
    gains = np.atleast_1d(gains)
    v = np.full(gains.shape, initial_v)
    q = gains * inputs[0]
    prediction = [v.copy()]
    for i, interval in enumerate(np.diff(t)):
        steps = max(1, int(np.ceil(interval / 0.005)))
        dt = interval / steps
        for _ in range(steps):
            if tau > 0:
                steady = gains * inputs[i]
                decay = np.exp(-dt / tau)
                q_mean = steady + (q - steady) * tau / dt * (1.0 - decay)
                q = steady + (q - steady) * decay
            else:
                q = gains * inputs[i]
                q_mean = q
            resistance = loss[0] + loss[1] * v + loss[2] * v**2
            v = np.maximum(0.0, v + (q_mean - resistance) * dt)
        prediction.append(v.copy())
    return np.asarray(prediction)


def coast_prediction(trial, coefficients):
    data = trial['data']
    indices = np.flatnonzero(data['coast'])
    t, measured = data['t'][indices], data['v_raw'][indices]
    prediction = simulate(t, np.zeros(len(t)), measured[0], coefficients, [1.0], 0.0)[:, 0]
    return t, measured, prediction


def coast_rmse(trials, loss):
    errors = [measured - predicted for trial in trials
              for _, measured, predicted in [coast_prediction(trial, loss)]]
    return float(np.sqrt(np.mean(np.concatenate(errors)**2)))


def dynamic_data(trial):
    data = trial['data']
    # O modelo solicitado cobre 0.5--2.5 m/s, nao a partida/aderencia em zero.
    # Usa 0.45 para incluir a tolerancia de soltura do ensaio de 0.5 m/s.
    indices = np.flatnonzero((data['t'] >= 0.30) & (data['v_raw'] >= 0.45))
    if len(indices) < 5:
        raise ValueError('Trecho dinamico insuficiente na faixa de operacao.')
    # Preserva todos os instantes entre os cruzamentos: ruido na fronteira
    # nao pode apagar amostras e criar um salto artificial no tempo.
    section = slice(indices[0], indices[-1] + 1)
    return data['t'][section], data['u_motor'][section], data['v_raw'][section]


def identify(folder, plots=True):
    manifest, trials = load_trials(folder)
    last_rep = manifest['repetitions']
    train = [trial for trial in trials if trial['info']['repetition'] < last_rep]
    validation = [trial for trial in trials if trial['info']['repetition'] == last_rep]
    matrix, response = integral_equations(train)
    selection = [trial for trial in train if trial['info']['repetition'] == last_rep - 1] if last_rep >= 3 else []
    calibration = [trial for trial in train if trial['info']['repetition'] < last_rep - 1] if selection else train
    selection_matrix, selection_response = integral_equations(calibration or train)
    candidates = []
    for terms in [1, 2, 3]:
        loss = np.zeros(3)
        loss[:terms] = nnls_small(selection_matrix[:, :terms], selection_response)
        if selection:
            score = coast_rmse(selection, loss)
        else:
            residual = selection_response - selection_matrix @ loss
            count = len(residual)
            score = count * np.log(max(np.mean(residual**2), 1e-12)) + np.count_nonzero(loss) * np.log(count)
        candidates.append({'terms': terms, 'loss': loss,
                           'selection_score': float(score)})
    lowest_error = min(candidate['selection_score'] for candidate in candidates)
    # Prefere simplicidade se o ganho no erro for menor que 0.005 m/s.
    selected = next(candidate for candidate in candidates
                    if candidate['selection_score'] <= lowest_error + (0.005 if selection else 0.0))
    loss = np.zeros(3)
    loss[:selected['terms']] = nnls_small(matrix[:, :selected['terms']], response)
    gains = np.linspace(0.6, 1.6, 101)
    sample_time = float(np.median(np.diff(trials[0]['data']['t'])))
    taus = sorted(set([0.0, sample_time / 2, sample_time, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]))
    grid = []
    for tau in taus:
        residuals = []
        for trial in train:
            t, u, measured = dynamic_data(trial)
            predicted = simulate(t, u, measured[0], loss, gains, tau)
            residuals.append((predicted - measured[:, None])**2)
        mse = np.mean(np.concatenate(residuals), axis=0)
        idx = int(np.argmin(mse))
        grid.append({'tau': tau, 'gain': float(gains[idx]), 'train_rmse': float(np.sqrt(mse[idx]))})
    best = min(grid, key=lambda candidate: candidate['train_rmse'])
    gain, tau = best['gain'], best['tau']
    validation_metrics = []
    reduced_metrics = []
    for trial in validation:
        t, u, measured = dynamic_data(trial)
        predicted = simulate(t, u, measured[0], loss, [gain], tau)[:, 0]
        validation_metrics.append({
            'target': trial['info']['target'], 'file': trial['info']['file'],
            'rmse_m_s': float(np.sqrt(np.mean((predicted - measured)**2))),
            'max_error_m_s': float(np.max(np.abs(predicted - measured))),
        })
        reduced = simulate(t, u, measured[0], loss, [gain], 0.0)[:, 0]
        reduced_metrics.append({'target': trial['info']['target'],
                                'rmse_m_s': float(np.sqrt(np.mean((reduced - measured)**2)))})
    valid = all(metric['rmse_m_s'] <= 0.05 for metric in validation_metrics)
    resolved_tau = tau > sample_time
    reduced_valid = all(metric['rmse_m_s'] <= 0.05 for metric in reduced_metrics)
    recommended = None
    if selected['terms'] == 1 and valid and (resolved_tau or reduced_valid):
        recommended = {'gain_for_set_u': 1.26 * gain,
                       'tau_s': tau if resolved_tau else 0.0,
                       'formula': f'{1.26 * gain:.4f}/[s*({tau:.4f}*s+1)]' if resolved_tau else f'{1.26 * gain:.4f}/s',
                       'note': 'Incremental plant around forward motion on flat track. tau=0 here is a reduced approximation, not proof of zero physical lag.'}
    report = {
        'model': 'q_dot=(K*u_motor-q)/tau; v_dot=q-c0-c1*v-c2*v^2; tau=0 means q=K*u_motor',
        'coast_loss': {'c0_m_s2': float(loss[0]), 'c1_s_inv': float(loss[1]),
                       'c2_m_inv': float(loss[2]), 'terms': selected['terms'],
                       'validation_rmse_m_s': coast_rmse(validation, loss)},
        'gain_motor_normalized': gain, 'tau_best_grid_s': tau,
        'tau_resolved': resolved_tau, 'sample_time_s': sample_time,
        'input': 'u_motor=(torque_L+torque_R)/(mass_nominal*radius_nominal), m/s^2',
        'identification': 'Fit loss on coast; fit K,tau on accelerate+hold+coast, first repetitions only',
        'loss_model_selection': 'Second repetition for selection, then refit on first+second; last repetition never used for tuning. With two repetitions, use training BIC.',
        'dynamic_validation_band_m_s': [0.45, max(manifest['speeds']) + 0.05],
        'excluded_from_dynamic_model': 'startup/stiction below 0.45 m/s; slopes; reverse',
        'tau_grid': grid, 'validation': validation_metrics,
        'reduced_tau_zero_validation': reduced_metrics,
        'recommended_linear_model': recommended,
        'passes_0_05_m_s_rmse': valid, 'speed_range_m_s': [min(manifest['speeds']), max(manifest['speeds'])],
        'local_linear_plants': [
            {'operating_speed_m_s': speed,
             'gain_for_set_u': 1.26 * gain,
             'damping_s_inv': float(loss[1] + 2 * loss[2] * speed),
             'formula': 'G_u(s)=1.26*K/[(tau*s+1)*(s+c1+2*c2*v0)]'}
            for speed in manifest['speeds']
        ],
        'photo_model': {
            'formula': 'P(s)=1/[s*(tau*s+1)]',
            'status': 'not_confirmed' if not valid or not resolved_tau or selected['terms'] > 1 else 'compatible_after_input_normalization',
            'note': 'Coast alone does not identify input gain or actuator tau. '
                    'Do not treat an unresolved tau as an identified positive time constant. '
                    'A unity numerator requires normalizing the input by the identified gain.',
        },
    }
    (folder / 'identificacao.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Resultados dos ensaios de banguela', '',
             f"Serie: {len(trials)} ensaios, {min(manifest['speeds']):.1f} a {max(manifest['speeds']):.1f} m/s, amostragem {1000*sample_time:.0f} ms.", '',
             f'Perdas: dv/dt = -({loss[0]:.4f} + {loss[1]:.4f}*v + {loss[2]:.4f}*v^2).', '',
             f'Ganho normalizado do motor: {gain:.3f}. Ganho incremental para set_u: {1.26*gain:.4f}.', '',
             f"Melhor tau da grade: {1000*tau:.1f} ms. Atraso resolvido: {'sim' if resolved_tau else 'nao'}.", '',
             'Planta incremental recomendada: ' + (recommended['formula'] if recommended else 'nenhuma; ajuste ainda insuficiente') + '.', '',
             'A aproximacao reduzida com tau=0 nao comprova ausencia de atraso fisico. '
             'O modelo descreve variacoes de velocidade em movimento para frente no plano; '
             'o atrito constante deve ser tratado como perturbacao/compensacao. '
             'A partida, a parada, re e rampas ficam fora desta validacao.', '',
             '| Velocidade inicial desejada [m/s] | RMSE do melhor ajuste [m/s] | RMSE do integrador reduzido [m/s] |',
             '|---:|---:|---:|']
    lines += [f"| {full['target']:.1f} | {full['rmse_m_s']:.4f} | {simple['rmse_m_s']:.4f} |"
              for full, simple in zip(validation_metrics, reduced_metrics)]
    lines += ['', 'A ultima repeticao de cada velocidade foi reservada para validacao. '
              'As velocidades reais de soltura estao em serie.json. '
              'Os CSVs preservam as entradas aplicadas e todas as fases do ensaio.', '']
    (folder / 'RESULTADOS.md').write_text('\n'.join(lines), encoding='utf-8')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for trial in validation:
            t, measured, predicted = coast_prediction(trial, loss)
            line, = axes[0].plot(t - t[0], measured, label=f"{trial['info']['target']:.1f} m/s")
            axes[0].plot(t - t[0], predicted, '--', color=line.get_color())
            t, u, measured = dynamic_data(trial)
            predicted = simulate(t, u, measured[0], loss, [gain], tau)[:, 0]
            line, = axes[1].plot(t, measured)
            axes[1].plot(t, predicted, '--', color=line.get_color())
        axes[0].set_title('Banguela: repeticoes reservadas para validacao')
        axes[0].set_xlabel('Tempo desde a soltura [s]')
        axes[0].legend()
        axes[1].set_title('Missao: medido (linha) e previsto (tracejado)')
        axes[1].set_xlabel('Tempo da simulacao [s]')
        for ax in axes:
            ax.set_ylabel('Velocidade longitudinal [m/s]')
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(folder / 'validacao.png', dpi=150)
        plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    identify(args.folder, plots=not args.no_plots)
