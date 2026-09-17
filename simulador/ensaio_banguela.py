"""Serie de ensaios planos, com velocidade inicial controlada e torque zero."""
import argparse
import csv
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from fva_car.car import CAR, Car

TRACKS = {
    'cones': {'file': 'simulador_cones.ttt', 'start': (-24.0, -8.45), 'end_x': 21.0},
    # Para identificar perdas, usa somente o plano ANTES da rampa.
    'rampa': {'file': 'simulador_rampa.ttt', 'start': (-25.0, -8.5), 'end_x': -6.0},
}
FIELDS = ['t', 'phase', 'target', 'v_raw', 'v_filtered', 'x', 'y', 'z',
          'u', 'torque_L', 'torque_R', 'u_motor', 'dt']


def stop_simulation(car):
    """Encerra sem incluir uma frenagem ativa nos dados da banguela."""
    car.set_coast()
    car.sim.stopSimulation()
    deadline = time.monotonic() + 10.0
    while car.sim.getSimulationState() != car.sim.simulation_stopped:
        if time.monotonic() > deadline:
            raise RuntimeError('O simulador nao encerrou em 10 s.')
        time.sleep(0.01)


def run_trial(car, track, target, filename, max_time=25.0):
    phase = 'accelerate'
    stable_since = None
    stopped_since = None
    release = None
    reason = 'timeout'
    car.start_mission()
    with filename.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        while car.t < max_time:
            pos = car.sim.getObjectPosition(car.robot, car.sim.handle_world)
            if pos[0] >= track['end_x']:
                reason = 'limite_do_trecho_plano'
                break
            # Faixa de cerca de 4 m; reserva mais de 0.4 m para carro e borda.
            if abs(pos[1] - track['start'][1]) > 1.4:
                reason = 'desvio_lateral'
                break
            if not 0.05 < pos[2] < 0.35:
                reason = 'altura_fora_do_plano'
                break
            car.set_steer(0.0)
            if phase != 'coast':
                car.set_vel(target)
                # Espera contato estabilizado e velocidade REAL, sem depender
                # apenas do filtro ou de um tempo fixo de aceleracao.
                settled = car.t > 0.3 and abs(car.v_raw - target) <= max(0.02, 0.02 * target)
                if settled:
                    if stable_since is None:
                        stable_since = car.t
                    phase = 'hold'
                    if car.t - stable_since >= 1.0:
                        phase = 'coast'
                        car.set_coast()
                        release = {'t': float(car.t), 'v': float(car.v_raw), 'x': pos[0]}
                else:
                    stable_since = None
                    phase = 'accelerate'
            else:
                car.set_coast()
            # Linha em t: estado medido em t e entrada mantida em [t,t+dt).
            # Os dados do controlador e da banguela ficam separados por phase.
            writer.writerow({
                't': car.t, 'phase': phase, 'target': target,
                'v_raw': car.v_raw, 'v_filtered': car.v,
                'x': pos[0], 'y': pos[1], 'z': pos[2],
                'u': car.u, 'torque_L': car.motor_torque,
                'torque_R': car.motor_torque,
                'u_motor': 2 * car.motor_torque / (CAR['MASS'] * CAR['RW']),
                'dt': car.dt,
            })
            if phase == 'coast' and abs(car.v_raw) < 0.08:
                if stopped_since is None:
                    stopped_since = car.t
                elif car.t - stopped_since >= 0.15:
                    reason = 'parou_em_banguela'
                    break
            else:
                stopped_since = None
            if not car.step():
                reason = 'interrompido'
                break
        stream.flush()
    return {
        'file': filename.name, 'target': target, 'release': release,
        'complete': reason == 'parou_em_banguela', 'reason': reason,
        'end_time': float(car.t),
    }


def collect(args):
    track = TRACKS[args.scene]
    # O primeiro cliente permanece em stepping; nenhuma pista em uso e salva.
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient(port=args.port)
    sim = client.getObject('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Pare a simulacao antes de iniciar a serie.')
    sim.setStepping(True)
    scene_path = Path(__file__).resolve().parent / 'coppeliasim' / track['file']
    output = args.output or (Path(__file__).resolve().parent / 'ensaios' / 'dados' /
                             datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'scene': args.scene, 'speeds': args.speeds, 'repetitions': args.repetitions,
        'sample_time_s': args.sample_time,
        'car_parameters': {key: float(value) for key, value in CAR.items()},
        'track': track, 'input_alignment': 'u(t) held on [t, t_next)',
        'speed_measurement': 'signed projection on Car local Z; no filter',
        'trials': [],
    }
    manifest_path = output / 'serie.json'
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        previous.setdefault('sample_time_s', 0.05)
        for key in ['scene', 'speeds', 'repetitions', 'car_parameters', 'track', 'sample_time_s']:
            # JSON converte a tupla da posicao inicial em lista.
            if previous[key] != json.loads(json.dumps(manifest[key])):
                raise ValueError(f'Configuracao diferente na retomada: {key}')
        manifest = previous
        for result in list(manifest['trials']):
            if not result['complete']:
                original = output / result['file']
                failed = original.with_name(original.stem + '_incompleto_' +
                                            datetime.now().strftime('%H%M%S_%f') + '.csv')
                if original.exists():
                    original.rename(failed)
                archived = dict(result, file=failed.name)
                manifest.setdefault('failed_attempts', []).append(archived)
                manifest['trials'].remove(result)
    manifest['lateral_limit_m'] = 1.4
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    try:
        for speed in args.speeds:
            for repetition in range(1, args.repetitions + 1):
                if any(result['complete'] and result['target'] == speed and
                       result['repetition'] == repetition for result in manifest['trials']):
                    continue
                sim.loadScene(str(scene_path))
                sim.setFloatParam(sim.floatparam_simulation_time_step, args.sample_time)
                if not np.isclose(sim.getSimulationTimeStep(), args.sample_time):
                    raise RuntimeError('O simulador nao aceitou o intervalo de amostragem.')
                h = sim.getObject('/Car')
                p = sim.getObjectPosition(h, sim.handle_world)
                sim.setObjectPosition(h, [*track['start'], p[2]], sim.handle_world)
                name = f'banguela_{speed:.2f}ms_rep{repetition:02d}.csv'
                print(f'Ensaio {speed:.2f} m/s, repeticao {repetition}/{args.repetitions}', flush=True)
                with tempfile.TemporaryDirectory(prefix='fva_car_') as temp_logs:
                    car = Car({'ts': 25.0, 'save': False, 'logfile': temp_logs,
                               'beep': False, 'port': args.port})
                    try:
                        result = run_trial(car, track, speed, output / name)
                    except BaseException as error:
                        manifest['trials'].append({'file': name, 'target': speed,
                                                  'complete': False, 'reason': type(error).__name__,
                                                  'repetition': repetition})
                        save_manifest()
                        raise
                    finally:
                        stop_simulation(car)
                result['repetition'] = repetition
                manifest['trials'].append(result)
                save_manifest()
                print(f"  {result['reason']}; velocidade de soltura: "
                      f"{result['release']['v'] if result['release'] else None}", flush=True)
                if not result['complete']:
                    raise RuntimeError(f'Ensaio incompleto: {result["reason"]}. Dados preservados em {output}')
    finally:
        sim.setStepping(False)
    print(f'Dados salvos em {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=TRACKS, default='cones')
    parser.add_argument('--port', type=int, default=23000)
    parser.add_argument('--sample-time', type=float, choices=[0.005, 0.01, 0.02, 0.05], default=0.01)
    parser.add_argument('--speeds', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true', help='Retoma a serie indicada em --output')
    args = parser.parse_args()
    if args.resume and args.output is None:
        parser.error('--resume requer --output.')
    if args.repetitions < 2:
        parser.error('Use ao menos 2 repeticoes para separar ajuste e validacao.')
    if len(set(args.speeds)) != len(args.speeds):
        parser.error('As velocidades nao podem estar duplicadas.')
    if any(not np.isfinite(v) or not 0.5 <= v <= 2.5 for v in args.speeds):
        parser.error('As velocidades devem estar entre 0.5 e 2.5 m/s.')
    collect(args)


if __name__ == '__main__':
    main()
