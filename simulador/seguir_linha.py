"""Ensaio opt-in: camera -> linha amarela -> Stanley + PI longitudinal.

Na raiz: venv/bin/python simulador/seguir_linha.py --vref 1.5 --ts 60
Abra simulador_cones.ttt e mantenha a simulacao parada antes de executar.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy.stanley import StanleyConfig, StanleyController, curve_speed
from autonomy.yellow_line import GroundProjection, YellowLineDetector
from fva_car.car import CAR
from main import parameters, run_mission

# Medido na cena simulador_cones: set_steer positivo gira para a direita.
# O nucleo Stanley usa a convencao matematica, positiva para a esquerda.
SIM_STEERING_SIGN = -1.0
VELOCITY_REFERENCE = 1.5  # m/s
STANLEY_GAIN = 5.0  # 1/s; validado em ensaios comparativos e volta completa
SIMULATION_TIME = 120.0  # s


def place_car(sim, pose):
    """Reposiciona um ensaio parado; retorna a matriz original para restauracao."""
    if not all(math.isfinite(v) for v in pose):
        raise ValueError("Pose inicial deve conter valores finitos.")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise ValueError("Pare a simulacao antes de usar --initial-pose.")
    robot = sim.getObject("/Car")
    original = sim.getObjectMatrix(robot, -1)
    matrix = np.asarray(original).reshape(3, 4).copy()
    heading = math.atan2(matrix[1, 2], matrix[0, 2])
    angle = math.radians(pose[2]) - heading
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    matrix[:, :3] = rotation @ matrix[:, :3]
    matrix[0, 3], matrix[1, 3] = pose[:2]
    sim.setObjectMatrix(robot, matrix.ravel().tolist(), -1)
    return robot, original


def camera_projection(car, ground_z=0.00205):
    """Calibracao estatica no plano da pista, com suspensao estabilizada.

    Usa geometria apenas para calibrar a camera; nao consulta a trajetoria.
    A partir daqui, os erros laterais vem exclusivamente da imagem.
    """
    sim = car.sim
    resolution = sim.getVisionSensorRes(car.cam)
    if not sim.getObjectInt32Param(car.cam, sim.visionintparam_perspective_operation):
        raise ValueError("O ensaio requer uma camera em perspectiva.")
    fov = sim.getObjectFloatParam(car.cam, sim.visionfloatparam_perspective_angle)
    focal = max(resolution) / (2 * math.tan(fov / 2))
    width, height = resolution
    intrinsics = np.array(
        [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]
    )
    camera = np.asarray(sim.getObjectMatrix(car.cam, -1)).reshape(3, 4)
    robot = np.asarray(sim.getObjectMatrix(car.robot, -1)).reshape(3, 4)
    forward = robot[:, 2].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    axle = np.mean(
        [sim.getObjectPosition(h, -1) for h in (car.steerL, car.steerR)], axis=0
    )
    axle[2] = ground_z
    plane = np.column_stack((forward, left, axle - camera[:, 3]))
    # Sensor Coppelia: +Z optico, +X para a esquerda, +Y para cima.
    # Car.get_image entrega RGB e ja inverte verticalmente o buffer.
    ground_to_image = intrinsics @ np.diag([-1, -1, 1]) @ camera[:, :3].T @ plane
    return GroundProjection(np.linalg.inv(ground_to_image), resolution)


class YellowLineMission:
    def __init__(
        self,
        vref=VELOCITY_REFERENCE,
        gain=STANLEY_GAIN,
        lateral_acceleration=0.5,
        ground_z=0.00205,
        camera_pitch=30.0,
    ):
        if not 0 < vref <= CAR["VELMAX"]:
            raise ValueError(f"vref deve estar entre 0 e {CAR['VELMAX']} m/s.")
        if not math.isfinite(lateral_acceleration) or lateral_acceleration <= 0:
            raise ValueError("Aceleracao lateral deve ser positiva e finita.")
        self.vref = vref
        self.lateral_acceleration = lateral_acceleration
        if not math.isfinite(ground_z):
            raise ValueError("Altura do piso deve ser finita.")
        self.ground_z = ground_z
        if not math.isfinite(camera_pitch) or not 5 <= camera_pitch <= 60:
            raise ValueError("Inclinacao da camera deve estar entre 5 e 60 graus.")
        self.camera_pitch = camera_pitch
        self.original_camera = None
        self.stanley = StanleyController(
            StanleyConfig(gain=gain, max_steer=CAR["STEERMAX"])
        )
        self.detector = None
        self.last_seen = None
        self.last_estimate = None
        self.good_frames = 0
        self.stopped = True
        self.target_speed = 0.0
        self.last_image = None
        self.logfile = None
        self.next_status = 10.0

    def __call__(self, car):
        if self.detector is None and self.original_camera is None:
            sim = car.sim
            original = sim.getObjectMatrix(car.cam, car.robot)
            self.original_camera = (sim, car.cam, car.robot, original)
            camera = np.asarray(sim.getObjectMatrix(car.cam, -1)).reshape(3, 4)
            current_pitch = math.atan2(-camera[2, 2], np.linalg.norm(camera[:2, 2]))
            angle = math.radians(self.camera_pitch) - current_pitch
            rotation = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(angle), -math.sin(angle)],
                    [0, math.sin(angle), math.cos(angle)],
                ]
            )
            matrix = np.asarray(original).reshape(3, 4).copy()
            matrix[:, :3] = matrix[:, :3] @ rotation
            sim.setObjectMatrix(car.cam, matrix.ravel().tolist(), car.robot)
        if car.t < 0.5:
            # Evita calibrar durante a queda inicial e assentamento da suspensao.
            car.set_steer(0.0)
            car.set_coast()
            car.log(line_valid=0, line_stop=1)
            return
        if self.detector is None:
            self.logfile = car.logfile
            projection = camera_projection(car, self.ground_z)
            self.detector = YellowLineDetector(projection, color_order="RGB")
            path = Path(car.logfile) / "camera_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "resolution": projection.resolution,
                        "image_to_ground": projection.matrix.tolist(),
                        "color_order": "RGB",
                        "ground_z": self.ground_z,
                        "camera_pitch_deg": self.camera_pitch,
                        "origin": "front_axle",
                        "axes": "x_forward_y_left",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            config = self.stanley.config
            (Path(car.logfile) / "controller_config.json").write_text(
                json.dumps(
                    {
                        "controller": "Stanley",
                        "reference_speed_m_s": self.vref,
                        "gain_s_inv": config.gain,
                        "soft_speed_m_s": config.soft_speed,
                        "max_steer_deg": math.degrees(config.max_steer),
                        "max_steer_rate_deg_s": math.degrees(config.max_steer_rate),
                        "max_lateral_acceleration_m_s2": self.lateral_acceleration,
                        "camera_pitch_deg": self.camera_pitch,
                        "sample_time_s": car.parameters.get("sample_time"),
                        "duration_s": car.parameters.get("ts"),
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                "Camera calibrada; aguardando 3 quadros validos da linha.", flush=True
            )

        image = car.get_image(gray=False)
        self.last_image = image
        estimate = self.detector.detect(image)
        valid = estimate is not None and estimate.confidence >= 0.35
        if valid:
            self.last_seen = car.t
            self.last_estimate = estimate
            self.good_frames += 1
        else:
            self.good_frames = 0
        age = car.t - self.last_seen if self.last_seen is not None else math.inf
        should_stop = (
            self.last_seen is None
            or age > 0.15
            or (self.stopped and self.good_frames < 3)
        )

        if should_stop:
            if not self.stopped:
                print(f"t={car.t:.2f}s: linha perdida; freando.", flush=True)
                car.speed_controller.reset()
            self.stopped = True
            self.target_speed = 0.0
            car.vref = 0.0
            if abs(car.v) > 0.05:
                car.set_u(-CAR["ACCELMAX"])
            else:
                car.set_coast()
                car.set_steer(0.0)
                self.stanley.reset()
        else:
            if self.stopped:
                car.speed_controller.reset()
                print(
                    f"t={car.t:.2f}s: linha adquirida; seguindo a {self.vref:.2f} m/s.",
                    flush=True,
                )
            self.stopped = False
            line = self.last_estimate
            steering = self.stanley.update(
                line.cross_track, line.heading_error, car.v, car.dt
            )
            car.set_steer(SIM_STEERING_SIGN * steering)
            target = curve_speed(self.vref, line.curvature, self.lateral_acceleration)
            if not valid:
                target = min(target, 0.3)
            # Rampa de referencia, com desaceleracao maior para entrada em curva.
            change = target - self.target_speed
            rate = 0.4 if change >= 0 else 0.8
            self.target_speed += float(np.clip(change, -rate * car.dt, rate * car.dt))
            car.set_vel(self.target_speed)

        car.vision_frame = self.detector.annotate(image, estimate if valid else None)
        car.log(
            line_valid=int(valid),
            line_stop=int(self.stopped),
            line_confidence=estimate.confidence if estimate is not None else 0.0,
            cross_track_m=estimate.cross_track if valid else math.nan,
            heading_error_rad=estimate.heading_error if valid else math.nan,
            line_curvature_inv_m=estimate.curvature if valid else math.nan,
            line_fit_rms_m=estimate.residual if valid else math.nan,
            line_circular_fit=int(estimate.circular) if valid else 0,
            stanley_delta_rad=self.stanley.steering,
            stanley_gain_s_inv=self.stanley.config.gain,
            stanley_heading_term_rad=self.stanley.heading_term,
            stanley_cross_track_term_rad=self.stanley.cross_track_term,
            stanley_raw_delta_rad=self.stanley.raw_steering,
            stanley_target_delta_rad=self.stanley.target_steering,
            applied_steer_rad=float(car.st),
            commanded_speed_m_s=self.target_speed,
            vision_age_s=age if math.isfinite(age) else math.nan,
        )
        if car.t >= self.next_status:
            error = (
                f"e={estimate.cross_track:+.3f} m" if valid else "linha indisponivel"
            )
            print(
                f"t={car.t:.1f}s | v={car.v:.2f} m/s | {error} | parada={int(self.stopped)}",
                flush=True,
            )
            self.next_status = car.t + 10.0

    def save_diagnostics(self):
        if self.logfile is None or self.last_image is None:
            return
        import cv2
        from relatorio_stanley import generate_report

        folder = Path(self.logfile)
        overlay = self.detector.annotate(
            self.last_image, self.last_estimate if not self.stopped else None
        )
        cv2.imwrite(
            str(folder / "camera_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(str(folder / "yellow_mask.png"), self.detector.mask)
        cv2.imwrite(str(folder / "bird_mask.png"), self.detector.bird_mask)
        if (folder / "car.csv").exists():
            metrics = generate_report(folder)
            print(f'Relatorio: {folder / "stanley_report.png"}', flush=True)
            print(json.dumps(metrics, indent=2), flush=True)

    def restore_camera(self):
        if self.original_camera is not None:
            sim, camera, robot, original = self.original_camera
            sim.setObjectMatrix(camera, original, robot)
            self.original_camera = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vref",
        type=float,
        default=VELOCITY_REFERENCE,
        help="velocidade de cruzeiro [m/s]",
    )
    parser.add_argument(
        "--ganho", type=float, default=STANLEY_GAIN, help="ganho k do Stanley [1/s]"
    )
    parser.add_argument("--ts", type=float, default=SIMULATION_TIME, help="duracao [s]")
    parser.add_argument(
        "--port", type=int, default=23000, help="porta ZMQ do CoppeliaSim"
    )
    parser.add_argument(
        "--ground-z", type=float, default=0.00205, help="altura do plano da pista [m]"
    )
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=30,
        help="inclinacao da camera para baixo [graus]",
    )
    parser.add_argument(
        "--initial-pose",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW_DEG"),
        help="pose inicial de teste [m, m, graus]; restaurada ao terminar",
    )
    parser.add_argument("--sem-grafico", action="store_true")
    parser.add_argument("--sem-tempo-real", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.ts) or args.ts <= 0:
        parser.error("--ts deve ser positivo e finito.")
    if not math.isfinite(args.ground_z):
        parser.error("--ground-z deve ser finito.")
    mission = YellowLineMission(
        args.vref, args.ganho, ground_z=args.ground_z, camera_pitch=args.camera_pitch
    )
    placement = None
    try:
        if args.initial_pose is not None:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient

            sim = RemoteAPIClient(port=args.port).require("sim")
            robot, original = place_car(sim, args.initial_pose)
            placement = (sim, robot, original)
        run_mission(
            dict(
                parameters,
                ts=args.ts,
                port=args.port,
                beep=False,
                real_time=not args.sem_tempo_real,
                lateral_telemetry=True,
            ),
            display=not args.sem_grafico,
            control=mission,
        )
    finally:
        try:
            mission.restore_camera()
        finally:
            try:
                if placement is not None:
                    sim, robot, original = placement
                    sim.setObjectMatrix(robot, original, -1)
            finally:
                mission.save_diagnostics()


if __name__ == "__main__":
    main()
