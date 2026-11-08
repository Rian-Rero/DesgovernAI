"""Historico de poses do carro para levar o mapa de cones entre quadros.

O laco de controle registra (t, x, y, th) a cada passo; a thread de visao pede
o deslocamento do carro entre os instantes de captura de dois quadros. A pose
vem do modelo cinematico do Car (w = v/L*tan(st)), confiavel em intervalos
curtos. Fora do intervalo registrado, devolve None (a visao usa fluxo optico).
"""
from collections import deque
import math
import threading

import numpy as np


class PoseHistory:
    def __init__(self, maxlen=600):
        self.poses = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, t, x, y, th):
        values = (t, x, y, th)
        if not all(isinstance(v, (int, float, np.number)) and math.isfinite(v) for v in values):
            return
        with self.lock:
            if self.poses and t <= self.poses[-1][0]:
                return
            self.poses.append(tuple(float(v) for v in values))

    def _snapshot(self):
        with self.lock:
            return list(self.poses)

    def pose_at(self, t, slack=.1):
        """Pose interpolada em t; extrapola ate slack s alem do ultimo registro."""
        poses = self._snapshot()
        if len(poses) < 2 or t < poses[0][0] or t > poses[-1][0] + slack:
            return None
        times = [p[0] for p in poses]
        index = int(np.clip(np.searchsorted(times, t), 1, len(poses)-1))
        a, b = poses[index-1], poses[index]
        ratio = (t - a[0]) / (b[0] - a[0])
        turn = math.atan2(math.sin(b[3]-a[3]), math.cos(b[3]-a[3]))
        return (a[1] + ratio*(b[1]-a[1]), a[2] + ratio*(b[2]-a[2]), a[3] + ratio*turn)

    def delta(self, t0, t1):
        """(frente m, esquerda m, yaw rad) do carro de t0 a t1, no referencial de t0."""
        start, end = self.pose_at(t0), self.pose_at(t1)
        if start is None or end is None:
            return None
        dx, dy = end[0]-start[0], end[1]-start[1]
        c, s = math.cos(start[2]), math.sin(start[2])
        yaw = math.atan2(math.sin(end[2]-start[2]), math.cos(end[2]-start[2]))
        return (c*dx + s*dy, -s*dx + c*dy, yaw)


class SharedPoseHistory(PoseHistory):
    """O mesmo historico em memoria compartilhada entre processos.

    O laco de controle (processo principal) chama append(); o processo de
    visao chama delta()/pose_at(). Um anel de maxlen poses em RawArray, com
    trava entre processos: no ARM do Pi a ordem das escritas nao e garantida
    sem ela. Pode ser passado como argumento de multiprocessing.Process.
    """
    def __init__(self, maxlen=600, context=None):
        import multiprocessing

        context = context or multiprocessing.get_context('spawn')
        self.maxlen = int(maxlen)
        self.buffer = context.RawArray('d', 4*self.maxlen)
        self.count = context.RawValue('q', 0)
        self.lock = context.Lock()

    def append(self, t, x, y, th):
        values = (t, x, y, th)
        if not all(isinstance(v, (int, float, np.number)) and math.isfinite(v) for v in values):
            return
        with self.lock:
            count = self.count.value
            if count and t <= self.buffer[4*((count-1) % self.maxlen)]:
                return
            start = 4*(count % self.maxlen)
            self.buffer[start:start+4] = [float(v) for v in values]
            self.count.value = count+1

    def _snapshot(self):
        with self.lock:
            count = self.count.value
            data = np.frombuffer(self.buffer, dtype=np.float64).reshape(self.maxlen, 4).copy()
        if count <= self.maxlen:
            data = data[:count]
        else:
            data = np.roll(data, -(count % self.maxlen), axis=0)
        return [tuple(row) for row in data.tolist()]
