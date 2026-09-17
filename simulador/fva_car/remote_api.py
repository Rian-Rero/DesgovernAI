"""Agrupa as leituras e escritas do carro para reduzir chamadas remotas."""

from pathlib import Path


class CarRemoteAPI:
    def __init__(self, sim, robot):
        self.sim = sim
        code = Path(__file__).with_suffix('.lua').read_text(encoding='utf-8')
        self.script = sim.createScript(sim.scripttype_simulation, code)
        sim.setObjectAlias(self.script, 'fva_remote_api')
        sim.setObjectParent(self.script, robot, True)

    def close(self):
        self.sim.removeObjects([self.script])

    def get_state(self, robot):
        return self.sim.callScriptFunction('fvaCarGetState', self.script, robot)

    def set_motors(self, left, right, velocity, force):
        self.sim.callScriptFunction('fvaCarSetMotors', self.script, left, right, velocity, force)

    def set_steering(self, left, right, angle_left, angle_right):
        self.sim.callScriptFunction('fvaCarSetSteering', self.script,
                                    left, right, angle_left, angle_right)
