local sim = require('sim')

function sysCall_init()
end

function fvaCarGetState(robot)
    local linear, angular = sim.getObjectVelocity(robot)
    return {
        time = sim.getSimulationTime(),
        position = sim.getObjectPosition(robot, sim.handle_world),
        quaternion = sim.getObjectQuaternion(robot, sim.handle_world),
        matrix = sim.getObjectMatrix(robot, sim.handle_world),
        linear = linear,
        angular = angular,
    }
end

function fvaCarSetMotors(left, right, velocity, force)
    for _, motor in ipairs({left, right}) do
        sim.setJointTargetVelocity(motor, velocity)
        sim.setJointForce(motor, force)
    end
end

function fvaCarSetSteering(left, right, angleLeft, angleRight)
    sim.setJointTargetPosition(left, angleLeft)
    sim.setJointTargetPosition(right, angleRight)
end
