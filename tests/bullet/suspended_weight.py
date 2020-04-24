""" Re-implementation of opensim tug of war model. """

import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time
from farms_container import Container
from farms_muscle.musculo_skeletal_parameters import MuscleParameters
from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt
import transformations as T


def rendering(render=1):
    """Enable/disable rendering"""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, render)
    # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, render)


def main():
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)   # everything should fall down
    # this slows everything down, but let's be accurate...
    p.setTimeStep(0.001)
    p.setRealTimeSimulation(0)  # we want to be faster than real time :)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1e-3
        # numSolverIterations=100
    )

    rendering(0)

    ########## ADD FLOOR ##########
    plane = p.loadURDF("plane.urdf", [0, 0, 0], globalScaling=1)

    ########## LINKS ##########

    vis_static_block = p.createVisualShape(
        shapeType=p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
        visualFramePosition=[0., 0.0, 0.0])

    col_static_block = p.createCollisionShape(
        shapeType=p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
        collisionFramePosition=[0., 0.0, 0.0])

    vis_moving_block = p.createVisualShape(
        p.GEOM_SPHERE,
        visualFramePosition=[0, 0, 0.],
        visualFrameOrientation=[0, 0, 0, 1],
        radius=0.05)

    col_moving_block = p.createCollisionShape(
        p.GEOM_SPHERE,
        collisionFramePosition=[0, 0, 0.],
        collisionFrameOrientation=[0, 0, 0, 1],
        radius=0.05)

    base_mass = 0.  # : Static
    base_position = [0., 0.0, 1.]
    base_orientation = [1., 0., 0., 0.]

    #: Moving block
    mass = 20.
    position = [0., 0.0, 0.5]
    orientation = p.getQuaternionFromEuler([0., 0., 0.])

    system = p.createMultiBody(
        base_mass, col_static_block, vis_static_block, base_position, base_orientation,
        linkMasses=[mass], linkCollisionShapeIndices=[col_moving_block, ],
        linkVisualShapeIndices=[vis_moving_block, ],
        linkPositions=[position, ], linkOrientations=[orientation],
        linkInertialFramePositions=[position, ],
        linkInertialFrameOrientations=[orientation],
        linkParentIndices=[0], linkJointTypes=[p.JOINT_PRISMATIC],
        linkJointAxis=[[0., 0., 1.]])

    p.changeDynamics(system, 0, lateralFriction=0.0,
                     localInertiaDiagonal=[[0.1333, 0.1333, 0.1333]])

    rendering(1)

    ########## MUSCLE ##########
    container = Container()
    muscles = MusculoSkeletalSystem(
        container,
        '../../farms_muscle/conf/test_suspended_weight.yaml'
    )

    #: Initialize DAE
    container.initialize()

    #: integrator
    muscles.setup_integrator()

    u = container.muscles.activations
    stim1 = u.get_parameter('stim_m1')
    activation1 = p.addUserDebugParameter("Activation-1", 0, 1, 0.5)
    num_joints = p.getNumJoints(system)
    p.setJointMotorControlArray(system,
                                np.arange(num_joints),
                                p.VELOCITY_CONTROL,
                                forces=np.zeros((num_joints, 1)))

    #: RUN
    RUN = True
    TIME = 0.0
    START = time.time()
    while RUN:
        keys = p.getKeyboardEvents()
        if keys.get(113):
            RUN = False
            break
        _act1 = p.readUserDebugParameter(activation1)
        stim1.value = _act1
        muscles.step()
        p.stepSimulation()
        container.update_log()
        TIME += 0.001
        # time.sleep(2.)
    END = time.time()
    print(END-START)

    #: Dump results
    container.dump(OVERWRITE=True)

    #: Read results
    forces = pd.read_hdf("./Results/muscles/forces.h5")
    states = pd.read_hdf("./Results/muscles/states.h5")
    #:
    plt.figure()
    plt.title("Forces")
    plt.plot(forces)
    plt.grid(True)
    plt.figure()
    plt.title("states")
    plt.plot(states)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # profile.py
    import pstats
    import cProfile
    cProfile.runctx("main()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats(10)
    #: Plot results
    # import plot_results
    # plot_results.main('./Results')
