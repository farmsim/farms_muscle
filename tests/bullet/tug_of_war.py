""" Re-implementation of opensim tug of war model. """

import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time
from farms_dae.dae_generator import DaeGenerator
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

vis_static_block = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX],
                                            halfExtents=[
                                                [0.25, 0.05, 0.05], [0.25, 0.05, 0.05]],
                                            visualFramePositions=[
                                                [0., -0.35, 0.], [0., 0.35, 0.]])

col_static_block = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX],
                                            halfExtents=[
                                                [0.25, 0.05, 0.05], [0.25, 0.05, 0.05]],
                                            collisionFramePositions=[
                                                [0., -0.35, 0.], [0., 0.35, 0.]])

vis_moving_block = p.createVisualShape(
    p.GEOM_BOX,
    visualFramePosition=[0,0,0.],
    visualFrameOrientation=[0,0,0,1],
    halfExtents=[0.05, 0.05, 0.05])

col_moving_block = p.createCollisionShape(
    p.GEOM_BOX,
    collisionFramePosition=[0,0,0.],
    collisionFrameOrientation=[0,0,0,1],
    halfExtents=[0.05, 0.05, 0.05])

base_mass = 0. #: Static
base_position = [0., 0., 0.1]
base_orientation = [1., 0., 0., 0.]

#: Moving block
mass = 20.
position = [0., 0.0, 0.]
orientation = p.getQuaternionFromEuler([0., 0., 0.])

system = p.createMultiBody(
    base_mass, col_static_block, vis_static_block, base_position, base_orientation,
    linkMasses=[mass], linkCollisionShapeIndices=[col_moving_block,],
    linkVisualShapeIndices=[vis_moving_block,],
    linkPositions=[position,], linkOrientations=[orientation],
    linkInertialFramePositions=[position,],
    linkInertialFrameOrientations=[orientation],
    linkParentIndices=[0], linkJointTypes=[p.JOINT_PRISMATIC],
    linkJointAxis=[[0., 1., 0.]])

p.changeDynamics(system, 0, lateralFriction=0.0,
                 localInertiaDiagonal=[[0.1333, 0.1333, 0.1333]])

rendering(1)

########## MUSCLE ##########
#: DAE
muscles = MusculoSkeletalSystem('../../farms_muscle/conf/test_tug_of_war.yaml')

#: Initialize DAE
muscles.dae.initialize_dae()

#: integrator
muscles.setup_integrator()

u = muscles.dae.u
stim1 = u.get_param('stim_m1')
stim2 = u.get_param('stim_m2')
activation1 = p.addUserDebugParameter("Activation-1", 0, 1, 0.05)
activation2 = p.addUserDebugParameter("Activation-2", 0, 1, 0.05)

num_joints = p.getNumJoints(system)
p.setJointMotorControlArray(system,
                            np.arange(num_joints),
                            p.VELOCITY_CONTROL,
                            forces=np.zeros((num_joints, 1)))

#: RUN
RUN = True
TIME = 0.0
while RUN:
    keys = p.getKeyboardEvents()
    if keys.get(113):
        RUN=False
        break
    _act1 = p.readUserDebugParameter(activation1)
    _act2 = p.readUserDebugParameter(activation2)
    stim1.value = _act1
    stim2.value = _act2
    muscles.step()
    p.stepSimulation()
    TIME += 0.001

print(TIME)

musculo_dae = muscles.dae
#: Muscle Logging
musculo_x = pd.DataFrame(musculo_dae.x.log)
musculo_x.columns = musculo_dae.x.names
musculo_x.to_hdf('./Results/musculo_x.h5', 'musculo_x', mode='w')
musculo_xdot = pd.DataFrame(musculo_dae.xdot.log)
musculo_xdot.columns = musculo_dae.xdot.names
musculo_xdot.to_hdf('./Results/musculo_xdot.h5', 'musculo_xdot', mode='w')
musculo_y = pd.DataFrame(musculo_dae.y.log)
musculo_y.columns = musculo_dae.y.names
musculo_y.to_hdf('./Results/musculo_y.h5', 'musculo_y', mode='w')
musculo_p = pd.DataFrame(musculo_dae.p.log)
musculo_p.columns = musculo_dae.p.names
musculo_p.to_hdf('./Results/musculo_p.h5', 'musculo_p', mode='w')
musculo_u = pd.DataFrame(musculo_dae.u.log)
musculo_u.columns = musculo_dae.u.names
musculo_u.to_hdf('./Results/musculo_u.h5', 'musculo_u', mode='w')

#: Plot results
import plot_results
plot_results.main('./Results')
