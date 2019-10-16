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
    shapeType=p.GEOM_BOX, halfExtents= [0.05, 0.05, 0.05],
    visualFramePosition=[0., 0.0, 0.0])

col_static_block = p.createCollisionShape(
    shapeType=p.GEOM_BOX, halfExtents= [0.05, 0.05, 0.05],
    collisionFramePosition=[0., 0.0, 0.0])

vis_moving_block = p.createVisualShape(
    p.GEOM_SPHERE,
    visualFramePosition=[0,0,0.],
    visualFrameOrientation=[0,0,0,1],
    radius=0.05)

col_moving_block = p.createCollisionShape(
    p.GEOM_SPHERE,
    collisionFramePosition=[0,0,0.],
    collisionFrameOrientation=[0,0,0,1],
    radius=0.05)

base_mass = 0. #: Static
base_position = [0., 0.5, 1.]
base_orientation = [1., 0., 0., 0.]

#: Moving block
mass = 20.
position = [0., 0.0, 0.2]
orientation = p.getQuaternionFromEuler([0., 0., 0.])

system = p.createMultiBody(
    base_mass, col_static_block, vis_static_block, base_position, base_orientation,
    linkMasses=[mass], linkCollisionShapeIndices=[col_moving_block,],
    linkVisualShapeIndices=[vis_moving_block,],
    linkPositions=[position,], linkOrientations=[orientation],
    linkInertialFramePositions=[position,],
    linkInertialFrameOrientations=[orientation],
    linkParentIndices=[0], linkJointTypes=[p.JOINT_PRISMATIC],
    linkJointAxis=[[0., 0., 1.]])

p.changeDynamics(system, 0, lateralFriction=0.0,
                 localInertiaDiagonal=[[0.1333, 0.1333, 0.1333]])

rendering(1)

########## MUSCLE ##########
container = Container()
muscles = MusculoSkeletalSystem('../../farms_muscle/conf/test_suspended_weight.yaml')

#: Initialize DAE
container.initialize()

#: integrator
muscles.setup_integrator()

u = container.muscles.activations
stim1 = u.get_parameter('stim_m1')
activation1 = p.addUserDebugParameter("Activation-1", 0, 1, 0.05)
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
        RUN=False
        break
    _act1 = p.readUserDebugParameter(activation1)
    stim1.value = _act1
    muscles.step()
    p.stepSimulation()
    TIME += 0.001
    # time.sleep(2.)
END = time.time()
print(END-START)

musculo = container.muscles
#: Muscle Logging
musculo_x = pd.DataFrame(musculo.states.log)
musculo_x.columns = musculo.states.names
musculo_x.to_hdf('./Results/musculo_x.h5', 'musculo_x', mode='w')
musculo_xdot = pd.DataFrame(musculo.dstates.log)
musculo_xdot.columns = musculo.dstates.names
musculo_xdot.to_hdf('./Results/musculo_xdot.h5', 'musculo_xdot', mode='w')
musculo_y = pd.DataFrame(musculo.outputs.log)
musculo_y.columns = musculo.outputs.names
musculo_y.to_hdf('./Results/musculo_y.h5', 'musculo_y', mode='w')
musculo_p = pd.DataFrame(musculo.parameters.log)
musculo_p.columns = musculo.parameters.names
musculo_p.to_hdf('./Results/musculo_p.h5', 'musculo_p', mode='w')
musculo_u = pd.DataFrame(musculo.activations.log)
musculo_u.columns = musculo.activations.names
musculo_u.to_hdf('./Results/musculo_u.h5', 'musculo_u', mode='w')
musculo_f = pd.DataFrame(musculo.forces.log)
musculo_f.columns = musculo.forces.names
musculo_f.to_hdf('./Results/musculo_f.h5', 'musculo_f', mode='w')

#: Plot results
import plot_results
plot_results.main('./Results')
