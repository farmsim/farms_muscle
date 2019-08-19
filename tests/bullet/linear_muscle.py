""" Test file to add a simple linear muscle in bullet. """

from IPython import embed
import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time
from farms_dae_generator.dae_generator import DaeGenerator
from farms_muscle.musculo_skeletal_parameters import MuscleParameters
from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
import farms_pylog as pylog
import numpy as np
p.connect(p.GUI)
# without GUI: p.connect(p.DIRECT)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())


def rendering(render=1):
    """Enable/disable rendering"""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, render)
    # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, render)


p.setGravity(0, 0, -9.81)   # everything should fall down
# this slows everything down, but let's be accurate...
p.setTimeStep(0.0001)
p.setRealTimeSimulation(0)  # we want to be faster than real time :)
p.setPhysicsEngineParameter(
    fixedTimeStep=1e-4
    # numSolverIterations=100
)

rendering(0)

#: Add floor
plane = p.loadURDF("plane.urdf", [0, 0, 0], globalScaling=1)


colBoxId = p.createCollisionShape(
    p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])

mass = 15
visualShapeId = -1

boxUid = p.createMultiBody(mass, colBoxId, visualShapeId, [
                           0, 0, 2], useMaximalCoordinates=True)
p.changeDynamics(boxUid, -1, spinningFriction=0.001,
                 rollingFriction=0.001, linearDamping=0.0)

########## DEBUG ##########
activation = p.addUserDebugParameter("Activation", 0, 1, 0.75)
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
rendering(1)

########## MUSCLE ##########
#: DAE
muscles = MusculoSkeletalSystem('../../farms_muscle/conf/linear_muscle.yaml')

#: Integrate the network
muscle_act = {'m1': 0.75}

#: Initialize DAE
muscles.dae.initialize_dae()

x0 = np.array([0.11, 0.0])
(pos, _) = p.getBasePositionAndOrientation(boxUid)
mline = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 4],
    lineToXYZ=pos,
    lineColorRGB=[1, 0, 0],
    lineWidth=4,
    # parentObjectUniqueId=boxUid,
    # parentLinkIndex=-1,
    lifeTime=0)


#: integrator
muscles.setup_integrator(x0)
u = muscles.dae.u
force = muscles.dae.y.get_param('tendon_force_m1')
for j in range(100000):
    keys = p.getKeyboardEvents()
    if keys.get(113):
        break
    (pos, _) = p.getBasePositionAndOrientation(boxUid)
    length = 0.4-pos[2]*0.1
    _act = p.readUserDebugParameter(activation)
    # _act = np.abs(np.sin(2*np.pi*0.5*j/1000))
    u.values = np.array([length, _act])
    muscles.step()
    _force = force.value
    p.applyExternalForce(boxUid, -1, [0, 0, _force], [0, 0, 0], 1)
    # if _force > 0.:
    #     print(length, _force)
    # p.addUserDebugLine(
    #     lineFromXYZ=[0, 1, 0],
    #     lineToXYZ=pos,
    #     lineColorRGB=[1, 0, 0],
    #     lineWidth=2,
    #     parentObjectUniqueId=boxUid,
    #     parentLinkIndex=-1,
    #     replaceItemUniqueId=mline)
    # time.sleep(0.001)
    p.stepSimulation()

musculo_dae = muscles.dae
#: Muscle Logging
musculo_x = pd.DataFrame(musculo_dae.x.log)
musculo_x.columns = musculo_dae.x.names
musculo_x.to_hdf('./Results/musculo_x.h5', 'musculo_x', mode='w')
musculo_xdot = pd.DataFrame(musculo_dae.xdot.log)
musculo_xdot.columns = musculo_dae.xdot.names
musculo_xdot.to_hdf('./Results/musculo_xdot.h5',
                    'musculo_xdot', mode='w')
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
# plot_results.main('./Results')

