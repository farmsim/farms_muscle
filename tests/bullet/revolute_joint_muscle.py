""" Test file to add a simple revolute joint muscle in bullet. """

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
import matplotlib.pyplot as plt
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
sphereRadius = 0.05

# visBoxId = p.createVisualShape(
#     p.GEOM_BOX,
#     visualFramePosition=[0,0,-0.5],
#     visualFrameOrientation=[0,0,0,1],
#     halfExtents=[0.05, 0.05, 0.5])

mass = 0
visualShapeId = -1

num_links = 1
basePosition = [0,0,4+num_links]
baseOrientation = [0, 0, 0, 1]
baseColId = p.createCollisionShape(p.GEOM_CAPSULE,
                       height=1.,
                       radius=0.05,
                       collisionFramePosition=[0,0,-0.5])
link_Masses = [1 for j in range(num_links)]
linkCollisionShapeIndices = [p.createCollisionShape(p.GEOM_CAPSULE,
                                                    height=1.,
                                                    radius=0.05,
                                                    collisionFramePosition=[0,0,-0.5]
) for j in range(num_links)]
# linkVisualShapeIndices = [visBoxId]
linkVisualShapeIndices = [-1 for j in range(num_links)]
linkPositions = [[0, 0, -1] for j in range(num_links)]
linkOrientations = [[0, 0, 0, 1] for j in range(num_links)]
linkInertialFramePositions = [[0, 0, -0.5*(j+1)] for j in range(num_links)]
linkInertialFrameOrientations = [[0, 0, 0, 1] for j in range(num_links)]
indices = [j for j in range(num_links)]
jointTypes = [p.JOINT_REVOLUTE for j in range(num_links)]
axis = [[1, 0, 0] for j in range(num_links)]

chainUid = p.createMultiBody(
    mass, baseColId, visualShapeId, basePosition, baseOrientation,
    linkMasses=link_Masses, linkCollisionShapeIndices=linkCollisionShapeIndices,
    linkVisualShapeIndices=linkVisualShapeIndices,
    linkPositions=linkPositions, linkOrientations=linkOrientations,
    linkInertialFramePositions=linkInertialFramePositions,
    linkInertialFrameOrientations=linkInertialFrameOrientations,
    linkParentIndices=indices, linkJointTypes=jointTypes,
    linkJointAxis=axis)
p.changeDynamics(chainUid, -1, spinningFriction=0.0,
                 rollingFriction=0.0, linearDamping=0.0)
    
# for joint in range(p.getNumJoints(chainUid)):
    # p.setJointMotorControl2(
#     chainUid, joint, p.VELOCITY_CONTROL,force=1, targetVelocity=1)

p.setJointMotorControlArray(
    chainUid,
    np.arange(num_links),
    p.VELOCITY_CONTROL,
    forces=np.zeros(num_links))


p.getNumJoints(chainUid)
for i in range(p.getNumJoints(chainUid)):
    p.getJointInfo(chainUid, i)


########## DEBUG ##########
activation = p.addUserDebugParameter("Activation", 0, 1, 0.75)
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
rendering(1)

########## MUSCLE ##########
#: DAE
muscles = MusculoSkeletalSystem('../../farms_muscle/conf/muscles.yaml')

#: Integrate the network
muscle_act = {'m1': 0.75}

#: Initialize DAE
muscles.dae.initialize_dae()

x0 = np.array([0.11, 0.0])
# (pos, _) = p.getBasePositionAndOrientation(boxUid)
# mline = p.addUserDebugLine(
#     lineFromXYZ=[0, 0, 4],
#     lineToXYZ=pos,
#     lineColorRGB=[1, 0, 0],
#     lineWidth=2,
#     lifeTime=0)


#: integrator
muscles.setup_integrator(x0)
u = muscles.dae.u
force = muscles.dae.y.get_param('tendon_force_m1')
N = 10000
pos_world = np.zeros((N, 3))
pos_cart = np.zeros((N, 3))
pos_inert = np.zeros((N, 3))
for j in range(N):    
    p.setJointMotorControlArray(
        chainUid, np.arange(num_links), p.TORQUE_CONTROL,
        forces=np.arange(num_links))
    ls = p.getLinkState(chainUid, 0)

    # print(ls[4])
    # print(ls[0])
    pos_cart[j] = np.array(ls[0])
    pos_world[j] = np.array(ls[4])
    pos_inert[j] = np.array(ls[2])
    # p.setJointMotorControl2(
    #     chainUid, 0,p.VELOCITY_CONTROL,targetVelocity=-1,force=100)
    keys = p.getKeyboardEvents()
    if keys.get(113):
        break
    # (pos, _) = p.getBasePositionAndOrientation(boxUid)
    # length = 0.4-pos[2]*0.1
    # _act = p.readUserDebugParameter(activation)
    # _act = np.abs(np.sin(2*np.p1*0.5*j/1000))
    # u.values = np.array([length, _act])
    # muscles.step()
    # _force = force.value
    # p.applyExternalForce(boxUid, -1, [0, 0, _force], [0, 0, 0], 1)
    # if _force > 0.:
    #     print(length, _force)
    # p.addUserDebugLine(
    #     lineToXYZ=pos,
    #     lineFromXYZ=[0, 0, 4],
    #     lineColorRGB=[1, 0, 0],
    #     replaceItemUniqueId=mline)
    time.sleep(0.001)
    p.stepSimulation()

# plt.figure()
# plt.plot(pos_world)
# plt.legend(('x', 'y', 'z'))
# plt.grid(True)
# plt.figure()
# plt.plot(pos_cart)
# plt.legend(('x', 'y', 'z'))
# plt.grid(True)
# plt.figure()
# plt.plot(pos_inert)
# plt.legend(('x', 'y', 'z'))
# plt.grid(True)
# plt.show()

# musculo_dae = muscles.dae
# #: Muscle Logging
# musculo_x = pd.DataFrame(musculo_dae.x.log)
# musculo_x.columns = musculo_dae.x.names
# musculo_x.to_hdf('./Results/musculo_x.h5', 'musculo_x', mode='w')
# musculo_xdot = pd.DataFrame(musculo_dae.xdot.log)
# musculo_xdot.columns = musculo_dae.xdot.names
# musculo_xdot.to_hdf('./Results/musculo_xdot.h5',
#                     'musculo_xdot', mode='w')
# musculo_y = pd.DataFrame(musculo_dae.y.log)
# musculo_y.columns = musculo_dae.y.names
# musculo_y.to_hdf('./Results/musculo_y.h5', 'musculo_y', mode='w')
# musculo_p = pd.DataFrame(musculo_dae.p.log)
# musculo_p.columns = musculo_dae.p.names
# musculo_p.to_hdf('./Results/musculo_p.h5', 'musculo_p', mode='w')
# musculo_u = pd.DataFrame(musculo_dae.u.log)
# musculo_u.columns = musculo_dae.u.names
# musculo_u.to_hdf('./Results/musculo_u.h5', 'musculo_u', mode='w')

# #: Plot results
# import plot_results
# plot_results.main('./Results')
