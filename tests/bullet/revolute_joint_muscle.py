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
colBoxId = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[0.05, 0.05, 0.5],
    collisionFramePosition=[0,0,-0.5],
    collisionFrameOrientation=[0,0,0,1],)
visBoxId = p.createVisualShape(
    p.GEOM_BOX,
    visualFramePosition=[0,0,-0.5],
    visualFrameOrientation=[0,0,0,1],
    halfExtents=[0.05, 0.05, 0.5])

mass = 0
visualShapeId = -1


link_Masses = [1]
linkCollisionShapeIndices = [colBoxId]
# linkVisualShapeIndices = [visBoxId]
linkVisualShapeIndices = [-1]
linkPositions = [[0, 0, -1]]
linkOrientations = [[np.pi/2, 0, 0, 1]]
linkInertialFramePositions = [[0, 0, -0.5]]
linkInertialFrameOrientations = [[0, 0, 0, 1]]
indices = [0]
jointTypes = [p.JOINT_REVOLUTE]
axis = [[1, 0, 0]]


for k in range(1):
    basePosition = [0,0,4]
    baseOrientation = [0, 0, 0, 1]
    # if not (k & 2):
    sphereUid = p.createMultiBody(
        mass, colBoxId, visualShapeId, basePosition, baseOrientation,
        linkMasses=link_Masses, linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions, linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkParentIndices=indices, linkJointTypes=jointTypes,
        linkJointAxis=axis)
    p.changeDynamics(sphereUid, -1, spinningFriction=0.0,
                     rollingFriction=0.0, linearDamping=0.0)
    
    # for joint in range(p.getNumJoints(sphereUid)):
        # p.setJointMotorControl2(
        #     sphereUid, joint, p.VELOCITY_CONTROL,force=1, targetVelocity=1)

    p.setJointMotorControlArray(
        sphereUid,
        np.arange(1),
        p.VELOCITY_CONTROL,
        forces=np.zeros(1))


p.getNumJoints(sphereUid)
for i in range(p.getNumJoints(sphereUid)):
    p.getJointInfo(sphereUid, i)


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
for j in range(1000000):
    
    p.setJointMotorControl2(
        sphereUid, 0, p.TORQUE_CONTROL,force=0)
    # p.setJointMotorControl2(
    #     sphereUid, 0,p.VELOCITY_CONTROL,targetVelocity=-1,force=100)
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
