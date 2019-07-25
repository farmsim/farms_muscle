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
import transformations as T

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

# visBoxId = p.createVisualShape(
#     p.GEOM_BOX,
#     visualFramePosition=[0,0,-0.5],
#     visualFrameOrientation=[0,0,0,1],
#     halfExtents=[0.05, 0.05, 0.5])

length = 0.5
mass = 0
visualShapeId = -1

num_links = 1
basePosition = [0, 0,2*length+num_links]
baseOrientation = [0, 0, 0, 1]
baseColId = p.createCollisionShape(p.GEOM_CAPSULE,
                                   height=length,
                                   radius=length*0.05,
                                   collisionFramePosition=[0, 0, -length*0.5])
link_Masses = [2 for j in range(num_links)]
linkCollisionShapeIndices = [p.createCollisionShape(p.GEOM_CAPSULE,
                                                    height=length,
                                                    radius=length*0.05,
                                                    collisionFramePosition=[
                                                        0, 0, -length*0.5]
                                                    ) for j in range(num_links)]
# linkVisualShapeIndices = [visBoxId]
linkVisualShapeIndices = [-1 for j in range(num_links)]
linkPositions = [[0, 0, -length] for j in range(num_links)]
linkOrientations = [[0.0, 0, 0, 1] for j in range(num_links)]
linkInertialFramePositions = [
    [0, 0, -length*0.5*(j+1)] for j in range(num_links)]
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
activation1 = p.addUserDebugParameter("Activation-1", 0, 1, 0.5)
activation2 = p.addUserDebugParameter("Activation-2", 0, 1, 1.)
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

pbase = p.getBasePositionAndOrientation(chainUid)

plink = p.getLinkState(chainUid, 0)

m1a1 = np.array([0.0, 0.025, -0.4, 1.])
m1a2 = np.array([0.0, 0.025, -0.5, 1.])
m1a3 = np.array([0.0, 0.025, -0.15, 1.])

m2a1 = np.array([0.0, -0.025, -0.4, 1.])
m2a2 = np.array([0.0, -0.025, -0.5, 1.])
m2a3 = np.array([0.0, -0.025, -0.15, 1.])


# m1attach1 = p.addUserDebugLine(
#     lineFromXYZ=m1a1[:3],
#     lineToXYZ=(m1a1+np.array([0,0,-0.05,0]))[:3],
#     lineColorRGB=[1, 1, 0],
#     lineWidth=2,
#     lifeTime=0,
#     parentObjectUniqueId=chainUid,
#     parentLinkIndex=-1)

# m1attach2 = p.addUserDebugLine(
#     lineFromXYZ=m1a2[:3],
#     lineToXYZ=m1a3[:3],
#     lineColorRGB=[1, 0, 1],
#     lineWidth=2,
#     lifeTime=0,
#     parentObjectUniqueId=chainUid,
#     parentLinkIndex=0)


f1line1 = p.addUserDebugLine(
        lineToXYZ=[0,0,0],
        lineFromXYZ=[0,0,0],
        lineColorRGB=[1, 0, 0],
        lineWidth=4,
        lifeTime=0)
f1line2 = p.addUserDebugLine(
        lineToXYZ=[0,0,0],
        lineFromXYZ=[0,0,0],
        lineColorRGB=[1, 0, 0],
        lineWidth=4,
        lifeTime=0)

f2line1 = p.addUserDebugLine(
        lineToXYZ=[0,0,0],
        lineFromXYZ=[0,0,0],
        lineColorRGB=[1, 0, 0],
        lineWidth=4,
        lifeTime=0)
f2line2 = p.addUserDebugLine(
        lineToXYZ=[0,0,0],
        lineFromXYZ=[0,0,0],
        lineColorRGB=[1, 0, 0],
        lineWidth=4,
        lifeTime=0)

def distance_bw_points(p1, p2):
    """ Compute distance between two points. """
    return np.linalg.norm(np.array(p1)-np.array(p2))

#: integrator
muscles.setup_integrator(x0)
u = muscles.dae.u
force1 = muscles.dae.y.get_param('tendon_force_m1')
force2 = muscles.dae.y.get_param('tendon_force_m2')
N = 2000
length1 = np.zeros((N,1))
length2 = np.zeros((N,1))
jangle = np.zeros((N,1))

p.resetJointState(chainUid, 0, targetValue=0.)

#: Init
for j in range(N):
    keys = p.getKeyboardEvents()
    if keys.get(113):
        break
    # p.setJointMotorControlArray(
    #     chainUid, np.arange(num_links), p.POSITION_CONTROL,
    #     targetPositions=np.arange(num_links)*np.sin(2*np.pi*j/1000))
    # p.setJointMotorControlArray(
    #     chainUid, np.arange(num_links), p.TORQUE_CONTROL,
    #     forces=np.arange(num_links)*0.)
    # p.setJointMotorControl2(
    #     chainUid, 0, p.POSITION_CONTROL,
    #     targetPosition=np.sin(2*np.pi*j/1000*1.))# *np.pi/2
    p.setJointMotorControl2(chainUid, 0, p.TORQUE_CONTROL,force=0.)
    
    ls = p.getLinkState(chainUid, 0)

    #: Build Homogeneous Matrix
    base_trans = T.compose_matrix(angles=p.getEulerFromQuaternion(pbase[1]),
                               translate=pbase[0])
    l1_trans = T.compose_matrix(angles=p.getEulerFromQuaternion(ls[5]),
                               translate=ls[4])

    p11 = np.dot(base_trans, m1a1)[:3]
    p12 = np.dot(base_trans, m1a2)[:3]
    p13 = np.dot(l1_trans, m1a3)[:3]
    dist11 = distance_bw_points(p11, p12)
    dist12 = distance_bw_points(p12, p13)
    _length1 = dist11+dist12

    p21 = np.dot(base_trans, m2a1)[:3]
    p22 = np.dot(base_trans, m2a2)[:3]
    p23 = np.dot(l1_trans, m2a3)[:3]
    dist21 = distance_bw_points(p21, p22)
    dist22 = distance_bw_points(p22, p23)
    _length2 = dist21+dist22
    
    length1[j] = _length1
    length2[j] = _length2
    jangle[j] = p.getJointState(chainUid, 0)[0]
    _act1 = p.readUserDebugParameter(activation1)
    _act2 = p.readUserDebugParameter(activation2)
    u.values = np.array([_length1, _act1, _length2, _act2])
    muscles.step()
    _force1 = force1.value
    _force2 = force2.value
    _f_vec1 = T.unit_vector(p13-p12)*_force1
    _f_vec2 = T.unit_vector(p23-p22)*_force2
    print(_f_vec1, _f_vec2)
    p.applyExternalForce(chainUid, 0, _f_vec1, p13, flags=p.WORLD_FRAME)
    p.applyExternalForce(chainUid, 0, _f_vec2, p23, flags=p.WORLD_FRAME)
    # _f_vec1 = T.unit_vector(p11-p12)*_force1
    # _f_vec2 = T.unit_vector(p21-p22)*_force2
    # p.applyExternalForce(chainUid, -1, _f_vec1, p12, flags=p.WORLD_FRAME)
    # p.applyExternalForce(chainUid, -1, _f_vec2, p22, flags=p.WORLD_FRAME)

    # if _force1 > 0.:
    #     print(length[j], _force1, _f_vec1, p13)

    # p.addUserDebugLine(
    #     lineFromXYZ=p11,
    #     lineToXYZ=p12,
    #     lineColorRGB=[1, 0, 0],
    #     lineWidth=4,
    #     replaceItemUniqueId=f1line1)
        
    # p.addUserDebugLine(
    #     lineFromXYZ=p12,
    #     lineToXYZ=p13,
    #     lineColorRGB=[1, 0, 0],
    #     lineWidth=4,
    #     replaceItemUniqueId=f1line2)

    # p.addUserDebugLine(
    #     lineFromXYZ=p21,
    #     lineToXYZ=p22,
    #     lineColorRGB=[1, 1, 0],
    #     lineWidth=4,
    #     replaceItemUniqueId=f2line1)
        
    # p.addUserDebugLine(
    #     lineFromXYZ=p22,
    #     lineToXYZ=p23,
    #     lineColorRGB=[1, 1, 0],
    #     lineWidth=4,
    #     replaceItemUniqueId=f2line2)
    
    # time.sleep(0.001)
    p.stepSimulation()

# plt.figure()
# plt.plot(rot_cart)
# plt.legend(('x', 'y', 'z'))
# plt.grid(True)
# plt.figure()
# plt.plot(np.rad2deg(jangle), length)
# plt.grid(True)
plt.figure()
plt.plot(jangle, length1)
plt.plot(jangle, length2)
plt.legend(('1', '2'))
plt.grid(True)
plt.show()

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
plot_results.main('./Results')
