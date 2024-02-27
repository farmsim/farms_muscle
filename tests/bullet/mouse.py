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
plane = p.loadURDF("plane.urdf", [0, 0, -5], globalScaling=1)

########## ADD MOUSE ##########
mouse = p.loadSDF(
    "/home/tatarama/.gazebo/models/mouse_locomotion_bullet/sdf/mouse_locomotion_bullet.sdf")[0]

pylog.debug('Model id : {}'.format(mouse))

########## DEBUG ##########
activation1 = p.addUserDebugParameter("Activation-1", 0, 1, 0.05)
activation2 = p.addUserDebugParameter("Activation-2", 0, 1, 0.5)
rendering(1)

########## MUSCLE ##########
# DAE
muscles = MusculoSkeletalSystem('../../farms_muscle/conf/test_mouse.yaml')

# Initialize DAE
muscles.dae.initialize_dae()


# integrator
muscles.setup_integrator()
u = muscles.dae.u
stim1 = u.get_param('stim_m1')
stim2 = u.get_param('stim_m2')

N = 10000

# Init
for j in range(N):
    keys = p.getKeyboardEvents()
    if keys.get(113):
        break

    # p.setJointMotorControlArray(
    #     mouse, np.arange(num_links), p.POSITION_CONTROL,
    #     targetPositions=np.arange(num_links)*np.sin(2*np.pi*j/1000))
    # p.setJointMotorControlArray(
    #     mouse, np.arange(num_links), p.TORQUE_CONTROL,
    #     forces=np.arange(num_links)*0.)
    # p.setJointMotorControl2(
    #     mouse, 0, p.POSITION_CONTROL,
    #     targetPosition=np.sin(2*np.pi*j/1000*1.)*np.pi)#
    # p.setJointMotorControl2(mouse, 0, p.TORQUE_CONTROL,force=0.)

    _act1 = p.readUserDebugParameter(activation1)
    _act2 = p.readUserDebugParameter(activation2)
    stim1.value = _act1
    stim2.value = _act2
    muscles.step()
    # time.sleep(0.001)
    p.stepSimulation()

musculo_dae = muscles.dae
# Muscle Logging
musculo_x = pd.DataFrame(musculo_dae.x.log)
musculo_x.columns = musculo_dae.x.names
musculo_x.to_hdf('./Results/mouse_musculo_x.h5', 'musculo_x', mode='w')
musculo_xdot = pd.DataFrame(musculo_dae.xdot.log)
musculo_xdot.columns = musculo_dae.xdot.names
musculo_xdot.to_hdf('./Results/mouse_musculo_xdot.h5',
                    'musculo_xdot', mode='w')
musculo_y = pd.DataFrame(musculo_dae.y.log)
musculo_y.columns = musculo_dae.y.names
musculo_y.to_hdf('./Results/mouse_musculo_y.h5', 'musculo_y', mode='w')
musculo_p = pd.DataFrame(musculo_dae.p.log)
musculo_p.columns = musculo_dae.p.names
musculo_p.to_hdf('./Results/mouse_musculo_p.h5', 'musculo_p', mode='w')
musculo_u = pd.DataFrame(musculo_dae.u.log)
musculo_u.columns = musculo_dae.u.names
musculo_u.to_hdf('./Results/mouse_musculo_u.h5', 'musculo_u', mode='w')

# Plot results
import plot_results
# plot_results.main('./Results')
