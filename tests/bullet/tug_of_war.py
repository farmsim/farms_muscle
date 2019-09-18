""" Re-implementation of opensim tug of war model. """

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

# visBoxId = p.createVisualShape(
#     p.GEOM_BOX,
#     visualFramePosition=[0,0,-0.5],
#     visualFrameOrientation=[0,0,0,1],
#     halfExtents=[0.05, 0.05, 0.5])

length = 0.5
mass = 0.
link_mass = 1
visualShapeId = -1

num_links = 4
basePosition = [0, 0,2*length+num_links]
baseOrientation = [0, 0, 0, 1]
baseColId = p.createCollisionShape(p.GEOM_CAPSULE,
                                   height=length,
                                   radius=length*0.05,
                                   collisionFramePosition=[0, 0, -length*0.5])
link_Masses = [link_mass for j in range(num_links)]
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

#: RUN
for j in range(N):
    keys = p.getKeyboardEvents()
    if keys.get(113):
        break

    time.sleep(0.001)
    p.stepSimulation()
