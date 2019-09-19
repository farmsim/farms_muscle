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

vis_static_block = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX],
                                            halfExtents=[
                                                [0.5, 0.1, 0.1], [0.5, 0.1, 0.1]],
                                            visualFramePositions=[
                                                [0., -0.35, 0.], [0., 0.35, 0.]])

col_static_block = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX],
                                            halfExtents=[
                                                [0.5, 0.1, 0.1], [0.5, 0.1, 0.1]],
                                            collisionFramePositions=[
                                                [0., -0.35, 0.], [0., 0.35, 0.]])

vis_moving_block = p.createVisualShape(
    p.GEOM_BOX,
    visualFramePosition=[0,0,0.],
    visualFrameOrientation=[0,0,0,1],
    halfExtents=[0.1, 0.1, 0.1])

col_moving_block = p.createCollisionShape(
    p.GEOM_BOX,
    collisionFramePosition=[0,0,0.],
    collisionFrameOrientation=[0,0,0,1],
    halfExtents=[0.1, 0.1, 0.1])

base_mass = 0. #: Static
base_position = [0., 0., 0.1]
base_orientation = [1., 0., 0., 0.]

#: Moving block
mass = 20.
position = [0., 0., 0.]
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

#: RUN
RUN = True
while RUN:
    keys = p.getKeyboardEvents()
    if keys.get(113):
        RUN=False
        break
    time.sleep(0.001)
    p.stepSimulation()
