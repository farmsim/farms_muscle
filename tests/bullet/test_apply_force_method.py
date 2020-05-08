""" Test apply force method. """
import pybullet as p
import time
import numpy as np

#: Connect to GUI
p.connect(p.GUI)

#: Create a plane
plane = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, plane)

#: Create a multi-body with links and joints
sphereRadius = 0.25
colBoxId = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[sphereRadius, sphereRadius*4, sphereRadius],
  collisionFramePosition=[0, sphereRadius*4, 0]

)

mass = 1
visualShapeId = -1

link_Masses = []
linkCollisionShapeIndices = []
linkVisualShapeIndices = []
linkPositions = []
linkOrientations = []
linkInertialFramePositions = []
linkInertialFrameOrientations = []
indices = []
jointTypes = []
axis = []

for i in range(2):
  link_Masses.append(1)
  linkCollisionShapeIndices.append(colBoxId)
  linkVisualShapeIndices.append(-1)
  linkPositions.append([0, sphereRadius * 2.0*4.0, 0])
  linkOrientations.append([0, 0, 0, 1])
  linkInertialFramePositions.append([0, 4*sphereRadius, 0])
  linkInertialFrameOrientations.append([0, 0, 0, 1])
  indices.append(i)
  jointTypes.append(p.JOINT_REVOLUTE)
  axis.append([1, 0, 0])

basePosition = [0, 0, sphereRadius]
baseOrientation = [0, 0, 0, 1]
model_id = p.createMultiBody(mass,
                              colBoxId,
                              visualShapeId,
                              basePosition,
                              baseOrientation,
                             baseInertialFramePosition=[0, 4*sphereRadius, 0],
                              linkMasses=link_Masses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=indices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis,
                              )

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)

dt = 1. / 240.

for i in range(p.getNumJoints(model_id)):  
  p.setJointMotorControlArray(
    model_id,
    np.arange(p.getNumJoints(model_id)),
    p.VELOCITY_CONTROL,
    targetVelocities=np.arange(p.getNumJoints(model_id))*0.0,
    forces=np.arange(p.getNumJoints(model_id))*0.0
  )

debug_force = p.addUserDebugParameter(
  "force", -100, 100, 0
)
debug_line = p.addUserDebugLine(
  lineFromXYZ=[0., 0., 0],
  lineToXYZ=[0., 0., 0],
  lineWidth=4,
  lineColorRGB=[0, 0, 1],
)

link_id = -1
for j in range(-1, p.getNumJoints(model_id)):
  print(
    'Joint {} : {}'.format(j, p.getDynamicsInfo(model_id, j)[3])
  )
while (1):  
  f_pos = np.asarray([0.0, 0.0, 0.0])
  f_vec = np.asarray([0., 0., p.readUserDebugParameter(debug_force)],)
  p.applyExternalForce(
      model_id,
      link_id,
      f_vec,
      f_pos,
      p.LINK_FRAME
  )
  p.addUserDebugLine(
      lineFromXYZ=f_pos,
      lineToXYZ=f_pos+(f_vec/np.linalg.norm(f_vec)),
      lineWidth=4,
      lineColorRGB=[0, 1, 0],
      parentObjectUniqueId=model_id,
      parentLinkIndex=link_id,
      replaceItemUniqueId=debug_line
  )
  
  p.stepSimulation()
  time.sleep(dt)
