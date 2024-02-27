""" Test apply force method. """
import pybullet as p
import time
import numpy as np

def convert_local_to_global(body_id, link_id, local_coordinate):
    """Convert local coordinate to global coordinates

    Parameters
    ----------
    body_id : <int>
        Index of the body in pybullet
    link_id : <int>
        Index of the link element in pybullet body
    local_coordinate : <list/tuple>
        Coordinate in the link frame

    Returns
    -------
    global_coordinate : <list/tuple>
        Coordinate in global frame
    """
    if link_id != -1:
        global_coordinate, _ = p.multiplyTransforms(
            *p.getLinkState(body_id, link_id)[4:6],
            local_coordinate, [0, 0, 0, 1]
        )
    else:
        # global_coordinate, _ = p.multiplyTransforms(
        #     *p.getBasePositionAndOrientation(body_id),
        #     local_coordinate, [0, 0, 0, 1]
        # )
        global_coordinate, _ = p.multiplyTransforms(
            *p.getBasePositionAndOrientation(body_id),
            *(p.multiplyTransforms(
                *(p.invertTransform(
                    *p.getDynamicsInfo(body_id, -1)[3:5]
                )),
                local_coordinate, [0, 0, 0, 1]
            ))
        )
    return global_coordinate


def convert_global_to_local(body_id, link_id, global_coordinate):
    """Convert global coordinate to global coordinates

    Parameters
    ----------
    body_id : <int>
        Index of the body in pybullet
    link_id : <int>
        Index of the link element in pybullet body
    global_coordinate : <list/tuple>
        Coordinate in the link frame

    Returns
    -------
    local_coordinate : <list/tuple>
        Coordinate in local link frame
    """
    if link_id != -1:
        local_coordinate, _ = p.multiplyTransforms(
            *p.invertTransform(*p.getLinkState(body_id, link_id)[4:6]),
            global_coordinate, [0, 0, 0, 1]
        )
    else:
        local_coordinate, _ = p.multiplyTransforms(
            *p.invertTransform(*p.getBasePositionAndOrientation(body_id)),
            global_coordinate, [0, 0, 0, 1]
        )
    return local_coordinate


def convert_local_to_inertial(body_id, link_id, local_coordinate):
    """Convert local coordinate to local inertial coordinates

    Parameters
    ----------
    body_id : <int>
        Index of the body in pybullet
    link_id : <int>
        Index of the link element in pybullet body
    local_coordinate : <list/tuple>
        Coordinate in the link/urdf frame

    Returns
    -------
    global_coordinate : <list/tuple>
        Coordinate in local inertial frame
    """
    inertial_coordinate, _ = p.multiplyTransforms(
        *p.invertTransform(*p.getDynamicsInfo(body_id, link_id)[3:5]),
        local_coordinate, [0, 0, 0, 1]
    )
    return inertial_coordinate


def create_floor(pybullet):
    """

    Parameters
    ----------
    pybullet :


    Returns
    -------
    out :

    """
    # Create a plane
    plane = pybullet.createCollisionShape(pybullet.GEOM_PLANE)
    floor = pybullet.createMultiBody(0, plane)
    return floor

def create_model(pybullet, n_links, **kwargs):
    """

    Parameters
    ----------
    pybullet :

    n_links :

    **kwargs :


    Returns
    -------
    out :

    """
    # Create a multi-body with links and joints
    box_dimensions = kwargs.pop('box_dimensions', [0.5, 2.0, 0.5])
    colBoxId = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[d*0.5 for d in box_dimensions],
        collisionFramePosition=[0, box_dimensions[1]*0.5, 0]
    )

    mass = kwargs.pop('mass', 1.)
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

    for i in range(n_links):
      link_Masses.append(mass)
      linkCollisionShapeIndices.append(colBoxId)
      linkVisualShapeIndices.append(-1)
      linkPositions.append([0, box_dimensions[1], 0])
      linkOrientations.append([0, 0, 0, 1])
      linkInertialFramePositions.append([0, box_dimensions[1]*0.5, 0])
      linkInertialFrameOrientations.append([0, 0, 0, 1])
      indices.append(i)
      jointTypes.append(pybullet.JOINT_REVOLUTE)
      axis.append([1, 0, 0])

    basePosition = [0, 0, box_dimensions[2]*0.5]
    baseOrientation = [0, 0, 0, 1]
    model_id = pybullet.createMultiBody(
        mass,
        colBoxId,
        visualShapeId,
        basePosition,
        baseOrientation,
        baseInertialFramePosition=[0, box_dimensions[1]*0.5, 0],
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
    return model_id

def main():
    # Connect to GUI
    p.connect(p.GUI)

    # Create floor
    floor_id = create_floor(p)

    # Create model
    box_dimensions = [0.5, 2.0, 0.5]
    model_id = create_model(p, 2, box_dimensions=box_dimensions)

    # Setup simulation
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)
    dt = 1. / 240.
    # Disable velocity controllers
    for i in range(p.getNumJoints(model_id)):
      p.setJointMotorControlArray(
        model_id,
        np.arange(p.getNumJoints(model_id)),
        p.VELOCITY_CONTROL,
        targetVelocities=np.arange(p.getNumJoints(model_id))*0.0,
        forces=np.arange(p.getNumJoints(model_id))*0.0
      )
    # Debug
    debug_force = p.addUserDebugParameter(
      "local URDF frame force", -100, 100, 0
    )

    debug_line = p.addUserDebugLine(
      lineFromXYZ=[0., 0., 0],
      lineToXYZ=[0., 0., 0],
      lineWidth=4,
      lineColorRGB=[0, 0, 1],
    )

    # Set link ID to apply force on
    link_id = -1
    for j in range(-1, p.getNumJoints(model_id)):
      print(
        'Joint {} : {}'.format(j, p.getDynamicsInfo(model_id, j)[3])
      )

    # Describe the force vector position in local URDF frame
    f_pos = np.asarray([0.0, box_dimensions[1]*0.0, 0.0])

    # apply force in local(0)/global(1)
    apply_force_frame = 1

    # Main loop
    while (1):
      f_vec = np.asarray(
          [0., 0., p.readUserDebugParameter(debug_force)],
      )
      if apply_force_frame == 0:
          p.applyExternalForce(
              model_id,
              link_id,
              convert_local_to_inertial(model_id, link_id, f_vec),
              convert_local_to_inertial(model_id, link_id, f_pos),
              p.LINK_FRAME
        )
          p.addUserDebugLine(
              lineFromXYZ=convert_local_to_inertial(
                  model_id, link_id, f_pos
              ),
              lineToXYZ=convert_local_to_inertial(
                  model_id, link_id, f_pos+(f_vec/np.linalg.norm(f_vec))),
              lineWidth=4,
              lineColorRGB=[0, 1, 0],
              parentObjectUniqueId=model_id,
              parentLinkIndex=link_id,
              replaceItemUniqueId=debug_line
          )
      elif apply_force_frame == 1:
          p.applyExternalForce(
              model_id,
              link_id,
              convert_local_to_global(model_id, link_id, f_vec),
              convert_local_to_global(model_id, link_id, f_pos),
              p.WORLD_FRAME
          )
          p.addUserDebugLine(
              lineFromXYZ=convert_local_to_global(model_id, link_id,
                  f_pos
              ),
              lineToXYZ=convert_local_to_global(model_id, link_id,
                  f_pos+(f_vec/np.linalg.norm(f_vec))
              ),
              lineWidth=4,
              lineColorRGB=[0, 1, 0],
              replaceItemUniqueId=debug_line
          )


      # step simulation
      p.stepSimulation()
      time.sleep(dt)

if __name__ == '__main__':
    main()
