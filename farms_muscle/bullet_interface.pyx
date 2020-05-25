# cython: cdivision=True
# cython: infer_types=True
# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

""" Interface between bullet physics engine and muscle model. """

from libc.stdio cimport printf
from cython.view cimport array as cvarray
import farms_pylog as pylog
import transformations as T
import numpy as np
from libc.math cimport pow as cpow
from numpy cimport NPY_FLOAT
import pybullet as p
from physics_interface cimport PhysicsInterface
cimport numpy as np
cimport cython

#: Utils for coordinate convertion


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
            *p.getLinkState(
                body_id, link_id, computeForwardKinematics=True
            )[4:6],
            local_coordinate, [0, 0, 0, 1]
        )
    else:
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
            *p.invertTransform(*p.getLinkState(
                body_id, link_id, computeForwardKinematics=True
            )[4:6]),
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
    inertial_coordinate : <list/tuple>
        Coordinate in local inertial frame
    """
    inertial_coordinate, _ = p.multiplyTransforms(
        *p.invertTransform(*p.getDynamicsInfo(body_id, link_id)[3:5]),
        local_coordinate, [0, 0, 0, 1]
    )
    return inertial_coordinate


cdef class BulletInterface(PhysicsInterface):
    """Interface between bullet physics engine and muscle model."""

    def __init__(
            self, model_id, lmtu, force, stim, waypoints,
            visualization=True, debug_visualization=True
    ):
        super(BulletInterface, self).__init__(lmtu, force, stim, 'BULLET')

        #: Initialization
        self.model_id = model_id

        self.n_attachments = len(waypoints)

        self.global_waypoints = np.ndarray(
            (self.n_attachments,), dtype=('(3,)d')
        )

        self.local_waypoints = np.ndarray(
            (self.n_attachments,),
            dtype=[
                ('link_id', 'i'), ('point', '(3,)d')
            ]
        )

        connection_mode = p.getConnectionInfo(0)['connectionMethod']
        self.visualization = visualization and (connection_mode == 1)

        self.debug_muscle_line_ids = np.ndarray(
            (self.n_attachments,), dtype=('I'))

        self.debug_visualization = debug_visualization

        self.debug_force_ids = np.ndarray(
            (2*self.n_attachments-2,), dtype=('I')
        )

        #: Add the waypoints
        self.initialize_waypoints(waypoints)

        #: Visualization
        if self.visualization:
            self.initialize_debug_muscle_line_visualization()
        if self.debug_visualization:
            self.initialize_debug_muscle_force_visualization()

    #################### PY-FUNCTIONS ####################
    def initialize_waypoints(self, waypoints):
        """ Initialize waypoints
        Parameters
        ----------
        waypoints: <list>
            List containing link ids and attachment points
        """
        link_name_to_index = {
            p.getJointInfo(self.model_id, index)[12].decode('UTF-8'): index
            for index in range(p.getNumJoints(self.model_id))
        }
        #: Add base link
        link_name_to_index[
            p.getBodyInfo(self.model_id)[0].decode('UTF-8')
        ] = -1

        for j, attachment in enumerate(waypoints):
            link_index = link_name_to_index[attachment[0]['link']]
            #: local
            self.local_waypoints[j][0] = link_index
            self.local_waypoints[j][1][0] = attachment[1]['point'][0]
            self.local_waypoints[j][1][1] = attachment[1]['point'][1]
            self.local_waypoints[j][1][2] = attachment[1]['point'][2]
            #: global
            self.global_waypoints[j][:] = attachment[1]['point'][:]

    def initialize_debug_muscle_line_visualization(self):
        """ Initialization of debug lines in pybullet for muscle line
        visualization.

        Parameters
        ----------
        None
        """
        for count in range(self.n_attachments-1):
            self.debug_muscle_line_ids[count] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, 0],
                lineColorRGB=[1, 0, 0],
                lineWidth=4,
                lifeTime=0
            )

    def initialize_debug_muscle_force_visualization(self):
        """ Initialization of debug lines in pybullet for muscle force
        visualization.

        Parameters
        ----------
        None
        """
        #: Debug : Show forces
        for count in range(2*self.n_attachments-2):
            self.debug_force_ids[count] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, 0],
                lineColorRGB=[0, 1, 0],
                lineWidth=4,
                lifeTime=0
            )

    def update_local_points_to_global(self):
        """Update the points in local coordinate to world. """
        cdef unsigned int p
        for p in range(self.n_attachments):
            self.global_waypoints[p] = convert_local_to_global(
                self.model_id,
                self.local_waypoints[p][0],
                self.local_waypoints[p][1][:]
            )

    def update_muscle_length(self):
        self.c_compute_muscle_length()

    #################### C-FUNCTIONS ####################
    cdef void c_compute_muscle_length(self):
        """ Compute the muscle length based on the physics simulator. """
        #: FUCK : This needs to be exposed to outside
        self.update_local_points_to_global()

        #: Compute the length
        cdef double length = 0.0
        cdef unsigned int j
        for j in range(self.n_attachments-1):
            length += c_distance_between_points(
                self.global_waypoints[j], self.global_waypoints[j+1]
            )
        #: Update the muscle length
        self.lmtu.c_set_value(length)

    cdef void c_apply_muscle_forces(self):
        """ Apply the forces generated by the muscle onto the
        physical links. """
        cdef double force = self.force.value
        cdef double[:] f_vec = np.zeros((3,), dtype='d')
        cdef short int link_id
        cdef unsigned int j
        cdef tuple local_f_vec
        cdef int k

        #: Apply forces
        for j in range(self.n_attachments - 1):
            #: link id
            link_id = self.local_waypoints[j][0]
            
            #: compute force vector
            c_scaled_unit_vector_from_points(
                self.global_waypoints[j],
                self.global_waypoints[j+1],
                force,
                f_vec
            )
            
            local_f_vec = convert_global_to_local(
                self.model_id, link_id, np.asarray(f_vec)
            )

            if self.debug_visualization:
                p.addUserDebugLine(
                    lineFromXYZ=np.asarray(self.global_waypoints[j]),
                    lineToXYZ=np.asarray(f_vec) + np.asarray(
                        self.global_waypoints[j]
                    ),
                    lineWidth=4,
                    lineColorRGB=[0, 1, 0],
                    replaceItemUniqueId=self.debug_force_ids[j]
                )

            #: Apply the force
            p.applyExternalForce(
                self.model_id,
                link_id,
                convert_local_to_inertial(
                    self.model_id, link_id, local_f_vec
                ),
                convert_local_to_inertial(
                    self.model_id, link_id, self.local_waypoints[j][1][:]
                ),
                flags=p.LINK_FRAME
            )

        for j in range(1, self.n_attachments):
            #: link id
            link_id = self.local_waypoints[j][0]
            
            c_scaled_unit_vector_from_points(
                self.global_waypoints[j],
                self.global_waypoints[j-1],
                force,
                f_vec
            )

            #: Apply the force
            local_f_vec = convert_global_to_local(
                self.model_id, link_id, np.asarray(f_vec)
            )
            if self.debug_visualization:
                p.addUserDebugLine(
                    lineFromXYZ=np.asarray(self.global_waypoints[j]),
                    lineToXYZ=np.asarray(f_vec) + np.asarray(
                        self.global_waypoints[j]
                    ),
                    lineWidth=4,
                    lineColorRGB=[0, 0, 1],                    
                    replaceItemUniqueId=self.debug_force_ids[
                        self.n_attachments + j - 2
                    ]
                )

            #: apply the force
            p.applyExternalForce(
                self.model_id,
                link_id,
                convert_local_to_inertial(
                    self.model_id, link_id, local_f_vec
                ),
                convert_local_to_inertial(
                    self.model_id, link_id,
                    self.local_waypoints[j][1][:]
                ),
                flags=p.LINK_FRAME
            )

    cdef void c_show_muscle(self):
        """ Visualize the muscle attachment. """
        if self.visualization:
            for j in range(self.n_attachments-1):
                p.addUserDebugLine(
                    lineFromXYZ=list(self.global_waypoints[j]),
                    lineToXYZ=list(self.global_waypoints[j+1]),
                    lineColorRGB=[self.stim.value, 0, 0],
                    lineWidth=4,
                    replaceItemUniqueId=self.debug_muscle_line_ids[j]
                )
