# cython: cdivision=True
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

""" Interface between bullet physics engine and muscle model. """

import pybullet as p
from physics_interface cimport PhysicsInterface
cimport numpy as cnp
from numpy cimport NPY_FLOAT
from libc.math cimport sqrt as csqrt
import numpy as np
import transformations as T
import farms_pylog as pylog 

cdef class BulletInterface(PhysicsInterface):
    """Interface between bullet physics engine and muscle model.
    """
    def __init__(self, model_id, lmtu, force, stim, waypoints, VISUALIZATION=True):
        super(BulletInterface, self).__init__(lmtu, force, stim, 'BULLET')
        #: [TO-DO] ADD KWARGS FOR VISUALIZATION
        #: Number of waypoints
        self.model_id = model_id
        pylog.debug('Model {} -> num points {}'.format(
            self.model_id, len(waypoints)))
        
        self.num_attachments = len(waypoints)        

        self._points = cnp.ndarray((self.num_attachments,),
                                   dtype=('(3,)d'))

        
        self.waypoints = cnp.ndarray((self.num_attachments,),
                                            dtype=[('link_id','i'),
                                                   ('point','(4,)d')])

        self._vis_ids = cnp.ndarray((self.num_attachments,),
                                    dtype=('I'))
        
        self.VISUALIZATION = VISUALIZATION        
        
        #: Add the waypoints
        self.add_waypoints(waypoints, VISUALIZATION)

    #################### PY-FUNCTIONS ####################
    def add_waypoints(self, waypoints, VISUALIZATION):
        """ Add new attachment point. 
        Parameters
        ----------
        waypoints: <list>
            List containing link ids and attachment points
        """

        _link_name_to_index = {p.getBodyInfo(self.model_id)[0].decode('UTF-8'):-1,}

        for _id in range(p.getNumJoints(self.model_id)):
            _name = p.getJointInfo(self.model_id, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id                
        pylog.debug('Link-Index -> {}'.format(_link_name_to_index))

        for j, attachment in enumerate(waypoints):
            _link_id = _link_name_to_index[attachment[0]['link']]
                
            self.waypoints[j][0] = _link_id
            self.waypoints[j][1][0] = attachment[1]['point'][0]
            self.waypoints[j][1][1] = attachment[1]['point'][1]
            self.waypoints[j][1][2] = attachment[1]['point'][2]
            self.waypoints[j][1][3] = 1.

            self._points[j][:] = attachment[1]['point'][:3]

            if VISUALIZATION and j <= self.num_attachments-2:
                self._vis_ids[j] = p.addUserDebugLine(
                    lineFromXYZ=self._points[j],
                    lineToXYZ=self._points[j+1],
                    lineColorRGB=[1, 0, 0],
                    lineWidth=4,
                    lifeTime=0)
            

    def py_compute_muscle_length(self):
        self.c_compute_muscle_length()
        
    def py_dist_between_points(self, p1, p2):
        return self.c_dist_between_points(p1, p2)

    def py_force_vector(self, p1, p2, force, f_vec):        
        return self.c_force_vector(p1, p2, force, f_vec)
    
    #################### C-FUNCTIONS ####################
    cdef void c_compute_muscle_length(self):
        """ Compute the muscle length based on the physics simulator. """

        cdef unsigned int _index = 0
        
        for link_id, point in self.waypoints:
            if link_id == -1:
                (pos, orient) = p.getBasePositionAndOrientation(
                    self.model_id)
            else:
                (_, _, _, _, pos, orient, *_) = p.getLinkState(
                    self.model_id, link_id)
            trans = T.compose_matrix(angles=p.getEulerFromQuaternion(orient),
                             translate=pos)
            self._points[_index][:] = np.dot(trans, point)[:3]
            _index += 1
            
        #: Compute the length
        cdef double _length = 0.0
        for j in range(self.num_attachments-1):
            _length +=self.c_dist_between_points(self._points[j],
                                                 self._points[j+1])

        #: Update the muscle length
        self.lmtu.c_set_value(_length)

    cdef inline double c_dist_between_points(self, double[:] p1, double[:] p2) nogil:
        """ Compute distance between two points. """
        cdef double dist = 0.
        
        for j in range(3):
            dist += (p1[j]-p2[j])*(p1[j]-p2[j])
        return csqrt(dist)

    cdef inline void c_force_vector(self, double[:] p1, double[:] p2, double force, double[:] f_vec) nogil:
        """ Compute the force vector between two given points. """
        for j in range(3):
            f_vec[j] = p1[j] - p2[j]

        #: Normalize
        cdef double mag = 0.0
        for j in range(3):
            mag += f_vec[j]*f_vec[j]
        mag = csqrt(mag)
        for j in range(3):
            f_vec[j] = f_vec[j]*force/mag
            
    cdef void c_apply_muscle_forces(self):
        """ Apply the forces generated by the muscle onto the physical links. """
        cdef double _force = self.force.value
        cdef double[:] f_vec = np.zeros((3,),dtype='d')
        cdef int _link_id

        #: Apply forces
        for j in range(self.num_attachments - 1):
            #: link id
            _link_id = self.waypoints[j][0]
            
            self.c_force_vector(
                self._points[j+1], self._points[j], _force, f_vec)
            
            #: To be checked
            if _link_id == -1:
                (pos, orient) = p.getBasePositionAndOrientation(
                    self.model_id)
            else:
                (_, _, _, _, pos, orient, *_) = p.getLinkState(
                    self.model_id, _link_id)
                
            trans = T.inverse_matrix(
                T.compose_matrix(angles=p.getEulerFromQuaternion(orient),
                                 translate=pos))
            f_vec = np.dot(trans, np.append(f_vec,[1]))[:3]

            #: Apply the force
            p.applyExternalForce(
                self.model_id, _link_id, f_vec, self.waypoints[j][1][:3],
                flags=p.LINK_FRAME)

        for j in range(1, self.num_attachments):
            #: link id
            _link_id = self.waypoints[j][0]
            
            self.c_force_vector(
                self._points[j-1], self._points[j], _force, f_vec)
            
            #: To be checked
            if _link_id == -1:
                (pos, orient) = p.getBasePositionAndOrientation(
                    self.model_id)
            else:
                (_, _, _, _, pos, orient, *_) = p.getLinkState(
                    self.model_id, _link_id)
                
            trans = T.inverse_matrix(
                T.compose_matrix(angles=p.getEulerFromQuaternion(orient),
                                 translate=pos))
            f_vec = np.dot(trans, np.append(f_vec,[1]))[:3]

            #: Apply the force
            p.applyExternalForce(
                self.model_id, _link_id, f_vec, self.waypoints[j][1][:3],
                flags=p.LINK_FRAME)

    cdef void c_show_muscle(self, bint VISUALIZATION=True):
        """ Visualize the muscle attachment. """
        if VISUALIZATION:
            for j in range(self.num_attachments-1):
                p.addUserDebugLine(
                    lineFromXYZ=np.asarray(self._points[j]),
                    lineToXYZ=np.asarray(self._points[j+1]),
                    lineColorRGB=[self.stim.value, 0, 0],
                    lineWidth=4,
                    lifeTime=0,
                    replaceItemUniqueId=self._vis_ids[j])
