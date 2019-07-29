""" Interface between bullet physics engine and muscle model. """

import pybullet as p
from physics_interface cimport PhysicsInterface
cimport numpy as cnp
import numpy as np

cdef class BulletInterface(PhysicsInterface):
    #: Properties
    cdef:
        int model_id
        int num_attachments
        cnp.ndarray attachment_points
        cnp.ndarray _points
        
    #################### C-FUNCTIONS ####################    
    cdef:
        inline double c_dist_between_points(self, double[:] p1, double[:] p2) nogil
        inline void c_force_vector(self, double[:] p1, double[:] p2, double force, double[:] f_vec) nogil
        void c_compute_muscle_length(self)
        void c_apply_muscle_forces(self)