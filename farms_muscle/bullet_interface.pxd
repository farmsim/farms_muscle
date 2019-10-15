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
        bint VISUALIZATION
        cnp.ndarray waypoints
        cnp.ndarray _points
        cnp.ndarray _vis_ids
        cnp.ndarray _base_pos
        
    #################### C-FUNCTIONS ####################
    @staticmethod
    cdef:
        inline cnp.ndarray[double, ndim=2] c_compose_matrix(
            double[:] position, 
            double[:] orientation)
    cdef:
        inline double c_dist_between_points(
            self, double[:] p1, double[:] p2) nogil
        inline void c_force_vector(
            self, double[:] p1, double[:] p2, double force,
            double[:] f_vec) nogil
        cnp.ndarray[double, ndim=2] c_get_link_transform(
            self, int link_id)
        cnp.ndarray[double, ndim=2] c_get_base_transform(self)
        cnp.ndarray[double, ndim=1] c_transform_point(
            self, int link_id, cnp.ndarray[double, ndim=1] point)
        void c_compute_muscle_length(self)
        void c_apply_muscle_forces(self)
        void c_show_muscle(self)
