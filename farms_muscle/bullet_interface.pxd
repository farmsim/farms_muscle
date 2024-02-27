""" Interface between bullet physics engine and muscle model. """

from libc.math cimport sqrt as csqrt
import numpy as np
import pybullet as p
from farms_muscle.physics_interface cimport PhysicsInterface
cimport numpy as np


# Vector operations
cdef inline double c_distance_between_points(
    double[:] point_1, double[:] point_2
):
    """ Compute distance between two points. """
    cdef double dist = 0.
    cdef unsigned int j
    for j in range(point_1.shape[0]):
        dist += (point_1[j]-point_2[j])*(point_1[j]-point_2[j])
    return csqrt(dist)

cdef inline void c_vector_from_points(
    double[:] point_1, double[:] point_2, double[:] output_vector
):
    """ Compute vector from points """
    cdef unsigned int j
    for j in range(point_1.shape[0]):
        output_vector[j] = point_2[j] - point_1[j]

cdef inline void c_unit_vector_from_points(
    double[:] point_1, double[:] point_2, double[:] output_vector
):
    """ Compute unit vector from points """
    cdef unsigned int j
    cdef double norm = c_distance_between_points(point_1, point_2)
    for j in range(point_1.shape[0]):
        output_vector[j] = (point_2[j] - point_1[j])/norm

cdef inline double c_vector_norm(double[:] vector):
    """ Compute vector norm. """
    cdef double norm = 0.
    cdef unsigned int j
    for j in range(vector.shape[0]):
        norm += (vector[j])*(vector[j])
    return csqrt(norm)

cdef inline void c_unit_vector_from_vector(
    double[:] vector, double[:] output_vector
):
    """ Compute unit vector of the given vector """
    cdef unsigned int j
    cdef double norm = c_vector_norm(vector)
    for j in range(vector.shape[0]):
        output_vector[j] = vector[j]/norm

cdef inline void c_scaled_unit_vector_from_points(
    double[:] point_1, double[:] point_2, double scale_factor,
    double[:] output_vector
):
    """ Compute a scaled unit vector from the given set of points. """
    cdef unsigned int j
    cdef double norm = c_distance_between_points(point_1, point_2)
    for j in range(point_1.shape[0]):
        output_vector[j] = (scale_factor*(point_2[j] - point_1[j]))/norm


cdef class BulletInterface(PhysicsInterface):
    # Properties
    cdef:
        int model_id
        unsigned int n_attachments
        bint visualization
        bint debug_visualization
        np.ndarray local_waypoints
        np.ndarray global_waypoints
        np.ndarray debug_muscle_line_ids
        np.ndarray debug_force_ids

    #################### C-FUNCTIONS ####################
    cdef:
        void c_compute_muscle_length(self)
        void c_apply_muscle_forces(self)
        void c_show_muscle(self)
