"""Muscle abstract class"""

from farms_dae_generator.param cimport Param
from farms_dae_generator.parameters cimport Parameters
from farms_muscle.physics_interface cimport PhysicsInterface
from farms_muscle.bullet_interface cimport BulletInterface

cdef class Muscle(object):
    #: Properties
    cdef:
        str _name
        str _physics_engine
        double _l_slack
        double _l_opt
        double _f_max
        double _v_max
        double _pennation
        PhysicsInterface p_interface

    #: States
    cdef:
        Param _activation
        Param _l_ce

    #: Access to inputs of the system
    cdef:
        Parameters u

    #: Methods
    cdef:
        void c_ode_rhs(self) nogil
        #: OUTPUT
        void c_output(self) nogil
