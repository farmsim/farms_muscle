"""Muscle abstract class"""

from farms_container.parameter cimport Parameter
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
        #: Time Step
        double dt

    #: States
    cdef:
        Parameter _activation
        Parameter _l_ce

    #: Sensory afferents
    cdef:
        Parameter _Ia_aff
        Parameter _II_aff
        Parameter _Ib_aff

    #: Methods
    cdef:
        void c_ode_rhs(self) nogil        
        #: OUTPUT
        void c_output(self) nogil
        #: Sensory afferents
        void c_compute_Ia(self) nogil
        void c_compute_II(self) nogil
        void c_compute_Ib(self) nogil
        void c_update_sensory_afferents(self) nogil
