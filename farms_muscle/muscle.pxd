"""Muscle abstract class"""

from farms_dae_generator.param cimport Param

cdef class Muscle(object):
    #: Properties
    cdef:
        str _name
        double _l_slack
        double _l_opt
        double _f_max
        double _v_max
        double _pennation

    #: States
    cdef:
        Param _activation
        Param _l_ce

    #: Methods
    cdef:
        void c_ode_rhs(self) nogil
        #: OUTPUT
        void c_output(self) nogil
