"""Implementation of Geyer muscle model."""
from farms_dae_generator.param cimport Param
from farms_dae_generator.parameters cimport Parameters
from farms_muscle.muscle cimport Muscle
from libc.math cimport exp as cexp
from libc.math cimport acosh as cacosh
cimport numpy as cnp

cdef class GeyerMuscle(Muscle):
    # Default Muscle Parameters
    cdef:
        readonly float  c
        readonly float  N
        readonly float  K
        readonly float  E_REF  # : Reference strain
        readonly float  W  # : Shape factor pylint: disable=invalid-name
        readonly float  tau_act   #: Time constant for the activation function
        readonly unsigned int  F_per_m2   #: Force per m2 of muscle PCSA
        readonly unsigned int  density

    #: Muscle Parameters
    cdef:
        double tol
        double _td_to_sc
        double _td_from_sc
        double _motiontype
        # str _name
        str m_id
        str _type
        unsigned short int num_joints

        #: Inputs
        Param _stim
        Param _l_mtu

        #: Derivatives
        Param _v_ce
        Param _adot

        #: Outputs
        Param _l_se
        Param _l_mtc
        Param _f_be
        Param _f_pe
        Param _f_lce
        Param _f_vce
        Param _f_ce
        Param _f_se

    cdef:
        #: SUB-MUSCLE FUNCTIONS
        inline double c_tendon_force(self, double l_se) nogil
        inline double c_parallel_star_force(self, double l_ce) nogil
        inline double c_belly_force(self, double l_ce) nogil
        inline double c_activation_rate(self, double act, double stim) nogil
        inline double c_force_length(self, double l_ce) nogil
        inline double c_force_velocity(self, double v_ce) nogil
        inline double c_force_velocity_from_force(
            self, double f_se, double f_be, double act, double f_l, double f_pe_star) nogil
        inline double c_contractile_velocity(self, double f_v) nogil
        inline double c_contractile_force(
            self, double activation, double l_ce, double v_ce) nogil
        #: ODE
        void c_ode_rhs(self) nogil
        #: OUTPUT
        void c_output(self) nogil
