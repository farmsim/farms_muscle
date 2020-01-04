"""Implementation of MillardRigidTendonMuscle model."""
from farms_container.parameter cimport Parameter
from farms_muscle.muscle cimport Muscle
from libc.math cimport exp as cexp
from libc.math cimport acosh as cacosh
cimport numpy as cnp

cdef class MillardRigidTendonMuscle(Muscle):
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

        #: Pennation angle alpha
        double _alpha
        double _cos_alpha
        double _sin_alpha

        #: Damping constant
        double _beta
        
        #: Inputs
        Parameter _stim
        Parameter _l_mtu

        #: Derivatives
        Parameter _v_ce
        Parameter _adot

        #: Outputs
        Parameter _l_se
        Parameter _l_mtc
        Parameter _f_be
        Parameter _f_pe
        Parameter _f_lce
        Parameter _f_vce
        Parameter _f_ce
        Parameter _f_se

        #: Afferents
        double _kv
        double _pv
        double _k_dI
        double _k_nI
        double _const_I
        double _lth

        #: II afferent constants
        double _kF
        double _fth

        #: Ib afferent constants
        double _k_dII
        double _k_nII
        double _const_II

    cdef:
        #: SUB-MUSCLE FUNCTIONS
        inline double c_tendon_force(self, double l_se) nogil
        inline double c_parallel_star_force(self, double l_ce) nogil
        inline double c_belly_force(self, double l_ce) nogil
        inline double c_activation_rate(self, double act, double stim) nogil
        inline double c_force_length(self, double l_ce) nogil
        inline double c_force_velocity(self, double v_ce) nogil        
        inline double c_contractile_force(
            self, double activation, double l_ce, double v_ce) nogil
        inline double c_muscle_velocity(
            self, double l_mtu_curr, double l_mtu_prev, double dt) nogil        
        inline double c_fiber_length(
            self, double l_mtu, double l_slack) nogil
        inline double c_fiber_velocity(
            self, double v_mtu, double l_ce) nogil

        #: Sensory afferents
        void c_compute_Ia(self) nogil
        void c_compute_II(self) nogil
        void c_compute_Ib(self) nogil
        void c_update_sensory_afferents(self) nogil
        #: ODE
        void c_ode_rhs(self) nogil
        #: OUTPUT
        void c_output(self) nogil