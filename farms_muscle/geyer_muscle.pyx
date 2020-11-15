# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

"""Implementation of Geyer muscle model."""
import farms_pylog as pylog
from farms_muscle.muscle cimport Muscle
from farms_muscle.physics_interface cimport PhysicsInterface
from farms_muscle.bullet_interface cimport BulletInterface
from libc.stdio cimport printf
from libc.math cimport sqrt as csqrt
from libc.math cimport pow as cpow
from libc.math cimport fmax as cfmax
from libc.math cimport fabs as cfabs
import numpy as np
cimport numpy as cnp

cdef class GeyerMuscle(Muscle):
    """Implementation of geyer muscle model.
    The muscle model is based on the hill-type muscle model by Geyer et.al
    """

    def __init__(self, container, parameters, dt=0.001, physics_engine='BULLET', model_id=1):
        """This function initializes the muscle model.
        A default muscle name is given as muscle

        Parameters
        ----------
        parameters : <MuscleParameters>
            Instance of MuscleParameters class
        model_id : <int>
            Only needed for bullet engine

        Returns:
        -------
        Muscle : <Muscle>
            Returns an instance of class Muscle

        """
        super(GeyerMuscle, self).__init__(parameters.name, dt, physics_engine)

        self.c = float(np.log(0.05))  # pylint: disable=no-member
        self.N = 1.5
        self.K = 5.0
        self.E_REF = 0.04  #: Reference strain
        self.W = 0.56  #: Shape factor pylint: disable=invalid-name
        self.tau_act = 0.01  # Time constant for the activation function
        self.F_per_m2 = 300000  # Force per m2 of muscle PCSA

        self.density = 1060
        self.tol = 1e-6  #: Tolerance

        #: Internal properties
        (_, self._l_slack) = container.muscles.constants.add_parameter(
            'l_slack_' + self._name, parameters.l_slack)
        (_, self._l_opt) = container.muscles.constants.add_parameter(
            'l_opt_' + self._name, parameters.l_opt)
        (_, self._v_max) = container.muscles.constants.add_parameter(
            'v_max_' + self._name, parameters.v_max)
        (_, self._f_max) = container.muscles.constants.add_parameter(
            'f_max_' + self._name, parameters.f_max)
        (_, self._pennation) = container.muscles.constants.add_parameter(
            'pennation_' + self._name, parameters.pennation)

        self._cos_alpha = np.cos(np.deg2rad(self._pennation))
        self._sin_alpha = np.sin(np.deg2rad(self._pennation))

        self._type = parameters.muscle_type

        # #: MUSCLE STATES
        # #: Muscle Contractile Length
        self._l_ce = container.muscles.states.add_parameter(
            'l_ce_' + self._name, parameters.l_ce0)[0]
        #: Muscle Activation
        self._activation = container.muscles.states.add_parameter(
            'activation_' + self._name, parameters.a0)[0]

        #: INPUTS TO THE MODEL
        #: Muscle length change
        self._l_mtu = container.muscles.parameters.add_parameter(
            'lmtu_'+self._name)[0]
        #: External Muscle stimulation
        self._stim = container.muscles.activations.add_parameter(
            'stim_' + self._name)[0]

        #: Derivatives
        self._v_ce = container.muscles.dstates.add_parameter(
            "v_ce_" + self._name, 0.0)[0]
        self._adot = container.muscles.dstates.add_parameter(
            "dA_" + self._name, 0.0)[0]

        #: Outputs
        self._l_se = container.muscles.outputs.add_parameter(
            "tendon_length_"+self._name, self._l_slack)[0]
        self._f_be = container.muscles.outputs.add_parameter(
            "belly_force_"+self._name, 0.0)[0]
        self._f_pe = container.muscles.outputs.add_parameter(
            "parallel_force_"+self._name, 0.0)[0]
        self._f_lce = container.muscles.outputs.add_parameter(
            "force_length_"+self._name, 0.0)[0]
        self._f_vce = container.muscles.outputs.add_parameter(
            "force_velocity_"+self._name, 0.0)[0]
        self._f_ce = container.muscles.outputs.add_parameter(
            "active_force_"+self._name, 0.0)[0]

        #: Main output of the muslce
        self._f_se = container.muscles.forces.add_parameter(
            "tendon_force_"+self._name, 0.0)[0]

        #: Sensory afferents
        #: Ia afferent constants
        self._kv = parameters.kv
        self._pv = parameters.pv
        self._k_dI = parameters.k_dI
        self._k_nI = parameters.k_nI
        self._const_I = parameters.const_I
        self._lth = parameters.lth

        #: Ib afferent constants
        self._kF = parameters.kF
        self._fth = parameters.fth

        #: II afferent constants
        self._k_dII = parameters.k_dII
        self._k_nII = parameters.k_nII
        self._const_II = parameters.const_II

        self._Ia_aff = container.muscles.Ia.add_parameter(
            "Ia_" + self._name, 0.0)[0]
        self._II_aff = container.muscles.II.add_parameter(
            "II_" + self._name, 0.0)[0]
        self._Ib_aff = container.muscles.Ib.add_parameter(
            "Ib_" + self._name, 0.0)[0]

        #: PhysicsInterface
        if physics_engine == 'NONE':
            self.p_interface = PhysicsInterface(self._l_mtu, self._f_se)
            pylog.warning(
                "Muscle {} connected to any physics engine".format(self._name))
        elif physics_engine == 'BULLET':
            self.p_interface = BulletInterface(
                model_id, self._l_mtu, self._f_se, self._stim,
                parameters.waypoints, parameters.visualize, parameters.debug)
            pylog.debug(
                "Muscle {} connected to any Bullet engine".format(self._name))

    def compute_initial_l_ce(self):
        """This function initializes the muscle lengths."""
        self.p_interface.update_muscle_length()
        l_mtu = self._l_mtu.c_get_value()
        if l_mtu < (self._l_slack + self._l_opt):
            l_ce = self.l_opt
        else:
            if (self._l_opt * self.W + self.E_REF * self._l_slack) != 0.0:
                _num = self._l_opt * self.W + \
                    self.E_REF * (l_mtu - self._l_opt)
                _den = self._l_opt * self.W + self.E_REF * self._l_slack
                l_se = self._l_slack*(_num/_den)
            else:
                l_se = self._l_slack
            l_ce = l_mtu - l_se
        return l_ce

    ########## C Wrappers ##########
    def _py_tendon_force(self, l_se):
        return self.c_tendon_force(l_se)

    def _py_parallel_star_force(self, l_ce):
        return self.c_parallel_star_force(l_ce)

    def _py_belly_force(self, l_ce):
        return self.c_belly_force(l_ce)

    def _py_activation_rate(self, act, stim):
        return self.c_activation_rate(act, stim)

    def _py_force_length(self, l_ce):
        return self.c_force_length(l_ce)

    def _py_force_velocity(self, v_ce):
        return self.c_force_velocity(v_ce)

    def _py_force_velocity_from_force(self, f_se,  f_be,  act,  f_l,
                                      f_pe_star):
        return self.c_force_velocity_from_force(
            f_se,  f_be,  act,  f_l,  f_pe_star)

    def _py_contractile_velocity(self, f_v):
        return self.c_contractile_velocity(f_v)

    #: Properties
    @property
    def global_waypoints(self):
        """Get global path points"""
        return self.p_interface.global_waypoints

    @property
    def local_waypoints(self):
        """Get local path points"""
        return self.p_interface.local_waypoints

    @property
    def muscle_force_idx(self):
        """Get the index of muscle force in the data table"""
        return self._f_se.idx

    @property
    def stim(self):
        """ Get current muscle stimulation """
        return self._stim.c_get_value()

    @stim.setter
    def stim(self, value):
        """ Set the muscle stimulation"""
        self._stim.c_set_value(value)

    #: LengthInfo
    @property
    def fiber_tendon_length(self):
        """ Get the length of muscle tendon unit.  """
        #### CHECK THIS ####
        return self._l_mtu.c_get_value()

    @property
    def fiber_length(self):
        """ Get the fiber length of the muscle.  """
        return self._l_ce.c_get_value()

    @property
    def tendon_length(self):
        """ Get the length of series tendon length  """
        return self.fiber_tendon_length - self.fiber_length

    #: Velocity Info
    @property
    def fiber_velocity(self):
        """ Get the fiber velocity.  """
        return self.c_contractile_velocity(self.force_velocity)

    #: Dynamics Info
    @property
    def activation(self):
        """ Get the muscle activation.  """
        return self._activation.c_get_value()

    @property
    def parallel_star_force(self):
        """ Get the force in parallel element*  """
        return self.c_parallel_star_force(self.fiber_length)

    @property
    def belly_force(self):
        """ Get the force in muscle belly.  """
        return self.c_belly_force(self.fiber_length)

    @property
    def tendon_force(self):
        """ Get the tendon force. """
        return self.c_tendon_force(self.tendon_length)

    @property
    def fiber_force(self):
        """ Get the force produced by the muscle fibers.  """
        return self.c_contractile_force(
            self._activation.c_get_value(),
            self._l_ce.c_get_value(),
            self._v_ce.c_get_value())

    #################### C-FUNCTIONS ####################
    cdef double[:] c_global_waypoints(self):
        """ Return global waypoints """
        return self.p_interface.global_waypoints

    cdef inline double c_tendon_force(self, double l_se) nogil:
        """ Setup the equations for tendon force. """
        cdef double _tendon_force
        cdef double _strain
        _strain = (l_se - self._l_slack) / (self._l_slack)
        _tendon_force = (self._f_max * (_strain / self.E_REF)**2) * (
            l_se > self._l_slack)
        # printf(
        #     'strain = %f \t force = %f \t slack = %f \n',
        #     _strain, _tendon_force, l_se,
        # )
        return min(self._f_max, _tendon_force)

    cdef inline double c_parallel_star_force(self, double l_ce) nogil:
        """ Setup the equations for parallell star pe* """
        cdef double _parallel_star_force
        _parallel_star_force = (self._f_max * (
            (l_ce - self._l_opt) / (self._l_opt * self.W))**2)*(
                l_ce > self._l_opt)
        return _parallel_star_force

    cdef inline double c_belly_force(self, double l_ce) nogil:
        """ Setup the equations for belly force  """
        cdef:
            double _f_be_cond, _num, _den, _belly_force
        _f_be_cond = self._l_opt * (1.0 - self.W)
        _num = l_ce - self._l_opt * (1.0 - self.W)
        _den = self._l_opt * self.W * 0.5
        _belly_force = self._f_max * \
            ((_num/_den)**2) * (l_ce <= _f_be_cond)
        return _belly_force

    cdef inline double c_activation_rate(self, double act, double stim) nogil:
        """ Define the change in activation. dA/dt. """

        cdef:
            double _stim_range, _d_act
        _stim_range = max(0.01, min(stim, 1.))
        _d_act = (_stim_range - act)/self.tau_act
        return _d_act

    cdef inline double c_force_length(self, double l_ce) nogil:
        """ Define the force length relationship. """

        cdef:
            double _val, _exposant, _force_length
        _val = cfabs(
            (l_ce - self._l_opt) / (self._l_opt * self.W))
        _exposant = self.c * _val**3
        _force_length = cexp(_exposant)
        return _force_length

    cdef inline double c_force_velocity(self, double v_ce) nogil:
        """ Define the force velocity relationship. """
        cdef:
            double _f_v_ce_eqn_1, _f_v_ce_eqn_2
        cdef double _v_max = self._v_max*self._l_opt
        _f_v_ce_eqn_1 = (
            self._v_max - v_ce)/(self._v_max + self.K*v_ce)
        _f_v_ce_eqn_2 = self.N + ((self.N - 1)*(
            self._v_max + v_ce)/(7.56*self.K*v_ce - self._v_max))
        if v_ce < 0.0:
            return _f_v_ce_eqn_1
        return _f_v_ce_eqn_2

    cdef inline double c_force_velocity_from_force(
            self, double f_se, double f_be, double act, double f_l, double f_pe_star) nogil:
        """ Define the force velocity relationship from forces."""
        cdef:
            double _f_v_ce_eqn_den, _f_v_ce_eqn_num, _f_v_ce_eqn, force_velocity
        _f_v_ce_eqn_num = f_se + f_be
        _f_v_ce_eqn_den = (self._f_max*act*f_l) + f_pe_star

        #: Check these TOLERANCES
        # if -self.tol < _f_v_ce_eqn_den < self.tol:
        #     _f_v_ce_eqn = 0.0
        # else:
        _f_v_ce_eqn = _f_v_ce_eqn_num/_f_v_ce_eqn_den
        force_velocity = max(0.0, min(1.5, _f_v_ce_eqn))
        return force_velocity

    cdef inline double c_contractile_velocity(self, double f_v) nogil:
        """ Define the contractile velocity."""
        cdef:
            double _v_ce_1, _v_ce_2, _v_max
        _v_max = self._v_max*self._l_opt
        _v_ce_1 = _v_max*(1. - f_v)/(1. + f_v*self.K)
        _v_ce_2 = _v_max*(f_v - 1.0)/(
            7.56*self.K * (f_v - self.N) + 1. - self.N)
        # printf('f_v = %f \n', f_v)
        # printf('v_ce_1 = %f \t v_ce_2 = %f \n', _v_ce_1, _v_ce_2)
        if f_v <= 1.0:
            return _v_ce_1
        return _v_ce_2

    cdef inline double c_contractile_force(
            self, double activation, double l_ce, double v_ce) nogil:
        """ Compute the active force. """
        return activation*self._f_max*self.c_force_length(l_ce)*self.c_force_velocity(v_ce)

    cdef void c_ode_rhs(self) nogil:
        """Muscle Model ODE rhs.
            Returns
            ----------
            ode_rhs: list < cas.SX >
                description
        """

        # printf('c_ode_rhs muscle ....\n')
        cdef double _act = self._activation.c_get_value()
        # printf('_act = %f \n', _act)
        cdef double _l_ce_tol = cfmax(self._l_ce.c_get_value(), 0.0)
        # printf('_l_ce_tol = %f \n', _l_ce_tol)

        #: Algrebaic Equation
        cdef double _l_mtu = self._l_mtu.c_get_value()
        # printf('_l_mtu = %f \n', _l_mtu)

        cdef double _l_se = _l_mtu - _l_ce_tol*self._cos_alpha
        # printf('_l_se = %f \n', _l_se)

        # #: Muscle Dynamics
        # #: Series Force
        cdef double _f_se = self.c_tendon_force(_l_se)
        # printf('self.c_tendon_force(_l_se) = %f \n', _f_se)

        # #: Muscle Belly Force
        cdef double _f_be = self.c_belly_force(_l_ce_tol)
        # printf('self.c_belly_force(_l_ce_tol) = %f \n', _f_be)

        # #: Force-Length Relationship
        cdef double _f_l = self.c_force_length(_l_ce_tol)
        # printf('self.c_force_length(_l_ce_tol) = %f \n', _f_l)

        # #: Force Parallel Element
        cdef double _f_pe_star = self.c_parallel_star_force(_l_ce_tol)
        # printf('self.c_parallel_star_force(_l_ce_tol) = %f \n', _f_pe_star)

        # #: Force Velocity Inverse Relation
        cdef double _f_v = self.c_force_velocity_from_force(
            _f_se,
            _f_be,
            _act,
            _f_l,
            _f_pe_star)
        # printf('self.c_force_velocity_from_force = %f \n', _f_v)

        #: State Update
        #: Muscle Actvation Dynamics
        # printf('self.c_activation_rate ....\n')
        self._adot.c_set_value(self.c_activation_rate(
            _act,
            self._stim.c_get_value()))

        #: Muscle Contractile Velocity
        # printf('self.c_contractile_velocity(_f_v)) ....\n')
        self._v_ce.c_set_value(self.c_contractile_velocity(_f_v))

    cdef void c_output(self) nogil:
        """ Compute the outputs of the system. """
        #: Attributes needed for output computation
        cdef double l_ce = self._l_ce.c_get_value()
        cdef double v_ce = self._v_ce.c_get_value()
        cdef double act = self._activation.c_get_value()
        cdef double l_mtu = self._l_mtu.c_get_value()
        cdef double l_se = l_mtu - l_ce*self._cos_alpha

        #: Tendon length
        self._l_se.c_set_value(l_se)
        #: Belly force
        self._f_be.c_set_value(self.c_belly_force(l_ce))
        #: Parallel force
        self._f_pe.c_set_value(self.c_parallel_star_force(l_ce))
        #: Force length
        self._f_lce.c_set_value(self.c_force_length(l_ce))
        #: Force velocity
        self._f_vce.c_set_value(self.c_force_velocity(v_ce))
        #: Contractile force
        self._f_ce.c_set_value(self.c_contractile_force(act, l_ce, v_ce))
        #: Tendon force
        self._f_se.c_set_value(self.c_tendon_force(l_se))

    #: Sensory afferents
    cdef void c_compute_Ia(self) nogil:
        """ Compute Ia afferent from muscle fiber. """
        cdef double _v_norm = self._v_ce.c_get_value()/self._lth

        cdef double _d_norm = (
            self._l_ce.c_get_value() - self._lth)/self._lth if self._l_ce.c_get_value() >= self._lth else 0.0

        self._Ia_aff.c_set_value(self._kv*cpow(
            cfabs(_v_norm), self._pv) + self._k_dI*_d_norm + self._k_nI*self._stim.c_get_value() + self._const_I)

    cdef void c_compute_II(self) nogil:
        """ Compute II afferent from muscle fiber. """
        cdef double _d_norm = (
            self._l_ce.c_get_value() - self._lth)/self._lth if self._l_ce.c_get_value() >= self._lth else 0.0

        self._II_aff.c_set_value(
            self._k_dII*_d_norm + self._k_nII*self._stim.c_get_value() + self._const_II)

    cdef void c_compute_Ib(self) nogil:
        """ Compute Ib afferent from muscle fiber. """
        cdef double _f_norm = (
            self._f_se.c_get_value() - self._fth)/self._f_max if self._f_se.c_get_value() >= self._fth else 0.0

        self._Ib_aff.c_set_value(self._kF*_f_norm)

    cdef void c_update_sensory_afferents(self) nogil:
        """ Compute all the sensory afferents and update them. """
        self.c_compute_Ia()
        self.c_compute_II()
        self.c_compute_Ib()
