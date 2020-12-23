
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

"""Implementation of Millard rigid tendon model."""
import farms_pylog as pylog
from farms_muscle.muscle cimport Muscle
from farms_muscle.physics_interface cimport PhysicsInterface
from farms_muscle.bullet_interface cimport BulletInterface
from libc.stdio cimport printf
from libc.math cimport sqrt as csqrt
from libc.math cimport sin as csin
from libc.math cimport pow as cpow
from libc.math cimport cos as ccos
from libc.math cimport atan as catan
from libc.math cimport acos as cacos
from libc.math cimport fmax as cfmax
from libc.math cimport fabs as cfabs
from libc.math cimport log as clog
import numpy as np
cimport numpy as cnp

cdef class MillardRigidTendonMuscle(Muscle):
    """Implementation of MillarRigidTendonMuscle.
    The muscle model is based on the hill-type muscle model by millard 2013
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
        super(MillardRigidTendonMuscle, self).__init__(parameters.name,
                                                       dt,
                                                       physics_engine)

        self.c = float(np.log(0.05))  # pylint: disable=no-member
        self.N = 1.5
        self.K = 5.0
        self.E_REF = 0.04  #: Reference strain
        self.W = 0.56  #: Shape factor pylint: disable=invalid-name
        self.tau_act = 0.01  # Time constant for the activation function
        self.F_per_m2 = 300000  # Force per m2 of muscle PCSA

        self.density = 1060
        self.tol = 1e-6  #: Tolerance

        #: Force-Length Constants
        self.b1[0] = 0.815
        self.b2[0] = 1.055
        self.b3[0] = 0.162
        self.b4[0] = 0.063

        self.b1[1] = 0.433
        self.b2[1] = 0.717
        self.b3[1] = -0.030
        self.b4[1] = 0.2

        self.b1[2] = 0.1
        self.b2[2] = 1.0
        self.b3[2] = 0.354
        self.b4[2] = 0.0

        #: Force-Velocity constants
        self.d1 = -0.318
        self.d2 = -8.149
        self.d3 = -0.374
        self.d4 = 0.886

        #: Passive element constants
        self.kpe = 4.0
        self.e0 = 0.6

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
        #: FUCK : Need to update beta in parameters class
        (_, self._beta) = container.muscles.constants.add_parameter(
            'beta_' + self._name, 0.1)

        self._cos_alpha = np.cos(np.deg2rad(self._pennation))
        self._sin_alpha = np.sin(np.deg2rad(self._pennation))

        self._parallelogram_height = self._l_opt*self._sin_alpha

        self._type = parameters.muscle_type

        # #: Muscle Contractile Length
        self._l_ce = container.muscles.parameters.add_parameter(
            "l_ce_" + self._name, parameters.l_ce0)[0]
        self._v_ce = container.muscles.parameters.add_parameter(
            "v_ce_" + self._name, 0.0)[0]

        #: INPUTS TO THE MODEL
        #: Muscle length change
        self._l_mtu = container.muscles.parameters.add_parameter(
            'lmtu_'+self._name)[0]
        #: External Muscle stimulation
        self._stim = container.muscles.activations.add_parameter(
            'stim_' + self._name)[0]

        # #: MUSCLE STATES
        #: Muscle Activation
        self._activation = container.muscles.states.add_parameter(
            'activation_' + self._name, parameters.a0)[0]

        #: Derivatives
        self._adot = container.muscles.dstates.add_parameter(
            "dA_" + self._name, 0.0)[0]

        #: Outputs
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
            self.p_interface = PhysicsInterface(
                self._l_mtu, self._f_se, self._stim)
            pylog.warning(
                "Muscle {} connected to any physics engine".format(self._name))
        elif physics_engine == 'BULLET':
            self.p_interface = BulletInterface(
                model_id, self._l_mtu, self._f_se, self._stim,
                parameters.waypoints, parameters.visualize)
            pylog.debug(
                "Muscle {} connected to any Bullet engine".format(self._name))

        pylog.debug("Muscle {} initialized".format(self._name))

    def compute_initial_l_ce(self):
        """This function initializes the muscle lengths."""
        self.p_interface.update_muscle_length()
        l_mtu = self._l_mtu.c_get_value()
        l_ce = self._l_opt
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
    def compute_activation_rate(self, act, stim):
        return self.c_activation_rate(act, stim)

    def compute_force_length(self, l_mtu):
        l_ce = self.compute_fiber_length(
            l_mtu, self.compute_pennation_angle(l_mtu))
        return self.c_force_length(l_ce/self._l_opt)

    def compute_force_velocity(self, l_mtu, v_mtu):
        v_ce = self.compute_fiber_velocity(
            v_mtu, self.compute_pennation_angle(l_mtu))
        return self.c_force_velocity(v_ce/self._v_max)

    def compute_pennation_angle(self, l_mtu):
        return self.c_pennation_angle(l_mtu)

    def compute_fiber_length(self, l_mtu, alpha):
        return self.c_fiber_length(l_mtu, alpha)

    def compute_fiber_velocity(self, v_mtu, alpha):
        return self.c_fiber_velocity(v_mtu, alpha)

    def compute_active_force(self, activation, l_mtu, v_mtu):
        alpha = self.compute_pennation_angle(l_mtu)
        return activation*self.compute_force_length(l_mtu)*self.compute_force_velocity(l_mtu, v_mtu)

    def compute_passive_force(self, l_mtu):
        l_ce = self.compute_fiber_length(
            l_mtu, self.compute_pennation_angle(l_mtu))
        return self.c_passive_force(l_ce/self._l_opt)

    def compute_tendon_force(self, activation, l_mtu, v_mtu):
        alpha = self.compute_pennation_angle(l_mtu)
        v_ce = self.compute_fiber_velocity(v_mtu, alpha)
        return self._f_max*(
            self.compute_active_force(activation, l_mtu, v_mtu) +
            self.compute_passive_force(l_mtu) + 0.1*v_ce/self._v_max
        )*ccos(alpha)

    #: Properties
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
        return self.c_contractile_force(self._activation.c_get_value(),
                                        self._l_ce.c_get_value(),
                                        self._v_ce.c_get_value())

    @property
    def fiber_force(self):
        """ Get the force produced by the muscle fibers.  """
        return self.c_contractile_force(
            self._activation.c_get_value(),
            self._l_ce.c_get_value(),
            self._v_ce.c_get_value())

    #################### C-FUNCTIONS ####################
    cdef inline double c_activation_rate(self, double act, double stim) nogil:
        """ Define the change in activation. dA/dt. """
        cdef:
            double _stim_range, _d_act
        _stim_range = max(0.01, min(stim, 1.))
        _d_act = (_stim_range - act)/self.tau_act
        return _d_act

    cdef inline double c_pennation_angle(self, double l_mtu) nogil:
        """ Compute the pennation angles """
        return catan(self._parallelogram_height/(l_mtu-self._l_slack))

    cdef inline double c_tendon_force(self, double l_se) nogil:
        """ Setup the equations for tendon force. """
        return 0.0

    cdef inline double c_passive_force(self, double l_ce) nogil:
        """ Setup the equations for passive force* """
        cdef double _den = cexp(self.kpe) - 1.0
        cdef double _num = cexp((self.kpe*(l_ce) - self.kpe)/self.e0) - 1.0
        return _num/_den

    cdef inline double c_force_length(self, double l_ce) nogil:
        """ Define the force length relationship. """
        cdef double _force_length = 0.0
        cdef double _num, _den
        cdef unsigned int j = 0
        for j in range(3):
            _num = -0.5*(l_ce - self.b2[j])**2
            _den = (self.b3[j] + self.b4[j]*l_ce)**2
            _force_length += self.b1[j]*cexp(_num/_den)
        return _force_length

    cdef inline double c_force_velocity(self, double v_ce) nogil:
        """ Define the force velocity relationship. """
        cdef double exp1 = self.d2*v_ce + self.d3
        cdef double exp2 = ((self.d2*v_ce + self.d3)**2) + 1.
        return self.d1*clog(exp1 + csqrt(exp2)) + self.d4

    cdef inline double c_contractile_force(
            self, double activation, double l_ce, double v_ce) nogil:
        """ Compute the active force. """
        return activation*self.c_force_length(l_ce)*self.c_force_velocity(v_ce)

    cdef inline double c_muscle_velocity(
            self, double l_mtu_curr, double l_mtu_prev, double dt) nogil:
        """ Compute the fiber length. """
        return (l_mtu_curr - l_mtu_prev)/(dt)

    cdef inline double c_fiber_length(self, double l_mtu, double alpha) nogil:
        """ Compute the fiber length. """
        return (l_mtu - self._l_slack)/ccos(alpha)

    cdef inline double c_fiber_velocity(self, double v_mtu, double alpha) nogil:
        """ Compute the fiber velocity. """
        return v_mtu/ccos(2*alpha)

    cdef void c_ode_rhs(self) nogil:
        """Muscle Model ODE rhs.
            Returns
            ----------
            ode_rhs: list < cas.SX >
                description
        """

        # printf('c_ode_rhs muscle ....\n')
        cdef double _act = self._activation.c_get_value()

        #: State Update
        #: Muscle Actvation Dynamics
        # printf('self.c_activation_rate ....\n')
        self._adot.c_set_value(self.c_activation_rate(
            _act,
            self._stim.c_get_value()))

    cdef void c_output(self) nogil:
        """ Compute the outputs of the system. """
        #: Attributes needed for output computation
        cdef double l_mtu = self._l_mtu.c_get_value()
        cdef double alpha = self.c_pennation_angle(l_mtu)
        cdef double l_ce_norm = self.c_fiber_length(l_mtu, alpha)/self._l_opt
        cdef double v_mtu = self.c_muscle_velocity(
            l_mtu, self._l_mtu.c_get_prev_value(), self.dt)
        cdef double v_ce_norm = self.c_fiber_velocity(v_mtu, alpha)/(self._l_opt*self._v_max)
        cdef double act = self._activation.c_get_value()
        cdef double l_se = l_mtu - l_ce_norm*ccos(alpha)*self._l_opt

        #: l_ce
        self._l_ce.c_set_value(l_ce_norm*self._l_opt)
        #: v_ce
        self._v_ce.c_set_value(v_ce_norm*self._l_opt*self._v_max)
        #: Tendon length
        self._l_se.c_set_value(self._l_slack)
        #: Parallel force
        self._f_pe.c_set_value(self.c_passive_force(l_ce_norm))
        #: Force length
        self._f_lce.c_set_value(self.c_force_length(l_ce_norm))
        #: Force velocity
        self._f_vce.c_set_value(self.c_force_velocity(v_ce_norm))
        #: Contractile force
        self._f_ce.c_set_value(
            self.c_contractile_force(act, l_ce_norm, v_ce_norm)
        )
        #: Tendon force
        cdef double force = self._f_max*(
            self._f_ce.c_get_value() + self._f_pe.c_get_value() + self._beta*v_ce_norm
        )*ccos(alpha)
        self._f_se.c_set_value(force)

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
