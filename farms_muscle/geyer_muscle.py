"""Implementation of Geyer muscle model."""

from muscle import Muscle
import numpy as np
import casadi as cas


class GeyerMuscle(Muscle):
    """Implementation of geyer muscle model.
    The muscle model is based on the hill-type muscle model by Geyer et.al
    """
    # Default Muscle Parameters

    c = np.log(0.05)  # pylint: disable=no-member
    N = 1.5
    K = 5.0
    tau_act = 0.01  # Time constant for the activation function
    F_per_m2 = 300000  # Force per m2 of muscle PCSA
    density = 1060

    def __init__(self, dae, parameters):
        """This function initializes the muscle model.
        A default muscle name is given as muscle

        Parameters
        ----------
        parameters : <MuscleParameters>
            Instance of MuscleParameters class

        Returns:
        -------
        Muscle : <Muscle>
            Returns an instance of class Muscle

        Attributes:
        ----------

        Methods:
        --------
        step : func
            Integrates muscle state by time step dt

        Example:
        --------
        >>> from SystemParameters import MuscleParameters
        >>> import Muscle
        >>> muscle_parameters = MuscleParameters()
        >>> muscle1 = GeyerMuscle.Muscle(muscle_parameters)
        >>> muscle1.stim = 0.05
        >>> muscle1.deltaLength = 0.01
        >>> muscle1.step(dt)
        """
        super(GeyerMuscle, self).__init__()

        self.tol = 1e-6  #: Tolerance
        self.dae = dae

        #:  Muscle specific parameters initialization
        self.e_ref = 0.04  #: Reference strain
        self.w = 0.56  #: Shape factor pylint: disable=invalid-name

        self.name = parameters.name  #: Muscle name
        self.m_id = parameters.m_id  #: Unique muscle id

        self.tendon_slack_length = self.dae.add_c('l_slack_' + self.m_id,
                                                  parameters.l_slack)
        self.optimal_fiber_length = self.dae.add_c('l_opt_' + self.m_id,
                                                   parameters.l_opt)
        self.max_contraction_velocity = self.dae.add_c('v_max_' + self.m_id,
                                                       parameters.v_max)
        self.max_isometric_force = self.dae.add_c('f_max_' + self.m_id,
                                                  parameters.f_max)
        self.pennation_angle = self.dae.add_c('pennation_' + self.m_id,
                                              parameters.pennation)

        self._td_to_sc = self.dae.add_c('td_to_sc_' + self.m_id,
                                        parameters.td_to_sc)
        self._td_from_sc = self.dae.add_c('td_from_sc_' + self.m_id,
                                          parameters.td_from_sc)

        if parameters.motiontype == 'flexor':
            self._motiontype = self.dae.add_c('motiontype_' + self.m_id,
                                              1)
        else:
            self._motiontype = self.dae.add_c('motiontype_' + self.m_id,
                                              1)

        #: MUSCLE STATES
        #: Muscle Contractile Length
        self._l_ce = self.dae.add_x('l_ce_' + self.m_id,
                                    parameters.l_ce0)
        #: Muscle Acitvation
        self._activation = self.dae.add_x('A_' + self.m_id,
                                          parameters.a0)

        #: INPUTS TO THE MODEL
        self._delta_length = self.dae.add_u('l_delta_' + self.m_id)
        #: External Muscle stimulation
        self._stim = self.dae.add_u('stim_' + self.m_id)

        def ode_rhs(self):
            """Muscle Model ODE rhs.
            Returns
            ----------
            ode_rhs: list<cas.SX>
                description
            """

            #: Bandpass l_ce
            #b, a = signal.butter(2, 50, 'low', analog=True)
            #l_ce_filt = signal.lfilter(b, a, self._l_ce.sym)

            l_ce_tol = cas.fmax(self._l_ce.sym, 0.0)
            _stim = cas.fmax(0.01, cas.fmin(self._stim.sym, 1.))

            #: Algrebaic Equation
            l_mtc = self._l_slack.val + self._l_opt.val + self._delta_length.sym
            l_se = l_mtc - l_ce_tol

            #: Muscle Acitvation Dynamics
            self._dA.sym = (
                _stim - self._activation.sym)/GeyerMuscle.tau_act

            #: Muscle Dynamics
            #: Series Force

            _f_se = (self._f_max.val * (
                (l_se - self._l_slack.val) / (
                    self._l_slack.val * self.e_ref))**2) * (
                        l_se > self._l_slack.val)

            _f_se_eqn = cas.Function(
                'tendon_force',
                [self._l_ce_sym, self._delta_length.sym],
                _f_se)

            #: Muscle Belly Force
            _f_be_cond = self._l_opt.val * (1.0 - self.w)

            _f_be = (
                (self._f_max.val * (
                    (l_ce_tol - self._l_opt.val * (1.0 - self.w)) / (
                        self._l_opt.val * self.w / 2.0))**2)) * (
                l_ce_tol <= _f_be_cond)

            #: Force-Length Relationship
            val = cas.fabs(
                (l_ce_tol - self._l_opt.val) / (self._l_opt.val * self.w))
            exposant = GeyerMuscle.c * val**3
            _f_l = cas.exp(exposant)

            #: Force Parallel Element
            _f_pe_star = (self._f_max.val * (
                (l_ce_tol - self._l_opt.val) / (self._l_opt.val * self.w))**2)*(
                    l_ce_tol > self._l_opt.val)

            #: Force Velocity Inverse Relation
            _f_v_eq = ((
                self._f_max.val * self._activation.sym * _f_l) + _f_pe_star)

            f_v_cond = cas.logic_and(
                _f_v_eq < self.tol, _f_v_eq > -self.tol)

            _f_v = cas.if_else(f_v_cond, 0.0, (_f_se + _f_be) / ((
                self._f_max.val * self._activation.sym * _f_l) + _f_pe_star))

            f_v = cas.fmax(0.0, cas.fmin(_f_v, 1.5))

            self._v_ce.sym = cas.if_else(
                f_v < 1.0, self._v_max.sym * self._l_opt.val * (
                    1.0 - f_v) / (1.0 + f_v * GeyerMuscle.K),
                self._v_max.sym*self._l_opt.val * (f_v - 1.0) / (
                    7.56 * GeyerMuscle.K *
                    (f_v - GeyerMuscle.N) + 1.0 - GeyerMuscle.N
                ))

            #: Active, Passive, Tendon Force Computation
            _f_v_ce = cas.if_else(
                self._v_ce.sym < 0.,
                (self._v_max.sym*self._l_opt.val - self._v_ce.sym) /
                (self._v_max.sym*self._l_opt.val + GeyerMuscle.K * self._v_ce.sym),
                GeyerMuscle.N + (GeyerMuscle.N - 1) * (
                    self._v_max.sym*self._l_opt.val + self._v_ce.sym
                ) / (
                    7.56 * GeyerMuscle.K * self._v_ce.sym - self._v_max.sym*self._l_opt.val
                ))

            self._a_force = self._activation.sym * _f_v_ce * _f_l * self._f_max.val
            self._p_force = _f_pe_star*_f_v - _f_be
            self._t_force = _f_se

            self._alg_tendon_force.sym = self._z_tendon_force.sym - self._t_force
            self._alg_active_force.sym = self._z_active_force.sym - self._a_force
            self._alg_passive_force.sym = self._z_passive_force.sym - self._p_force
            self._alg_v_ce.sym = self._z_v_ce.sym - self._v_ce.sym
            self._alg_l_mtc.sym = self._z_l_mtc.sym - l_mtc
            self._alg_dact.sym = self._z_dact.sym - self._dA.sym

            return True
