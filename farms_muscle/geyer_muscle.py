"""Implementation of Geyer muscle model."""

from farms_muscle.muscle import Muscle
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
    E_REF = 0.04  #: Reference strain
    W = 0.56  #: Shape factor pylint: disable=invalid-name
    tau_act = 0.01  # Time constant for the activation function
    F_per_m2 = 300000  # Force per m2 of muscle PCSA
    density = 1060

    def __init__(self, dae, parameters):
        """This function initializes the muscle model.
        A default muscle name is given as muscle

        Parameters
        ----------
        dae : <CasadiDaeGenerator>
            Instance of CasadiDaeGenerator class
        parameters : <MuscleParameters>
            Instance of MuscleParameters class

        Returns:
        -------
        Muscle : <Muscle>
            Returns an instance of class Muscle

        """
        super(GeyerMuscle, self).__init__()

        self.tol = 1e-6  #: Tolerance
        self.dae = dae

        self.name = parameters.name  #: Muscle name
        self.m_id = parameters.m_id  #: Unique muscle id

        self._l_slack = self.dae.add_c('l_slack_' + self.m_id,
                                       parameters.l_slack, param_type='val')

        self._l_opt = self.dae.add_c('l_opt_' + self.m_id,
                                     parameters.l_opt, param_type='val')
        self._v_max = self.dae.add_c('v_max_' + self.m_id,
                                     parameters.v_max, param_type='val')
        self._f_max = self.dae.add_c('f_max_' + self.m_id,
                                     parameters.f_max, param_type='val')
        self._pennation = self.dae.add_c('pennation_' + self.m_id,
                                         parameters.pennation, param_type='val')

        self._td_to_sc = self.dae.add_c('td_to_sc_' + self.m_id,
                                        parameters.td_to_sc, param_type='val')
        self._td_from_sc = self.dae.add_c('td_from_sc_' + self.m_id,
                                          parameters.td_from_sc, param_type='val')

        if parameters.motiontype == 'flexor':
            self._motiontype = self.dae.add_c('motiontype_' + self.m_id,
                                              1, param_type='val')
        else:
            self._motiontype = self.dae.add_c('motiontype_' + self.m_id,
                                              1, param_type='val')

        #: MUSCLE STATES
        #: Muscle Contractile Length
        self._l_ce = self.dae.add_x('l_ce_' + self.m_id,
                                    parameters.l_ce0)
        #: Muscle Activation
        self._activation = self.dae.add_x('A_' + self.m_id,
                                          parameters.a0)

        #: INPUTS TO THE MODEL
        self._delta_length = self.dae.add_u('l_delta_' + self.m_id)
        #: External Muscle stimulation
        self._stim = self.dae.add_u('stim_' + self.m_id)

        #: Derivatives
        self._v_ce = self.dae.add_ode('v_ce_' + self.m_id, 0.0)
        self._dA = self.dae.add_ode('dA_' + self.m_id, 0.0)

        #: Setup all functions
        self._setup_all()

        #: Generate the ODE
        self.ode_rhs()

    ########## SETUP METHODS ##########

    def _setup_tendon_force(self):
        """ Setup the casadi equations for tendon force. """
        _l_se = cas.SX.sym('l_se')
        _strain = (_l_se - self._l_slack.param) / (self._l_slack.param)
        _eqn = (self._f_max.param * (_strain / GeyerMuscle.E_REF)**2) * (
            _l_se > self._l_slack.param)
        self._tendon_force = cas.Function(
            'tendon_force', [_l_se],
            [_eqn], ['l_se'], ['f_se'])

    def _setup_parallel_star_force(self):
        """ Setup the casadi equations for parallell star pe*  """
        _l_ce = cas.SX.sym('l_ce')
        _eqn = (self._f_max.param * (
            (_l_ce - self._l_opt.param) / (self._l_opt.param * GeyerMuscle.W))**2)*(
            _l_ce > self._l_opt.param)
        self._parallel_star_force = cas.Function(
            'parallel_star_force', [_l_ce],
            [_eqn], ['l_ce'], ['f_pe_star'])

    def _setup_belly_force(self):
        """ Setup the casadi equations for belly force  """
        _l_ce = cas.SX.sym('l_ce')
        _f_be_cond = self._l_opt.param * (1.0 - GeyerMuscle.W)
        _num = _l_ce - self._l_opt.param * (1.0 - GeyerMuscle.W)
        _den = self._l_opt.param * GeyerMuscle.W * 0.5
        _eqn = self._f_max.param * ((_num/_den)**2) * (_l_ce <= _f_be_cond)
        self._belly_force = cas.Function(
            'belly_force', [_l_ce],
            [_eqn], ['l_ce'], ['f_be'])

    def _setup_activation_rate(self):
        """ Define the change in activation. dA/dt. """
        _act = cas.SX.sym('act')
        _stim = cas.SX.sym('stim')
        _stim_range = cas.fmax(0.01, cas.fmin(_stim, 1.))
        _eqn = (_stim_range - _act)/GeyerMuscle.tau_act
        self._d_act = cas.Function(
            'activation_rate', [_act, _stim], [_eqn],
            ['act', 'stim'], ['dA/dt'])

    def _setup_force_length(self):
        """ Define the force length relationship. """
        _l_ce = cas.SX.sym('l_ce')
        val = cas.fabs(
            (_l_ce - self._l_opt.param) / (self._l_opt.param * GeyerMuscle.W))
        exposant = GeyerMuscle.c * val**3
        _eqn = cas.exp(exposant)
        self._force_length = cas.Function(
            'force_length', [_l_ce],
            [_eqn], ['l_ce'], ['f_l'])

    def _setup_force_velocity(self):
        """ Define the force velocity relationship. """
        _v_ce = cas.SX.sym('v_ce')
        _f_v_ce_eqn_1 = (
            self._v_max.param - _v_ce)/(self._v_max.param + GeyerMuscle.K*_v_ce)
        _f_v_ce_eqn_2 = GeyerMuscle.N + ((GeyerMuscle.N - 1)*(
            self._v_max.param + _v_ce)/(7.56*GeyerMuscle.K*_v_ce - self._v_max.param))
        _eqn = cas.if_else(_v_ce >= 0.0, _f_v_ce_eqn_1, _f_v_ce_eqn_2)
        self._force_velocity = cas.Function(
            'force_velocity', [_v_ce],
            [_eqn], ['v_ce'], ['f_v'])

    def _setup_force_velocity_from_force(self):
        """ Define the force velocity relationship from forces."""
        _f_se = cas.SX.sym('f_se')
        _f_be = cas.SX.sym('f_be')
        _act = cas.SX.sym('act')
        _f_l = cas.SX.sym('f_l')
        _f_pe_star = cas.SX.sym('f_pe_star')
        _f_v_ce_eqn_num = (_f_se + _f_be)
        _f_v_ce_eqn_den = (self._f_max.param*_act*_f_l) + _f_pe_star
        #: Check these TOLERANCES
        _f_v_ce_eqn = cas.if_else(
            cas.logic_and(
                _f_v_ce_eqn_den < self.tol,
                _f_v_ce_eqn_den > -self.tol),
            0.0,
            _f_v_ce_eqn_num/_f_v_ce_eqn_den)
        _eqn = cas.fmax(0.0, cas.fmin(1.5, _f_v_ce_eqn))
        self._force_velocity_from_force = cas.Function(
            'force_velocity_from_forces',
            [_f_se, _f_be, _act, _f_l, _f_pe_star],
            [_eqn], ['_f_se', '_f_be', '_act', '_f_l', '_f_pe_star'],
            ['f_v'])

    def _setup_contractile_velocity(self):
        """ Define the contractile velocity."""
        _f_v = cas.SX.sym('f_v')
        _v_ce_1 = self._v_max.param*self._l_opt.param * \
            (1. - _f_v)/(1. + _f_v*GeyerMuscle.K)
        _v_ce_2 = self._v_max.param*self._l_opt.param * \
            (_f_v - 1.0)/(7.56*GeyerMuscle.K *
                          (_f_v - GeyerMuscle.N) + 1. - GeyerMuscle.N)
        _eqn = cas.if_else(_f_v < 1.0, _v_ce_1, _v_ce_2)
        self._contractile_velocity = cas.Function(
            'contractile_velocity',
            [_f_v],
            [_eqn], ['_f_v'],
            ['v_ce'])

    def _setup_all(self):
        """Setup all the internal functions."""
        self._setup_tendon_force()
        self._setup_parallel_star_force()
        self._setup_belly_force()
        self._setup_activation_rate()
        self._setup_force_length()
        self._setup_force_velocity()
        self._setup_force_velocity_from_force()
        self._setup_contractile_velocity()

    #: Properties
    #: LengthInfo
    @property
    def fiber_tendon_length(self):
        """ Get the length of muscle tendon unit.  """
        #### CHECK THIS ####
        return self._l_opt.param*self._pennation.param + self._l_slack.param \
            + self._delta_length.param

    @property
    def fiber_length(self):
        """ Get the fiber length of the muscle.  """
        return self._l_ce.param

    @property
    def tendon_length(self):
        """ Get the length of series tendon length  """
        return self.fiber_tendon_length - self.fiber_length

    @property
    def pennation_angle(self):
        """ Get the pennation angle.  """
        return cas.acosh(self._pennation.param)

    #: Velocity Info
    @property
    def fiber_velocity(self):
        """ Get the fiber velocity.  """
        return self._contractile_velocity(self.force_velocity)

    #: Dynamics Info
    @property
    def activation(self):
        """ Get the muscle activation.  """
        return self._activation.param

    @property
    def parallel_star_force(self):
        """ Get the force in parallel element*  """
        return self._parallel_star_force(self.fiber_length)

    @property
    def belly_force(self):
        """ Get the force in muscle belly.  """
        return self._belly_force(self.fiber_length)

    @property
    def tendon_force(self):
        """ Get the tendon force. """
        return self._tendon_force(self.tendon_length)

    @property
    def fiber_force(self):
        """  """
        return None

    @property
    def force_velocity(self):
        """ Get the force from force velocity relationship.  """
        return 0.0

    def ode_rhs(self):
        """Muscle Model ODE rhs.
            Returns
            ----------
            ode_rhs: list<cas.SX>
                description
        """

        l_ce_tol = cas.fmax(self._l_ce.param, 0.0)

        #: Algrebaic Equation
        l_mtc = self._l_slack.param + self._l_opt.param * \
            self._pennation.param + self._delta_length.param
        l_se = l_mtc - l_ce_tol

        #: Muscle Acitvation Dynamics
        self._dA.param = self._d_act(self._activation.param,
                                     self._stim.param)

        # #: Muscle Dynamics
        # #: Series Force
        _f_se = self._tendon_force(l_se)

        # #: Muscle Belly Force
        _f_be = self._belly_force(l_ce_tol)

        # #: Force-Length Relationship
        _f_l = self._force_length(l_ce_tol)

        # #: Force Parallel Element
        _f_pe_star = self._parallel_star_force(l_ce_tol)

        # #: Force Velocity Inverse Relation
        _f_v = self._force_velocity_from_force(_f_se,
                                               _f_be,
                                               self._activation.param,
                                               _f_l,
                                               _f_pe_star)

        self._v_ce.sym = self._contractile_velocity(_f_v)
        return True

    def update(self, stim, delta_length):
        """ Applies force to the joint and computes change in muscle length """
        self._stim.param = stim
        self._delta_length.param = delta_length

    def initialize_muscle_length(self):
        """ Initialize muscle length."""
        delta_length = 0.0

        #: Algrebaic Equation
        l_mtc = self._l_slack.param + \
            self._l_opt.param + delta_length

        if l_mtc < (self._l_slack.param + self._l_opt.param):
            l_ce = self._l_opt.param
            l_se = l_mtc - l_ce
        else:
            if self._l_opt.param * GeyerMuscle.W + GeyerMuscle.E_REF * self._l_slack.param != 0.0:
                l_se = self._l_slack.param * ((self._l_opt.param * GeyerMuscle.W + GeyerMuscle.E_REF * (
                    l_mtc - self._l_opt.param)) / (
                        self._l_opt.param * GeyerMuscle.W + GeyerMuscle.E_REF * self._l_slack.param))
            else:
                l_se = self._l_slack.param

            l_ce = l_mtc - l_se

        #: Initialize the muscle length
        self._l_ce.param = l_ce


def main():
    """Main function for testing."""
    from parameters import MuscleParameters
    from farms_casadi_dae import casadi_dae_generator
    parameters = MuscleParameters()
    parameters.m_id = '1'
    dae = casadi_dae_generator.CasadiDaeGenerator()
    muscle = GeyerMuscle(dae, parameters)
    muscle.ode_rhs()
    muscle.dae.print_dae()


if __name__ == '__main__':
    main()
