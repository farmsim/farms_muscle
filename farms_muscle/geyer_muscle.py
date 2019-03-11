"""Implementation of Geyer muscle model."""

from muscle import Muscle
import numpy as np


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
