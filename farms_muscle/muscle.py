#!/usr/bin/env python3

"""Muscle abstract class"""

import abc
import six

# SIX is only suppported for python2 and will be deprecated soon


########## DATA CONTAINERS ##########
class MuscleLengthInfo(object):
    """Container for Muscle Length Info attributes.
    """

    def __init__(self):
        """Initialize"""
        super(MuscleLengthInfo, self).__init__()
        #: Attributes
        self._fiber_length = 0.0
        self._fiber_length_along_tendon = 0.0
        self._norm_fiber_length = 0.0

        self._tendon_length = 0.0
        self._norm_tendon_length = 0.0
        self._tendon_strain = 0.0

        self._pennation_angle = 0.0
        self._cos_pennation_angle = 0.0
        self._sin_pennation_angle = 0.0

    @property
    def fiber_length(self):
        """ Get the fiber lenght of the muscle.  """
        return self._fiber_length

    @fiber_length.setter
    def fiber_length(self, value):
        """ Set the fiber length  """
        self._fiber_length = value


class MuscleVelocityInfo(object):
    """Container for Muscle Velocity info attributes.
    """

    def __init__(self):
        """Initialize.
        """
        super(MuscleVelocityInfo, self).__init__()
        #: Attributes
        self.fiber_velocity = 0.0
        self.fiber_velocity_along_tendon = 0.0
        self.norm_fiber_velocity = 0.0
        self.tendon_velocity = 0.0
        self.norm_tendon_velocity = 0.0


@six.add_metaclass(abc.ABCMeta)
class MuscleDynamicsInfo():
    """Container for Muscle Dynamics info attributes.
    """

    def __init__(self):
        super(MuscleDynamicsInfo, self).__init__()
        self._activation = 0.0

        self._fiber_force = 0.0
        self._fiber_force_along_tendon = 0.0
        self._norm_fiber_force = 0.0
        self._active_fibre_force = 0.0
        self._passive_fiber_force = 0.0

        self._tendon_force = 0.0
        self._norm_tendon_force = 0.0

        self._fiber_stiffness = 0.0
        self._fiber_stiffness_along_tendon = 0.0
        self._tendon_stiffness = 0.0
        self._muscle_stiffness = 0.0

        self._fiber_active_power = 0.0
        self._fiber_passive_power = 0.0
        self._tendon_power = 0.0
        self._muscle_power = 0.0

    @property
    @abc.abstractmethod
    def tendon_force(self):
        """Get the tendon force.  """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def fiber_force(self):
        """Get the tendon force.  """
        raise NotImplementedError()


class MuscleState(object):
    """Container muscle state.
    * Activation : act
    * Contractile Lenght : l_se
    """

    def __init__(self):
        """ Initialization. """
        super(MuscleState, self).__init__()
        self._activation_val = 0.0
        self._fiber_length_val = 0.0

        self.activation = 0.0
        self.fiber_length = 0.0

    @property
    def activation(self):
        """ Activation of the muscle.  """
        return self._activation_val

    @activation.setter
    def activation(self, value):
        """
        Set the value of the activation.
        Parameters
        ----------
        value : <float>
            Activation value between [0, 1]
        """
        self._activation_val = value

    @property
    def fiber_length(self):
        """ Get contractile length of the muscle.  """
        return self._fiber_length_val

    @fiber_length.setter
    def fiber_length(self, value):
        """
        Set the value of the contractile_length.
        Parameters
        ----------
        value : <float>
            Contractile length
        """
        self._fiber_length_val = value


@six.add_metaclass(abc.ABCMeta)
class Muscle(MuscleLengthInfo, MuscleVelocityInfo, MuscleDynamicsInfo):
    """Muscle abstract class.
    """

    def __init__(self):
        """Initialize"""
        super(Muscle, self).__init__()

        #: Properties
        self._max_isometric_force = 0.0
        self._optimal_fiber_length = 0.0
        self._tendon_slack_length = 0.0
        self._pennation_angle = 0.0
        self._max_contraction_velocity = 0.0
        self.state = MuscleState()

    @property
    def max_isometric_force(self):
        """Get maximum isometric force.  """
        return self._max_isometric_force

    @max_isometric_force.setter
    def max_isometric_force(self, value):
        """
        Set maximum isometric force
        Parameters
        ----------
        value : <float>
            Value of maximum isometric force
        """
        if value < 0.0:
            raise AttributeError(
                'Maximum isometric force cannot be negative')
        else:
            self._max_isometric_force = value

    @property
    def optimal_fiber_length(self):
        """Get optimal fiber length  """
        return self._optimal_fiber_length

    @optimal_fiber_length.setter
    def optimal_fiber_length(self, value):
        """
        Set optimal fiber length
        Parameters
        ----------
        value : <float>
            Value of optimal fiber length
        """
        if value < 0.0:
            raise AttributeError(
                'Optimal fiber length cannot be negative')
        else:
            self._optimal_fiber_length = value

    @property
    def tendon_slack_length(self):
        """Get tendon slack length.  """
        return self._tendon_slack_length

    @tendon_slack_length.setter
    def tendon_slack_length(self, value):
        """
        Set tendon slack length
        Parameters
        ----------
        value : <float>
            Value of tendon slack length
        """
        if value < 0.0:
            raise AttributeError(
                'Tendon slack length cannot be negative')
        else:
            self._tendon_slack_length = value

    @property
    def pennation_angle(self):
        """Get pennation angle  """
        return self._pennation_angle

    @pennation_angle.setter
    def pennation_angle(self, value):
        """
        Set pennation angle.
        Parameters
        ----------
        value : <float>
            Value of pennation angle
        """
        self._pennation_angle = value

    @property
    def max_contraction_velocity(self):
        """Get maximum contraction velocity.  """
        return self._max_contraction_velocity

    @max_contraction_velocity.setter
    def max_contraction_velocity(self, value):
        """
        Set maximum contraction velocity.
        Parameters
        ----------
        value : <float>
            Value of maximum contraction velocity
        """
        if value < 0.0:
            raise AttributeError(
                'Maximum contraction velocity cannot be negative')
        else:
            self._max_contraction_velocity = value
