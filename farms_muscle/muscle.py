#!/usr/bin/env python3

"""Muscle abstract class"""

from abc import ABC, abstractmethod


class Muscle(ABC):
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
