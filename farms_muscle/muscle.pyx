"""Muscle abstract class"""

import farms_pylog as pylog

cdef class Muscle(object):
    """Muscle base class.
    """

    def __init__(self, name, dt, physics_engine):
        """Initialize"""
        super(Muscle, self).__init__()
        self._physics_engine = physics_engine
        self._name = name
        self._f_max = 0.0
        self._v_max = 0.0
        self._pennation = 0.0

        self.dt = dt

    ########## Generic Properties ##########
    @property
    def name(self):
        """ Get the muscle name  """
        return self._name

    @property
    def l_slack(self):
        """ Get the series elastic element length  """
        return self._l_slack.c_get_value()

    @property
    def l_opt(self):
        """ Get the series elastic element length  """
        return self._l_opt.c_get_value()

    @property
    def f_max(self):
        """ Get the series elastic element length  """
        return self._f_max

    @property
    def v_max(self):
        """ Get the series elastic element length  """
        return self._v_max

    @property
    def pennation_angle(self):
        """ Get the pennation angle.  """
        return self._pennation

    #################### C-FUNCTIONS ####################
    cdef void c_compute_Ia(self) nogil:
        """ Function implementing the computation of Ia muscle afferent. """
        pass

    cdef void c_compute_II(self) nogil:
        """ Function implementing the computation of II muscle afferent. """
        pass

    cdef void c_compute_Ib(self) nogil:
        """ Function implementing the computation of Ib muscle afferent. """
        pass

    cdef void c_update_sensory_afferents(self) nogil:
        """ Function to update all sensory afferents. """
        pass

    cdef void c_ode_rhs(self) nogil:
        """ Function containing the ode of muscle model. """
        pass

    cdef void c_output(self) nogil:
        """ Function to update all the internal outputs of the muscle model. """
        pass
