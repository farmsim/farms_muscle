"""Muscle abstract class"""

cdef class Muscle(object):
    """Muscle base class.
    """

    def __init__(self):
        """Initialize"""
        super(Muscle, self).__init__()
        self._name = 'muscle'
        self._l_slack = 0.0
        self._l_opt = 0.0
        self._f_max = 0.0
        self._v_max = 0.0
        self._pennation = 0.0

    ########## Generic Properties ##########
    @property
    def name(self):
        """ Get the muscle name  """
        return self._name

    @property
    def l_slack(self):
        """ Get the series elastic element length  """
        return self._l_slack

    @property
    def l_opt(self):
        """ Get the series elastic element length  """
        return self._l_opt

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
    cdef void c_ode_rhs(self) nogil:
        """ Function containing the ode of muscle model. """
        pass

    cdef void c_output(self) nogil:
        """ Function to update all the internal outputs of the muscle model. """
        pass
