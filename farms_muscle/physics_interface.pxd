""" Template for interface between physics engine and muscle system. """

from farms_container.parameter cimport Parameter as cparameter

cdef class PhysicsInterface(object):
    """Interface between physics engine and muscle model
    """
    #: Properties
    cdef:
         str engine
         cparameter lmtu #: Parameter of muscle tendon unit
         cparameter force #: Parameter of Muscle-Tendon force
         cparameter stim #: Parameter of Muscle stimulation

    #################### C-FUNCTIONS ####################
    cdef:
        void c_compute_muscle_length(self)
        void c_apply_muscle_forces(self)
        void c_show_muscle(self)
