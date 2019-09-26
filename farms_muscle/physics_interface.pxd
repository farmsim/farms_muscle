""" Template for interface between physics engine and muscle system. """

from farms_dae.param cimport Param as cparam

cdef class PhysicsInterface(object):
    """Interface between physics engine and muscle model
    """
    #: Properties
    cdef:
         str engine
         cparam lmtu #: Parameter of muscle tendon unit
         cparam force #: Parameter of Muscle-Tendon force
         cparam stim #: Parameter of Muscle stimulation
        
    #################### C-FUNCTIONS ####################    
    cdef:
        void c_compute_muscle_length(self)
        void c_apply_muscle_forces(self)
        void c_show_muscle(self)
