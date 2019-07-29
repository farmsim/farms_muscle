""" Template for interface between physics engine and muscle system. """

from farms_dae_generator.param cimport Param as cparam

cdef class PhysicsInterface(object):
    """Interface between physics engine and muscle model
    """
    #: Properties
    cdef:
         str engine
         cparam lmtu #: Parameter of muscle tendon unit
         cparam force #: Parameter of Muscle-Tendon force
        
    #################### C-FUNCTIONS ####################    
    cdef:
        void c_compute_muscle_length(self)
        void c_apply_muscle_forces(self)
