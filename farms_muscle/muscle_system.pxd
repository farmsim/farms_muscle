from farms_dae_generator.parameters cimport Parameters
from farms_muscle.muscle cimport Muscle as CMuscle

cdef class MuscleSystemGenerator(object):
    cdef:
        CMuscle[:] c_muscles
        readonly dict muscles
        Parameters x
        Parameters xdot
        Parameters c
        Parameters u
        Parameters p
        Parameters y

        unsigned int num_muscles

    cdef:
        double[:] c_ode(self, double t, double[:] state)
        void c_update_outputs(self)
