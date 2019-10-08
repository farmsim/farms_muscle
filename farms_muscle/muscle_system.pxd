from farms_container.table cimport Table
from farms_muscle.muscle cimport Muscle as CMuscle

cdef class MuscleSystemGenerator(object):
    cdef:
        CMuscle[:] c_muscles
        readonly dict muscles

        #: ODE States
        Table states
        Table dstates
        #: Muscle parameters
        Table constants
        Table parameters
        #: Input to each muscle
        Table activations
        #: Output of each muscle
        Table forces
        Table outputs #: Secondary outputs
        #: Sensors
        Table Ia
        Table II
        Table Ib

        unsigned int num_muscles

    cdef:
        double[:] c_ode(self, double t, double[:] state)
        void c_update_outputs(self)
