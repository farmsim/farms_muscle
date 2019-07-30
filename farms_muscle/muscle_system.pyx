# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

""" Generate muscle system. """
from farms_dae_generator.dae_generator import DaeGenerator
from farms_dae_generator.parameters cimport Parameters
from cython.parallel import prange
import itertools
import farms_pylog as pylog
from farms_muscle.musculo_skeletal_parameters import MuscleParameters
from farms_muscle.muscle_factory import MuscleFactory
from farms_muscle.muscle cimport Muscle as CMuscle
import os
import yaml
import numpy as np
cimport numpy as cnp
cimport cython
pylog.set_level('debug')

cdef class MuscleSystemGenerator(object):
    """ Generate Muscle System.
    """

    def __init__(self, dae, num_muscles):
        """Initialize.

        Parameters
        ----------
        """
        super(MuscleSystemGenerator, self).__init__()

        #: Attributes
        self.x = <Parameters > dae.x
        self.xdot = <Parameters > dae.xdot
        self.c = <Parameters > dae.c
        self.u = <Parameters > dae.u
        self.p = <Parameters > dae.p
        self.y = <Parameters > dae.y

        #: _muscles dictionary
        self.muscles = {}

        #: Get the number of neurons in the model
        self.num_muscles = num_muscles

        self.c_muscles = cnp.ndarray((self.num_muscles,), dtype=CMuscle)

    def generate_muscles(self, dae, config_data):
        """Generate all the muscles in the system .
        Instatiate a muscle model for each muscle in the config.
        Returns
        -------
        out : <bool>
            Return true if successfully created the muscles
        """

        #: Get all the muscles data in the system
        muscles_data = config_data['muscles']

        #: Factory to generate different muscles
        factory = MuscleFactory()

        #: Generate the muscles
        for j, (name, muscle) in enumerate(muscles_data.items()):
            pylog.debug('Generating muscle model : {} of type {}'.format(
                name, muscle['model']))
            new_muscle = factory.gen_muscle(muscle['model'])
            self.muscles[muscle['name']] = new_muscle(dae,
                                                      MuscleParameters(**muscle))
            self.c_muscles[j] = < CMuscle > self.muscles[muscle['name']]
        return self.muscles

    #################### C-FUNCTIONS ####################
    cdef double[:] c_ode(self, double t, double[:] state):
        self.x.c_set_values(state)
        cdef unsigned int j
        cdef CMuscle m
        #: Loop over all the muscles
        for j in range(self.num_muscles):
            m = self.c_muscles[j]
            m.c_ode_rhs()
        return self.xdot.c_get_values()

    cdef void c_update_outputs(self):
        cdef unsigned int j
        cdef CMuscle m
        #: Loop over all the muscles
        for j in range(self.num_muscles):
            m = self.c_muscles[j]
            m.p_interface.c_compute_muscle_length()
            m.c_output()
            m.p_interface.c_apply_muscle_forces()
            m.p_interface.c_show_muscle(VISUALIZATION=True)

    #################### C-WRAPPERS ####################

    def ode(self, t, cnp.ndarray[double, ndim=1] state):
        return self.c_ode(t, state)

    def py_update_outputs(self):
        self.c_update_outputs()
