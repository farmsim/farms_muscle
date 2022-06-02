# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

""" Generate muscle system. """
from farms_container import Container
from farms_container.table cimport Table
from cython.parallel import prange
import itertools
import farms_pylog as pylog
from farms_muscle.musculo_skeletal_parameters import MuscleParameters
from farms_muscle.muscle_factory import MuscleFactory
from farms_muscle.muscle cimport Muscle as CMuscle
import os
import yaml
import numpy as npcpf
cimport numpy as cnp
cimport cython
pylog.set_level('debug')

cdef class MuscleSystemGenerator(object):
    """ Generate Muscle System.
    """

    def __init__(self, container, num_muscles):
        """Initialize.

        Parameters
        ----------
        """
        super(MuscleSystemGenerator, self).__init__()
        #: Attributes
        #: ODE States
        self.states = <Table > container.muscles.add_table('states')
        self.dstates = <Table > container.muscles.add_table('dstates')
        #: Muscle parameters
        self.constants = <Table > container.muscles.add_table(
            'constants', table_type='CONSTANT')
        self.parameters = <Table > container.muscles.add_table('parameters')
        #: Input to each muscle
        self.activations = <Table > container.muscles.add_table('activations')
        #: Output of each muscle
        self.forces = <Table > container.muscles.add_table('forces')
        #: Secondary outputs
        self.outputs = <Table > container.muscles.add_table('outputs')
        #: Sensors
        self.Ia = <Table > container.muscles.add_table('Ia')
        self.II = <Table > container.muscles.add_table('II')
        self.Ib = <Table > container.muscles.add_table('Ib')

        #: _muscles dictionary
        self.muscles = {}

        #: Get the number of neurons in the model
        self.num_muscles = num_muscles

        self.c_muscles = cnp.ndarray((self.num_muscles,), dtype=CMuscle)

    def generate_muscles(self, container, config_data, time_step):
        """Generate all the muscles in the system .
        Instatiate a muscle model for each muscle in the config.
        Returns
        -------
        out : <bool>
            Return true if successfully created the muscles
        """
        # Get all the muscles data in the system
        muscles_data = config_data['muscles']

        # Factory to generate different muscles
        factory = MuscleFactory()

        # Generate the muscles
        for j, (name, muscle) in enumerate(muscles_data.items()):
            pylog.debug('Generating muscle model : {} of type {}'.format(
                name, muscle['model']))
            new_muscle = factory.gen_muscle(muscle['model'])
            # ADD DT
            self.muscles[muscle['name']] = new_muscle(
                container, MuscleParameters(
                    **muscle), dt=time_step)
            self.c_muscles[j] = <CMuscle > self.muscles[muscle['name']]
        return self.muscles

    #################### C-FUNCTIONS ####################
    cdef double[:] c_ode(self, double t, double[:] state):
        self.states.c_set_values(state)
        cdef unsigned int j
        # Loop over all the muscles
        for j in range(self.num_muscles):
            (< CMuscle > self.c_muscles[j]).c_ode_rhs()
        return self.dstates.c_get_values()

    cdef void c_update_outputs(self):
        cdef unsigned int j
        cdef CMuscle m
        # Loop over all the muscles
        for j in range(self.num_muscles):
            m = self.c_muscles[j]
            m.p_interface.c_compute_muscle_length()
            m.c_output()
            m.c_update_sensory_afferents()
            m.p_interface.c_apply_muscle_forces()
            m.p_interface.c_show_muscle()

    #################### C-WRAPPERS ####################
    def ode(self, t, cnp.ndarray[double, ndim=1] state):
        return self.c_ode(t, state)

    def py_update_outputs(self):
        self.c_update_outputs()

    @property
    def num_states(self):
        """Number of states in the muscle system  """
        return len(self.states)
