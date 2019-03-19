"""Generation of Multiple Muscle Models in the System."""
import itertools
import os
from collections import OrderedDict

import casadi as cas
import farms_pylog as biolog
import numpy as np
import yaml

from farms_muscle.muscle_factory import MuscleFactory
from farms_casadi_dae.casadi_dae_generator import CasadiDaeGenerator
from farms_muscle.parameters import MuscleParameters

biolog.set_level('debug')


class MuscleSystem(OrderedDict):
    """Generate Muscle Models for the the animal.
    Inherit from Ordered Dictionary so the muscles can directly be
    stored within the class instead of a new container.
    """

    def __init__(self, muscle_config_path, dae):
        """Initialize.

        Parameters
        ----------
        muscle_config : <str>
            Path to muscle system configuration path.
        """
        super(MuscleSystem, self).__init__()
        self.muscle_config_path = muscle_config_path
        self.muscle_config = None  #: Muscle Configuration data
        #: Attributes
        self.activations = {}  #: Muscle activations in the system
        self.dae = dae
        self.opts = {}  #: Integration parameters
        self.fin = {}
        self.num = 0  # : Number of muscles
        self.integrator = None

        #: Methods
        self.load_config_file()
        self.generate_muscles()

    def load_config_file(self):
        """Load the animal configuration file"""
        try:
            stream = open(
                os.path.realpath(self.muscle_config_path), 'r')
            self.muscle_config = yaml.load(stream)
            biolog.info('Successfully loaded the file : {}'.format(
                os.path.split(self.muscle_config_path)[-1]))
            return
        except ValueError:
            biolog.error('Unable to read the file {}'.format(
                self.muscle_config_path))
            raise ValueError()

    def generate_muscles(self):
        """ Generate muscles. """
        factory = MuscleFactory()
        for _, muscle in sorted(self.muscle_config['muscles'].items()):
            _muscle = factory.gen_muscle(muscle['model'].lower())
            self[muscle['name']] = _muscle(
                self.dae,
                MuscleParameters(**muscle))
            #: Update the muscle count
            self.num += 1
            biolog.debug('Created muscle {}'.format(
                self[muscle['name']].name))

    def generate_opts(self, opts):
        """ Generate options for integration."""
        if opts is not None:
            self.opts = opts
        else:
            self.opts = {'tf': 0.001,
                         'jit': True,
                         "enable_jacobian": True,
                         "print_time": False,
                         "print_stats": False,
                         "reltol": 1e-2,
                         "abstol": 1e-2,
                         "max_num_steps": 10}

    #: pylint: disable=invalid-name
    def setup_integrator(self,
                         integration_method='cvodes',
                         opts=None):
        """Setup casadi integrator."""

        #: Generate Options for integration
        self.generate_opts(opts)
        #: Initialize states of the integrator
        self.fin['x0'] = self.dae.x
        self.fin['p'] = self.dae.u +\
            self.dae.p + self.dae.c
        self.fin['z0'] = self.dae.z
        self.fin['rx0'] = cas.DM([])
        self.fin['rp'] = cas.DM([])
        self.fin['rz0'] = cas.DM([])
        #: Set up the integrator
        self.integrator = cas.integrator('muscles',
                                         integration_method,
                                         self.dae.generate_dae(),
                                         self.opts)
        return self.integrator

    def step(self):
        """Step integrator."""
        self.fin['p'][:] = list(itertools.chain(*self.dae.params))
        #: Step the integrator
        res = self.integrator.call(self.fin)
        _xf = res['xf'].full()[:, 0]
        #: Update states
        self.dae.x.set_all_val_array(_xf)
        # self._update_states(res['xf'].reshape((self.num, 2)).full())
        #: Restart the state for next time step
        self.fin['x0'][:] = _xf
        # self.fin['z0'][:] = res['zf'].full()[:, 0]
        # self.fin['rx0'] = res['rxf']
        # self.fin['rz0'] = res['rzf']
        return res

    #: Deprecated
    def _update_states(self, results):
        """ Update all the states of the muscle """
        list(map(self._update_internal_muscle_state, self.values(),
                 results))

    def _update_internal_muscle_state(self, muscle, result):
        """ Update the internal state of the muscle every time the
        step function is called"""
        muscle.state.activation = result[1]
        muscle.state.fiber_length = result[0]


def main():
    """Main test function."""
    dae = CasadiDaeGenerator()
    muscles = MuscleSystem('conf/pendulum_config.yaml', dae)
    muscles.dae.print_dae()
    muscles.setup_integrator()
    muscle_inputs = muscles.dae.u
    for j in range(1, 8000):
        muscle_inputs.set_val('l_delta_1', 0.0)
        muscle_inputs.set_val('stim_1', 0.15)
        res = muscles.step()


if __name__ == '__main__':
    main()
