"""Generation of Multiple Muscle Models in the System."""
import itertools
import os
from collections import OrderedDict

import casadi as cas
import farms_pylog as biolog
import numpy as np
import yaml

from farms_casadi_dae.casadi_dae_generator import CasadiDaeGenerator
from .geyer_muscle import GeyerMuscle
from .parameters import MuscleParameters

biolog.set_level('debug')


class MuscleSystem(object):
    """Generate Muscle Models for the the animal.
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
        self.muscles = OrderedDict()  #: Muscle models in the system
        self.activations = {}  #: Muscle activations in the system
        self.dae = dae
        self.opts = {}  #: Integration parameters
        self.fin = {}
        self.integrator = None

        #: Methods
        self.load_config_file()
        self.generate_muscles()
        self.generate_muscle_position_container()

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
        for _, muscle in sorted(self.muscle_config['muscles'].items()):
            if muscle['model'].lower() == 'geyer':
                self.muscles[muscle['name']] = GeyerMuscle(
                    self.dae,
                    MuscleParameters(**muscle))
            biolog.debug('Created muscle {}'.format(
                self.muscles[muscle['name']].name))

    def generate_opts(self, opts):
        """ Generate options for integration."""
        if opts is not None:
            self.opts = opts
        else:
            self.opts = {'tf': 0.001,
                         'jit': False,
                         "enable_jacobian": True,
                         "print_time": True,
                         "print_stats": True,
                         "reltol": 1e-6,
                         "abstol": 1e-6,
                         "max_num_steps": 100}

    def generate_muscle_position_container(self):
        """Generate muscle position container. """
        for name in list(self.muscles.keys()):
            self.activations[name] = 0.05

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

    def step(self, delta_length):
        """Step integrator."""
        for name, muscle in self.muscles.items():
            muscle.update(1., delta_length[name])
        self.fin['p'][:] = list(itertools.chain(*self.dae.params))
        res = self.integrator.call(self.fin)
        self.fin['x0'][:] = res['xf'].full()[:, 0]
        self.fin['z0'][:] = res['zf'].full()[:, 0]
        self.fin['rx0'] = res['rxf']
        self.fin['rz0'] = res['rzf']
        return res


def main():
    """Main test function."""
    dae = CasadiDaeGenerator()
    muscles = MuscleSystem('conf/pendulum_config.yaml', dae)
    muscles.dae.print_dae()
    muscles.setup_integrator()
    res = muscles.step(0.0)
    print(res['xf'].full()[:, 0])


if __name__ == '__main__':
    main()
