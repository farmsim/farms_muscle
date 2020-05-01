""" MusculoSkeletalSystem class """

import yaml
import os
import sys
import numpy as np
from farms_muscle.muscle_system import MuscleSystemGenerator
from scipy.integrate import ode
import farms_pylog as pylog
pylog.set_level('debug')


class MusculoSkeletalSystem(object):
    """ Class to generate musculo-skeletal module.
    1. Muscle Generation
    2. Joint Generation
    3. Muscle-Joint Generation : Binding muscles and joints"""

    def __init__(self, container, config_path=None, opts=None):
        """ Initialize the joints and muscles.
        Need to initialize the class with a valid json file"""
        if config_path is None:
            pylog.error('Missing config file .....')
            raise RuntimeError()
        elif not os.path.isfile(config_path):
            pylog.error('Wrong config path .....')
            raise RuntimeError()

        #: Create muscles namespace in the container
        self.container = container
        self.container.add_namespace('muscles')
        self.muscles = {}
        self.muscles_sys = None
        self.integrator = None

        #: Load config file
        config_data = MusculoSkeletalSystem.load_config_file(
            config_path)

        #: Check for if the config file has muscles and joints defined
        self.is_muscles = True
        self.is_joints = True

        #: Generate the muscles in the system
        self.muscles = self.generate_muscles(config_data)

    @staticmethod
    def load_config_file(config_path):
        """Load the muscle configuration file"""

        try:
            stream = open(
                os.path.realpath(config_path), 'r')
            config_data = yaml.safe_load(stream)
            pylog.info('Successfully loaded the file : {}'.format(
                os.path.split(config_path)[-1]))
            return config_data
        except ValueError:
            pylog.error('Unable to read the file {}'.format(
                config_path))
            raise ValueError()

    def generate_muscles(self, config_data):
        """This function creates muscle objects based on the config file.
        The function stores the created muscle objects in a dict."""
        num_muscles = len(config_data['muscles'])
        self.muscle_sys = MuscleSystemGenerator(self.container, num_muscles)
        self.muscles = self.muscle_sys.generate_muscles(
            self.container, config_data)
        return self.muscles

    def setup_integrator(self, x0=None, integrator='dopri5', atol=1e-6,
                         rtol=1e-6, method='bdf'):
        """Setup system."""
        self.integrator = ode(self.muscle_sys.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol,
            verbosity=1)

        #: Initialize DAE
        if x0 is None:
            x0 = np.ones((2*len(self.muscles),))*0.05
            for j, muscle in enumerate(self.muscles.values()):
                x0[2*j] = muscle.compute_initial_l_ce()
            self.integrator.set_initial_value(
                self.container.muscles.states.values, 0.0
            )
        else:
            self.integrator.set_initial_value(x0, 0.0)

    def step(self, dt=0.001):
        """ Step the complete bio-mechanical system.
        Parameters
        ----------
        self: type
            description
        dt: float
            Integration time step
        muscle_stim: dict
            Dictionary of muscle activations
        """
        #: Step the musculo_skeletal_system.
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t + dt)
        self.integrator.integrate(self.integrator.t + dt)
        self.muscle_sys.py_update_outputs()

    def print_system(self):
        """
        Print the muscles and joints generated in the musculoskeletalsystem.
        """
        pylog.info('Muscles created in system : ')
        for j, muscle in enumerate(self.muscle_sys.muscles.values()):
            print(('{}. {}'.format(j, muscle.name)))

# ----------------------------------------------------------------------------


def main():
    from farms_container import Container
    container = Container()
    m = MusculoSkeletalSystem(
        config_path='../../mouse/webots/controllers/simple_mouse_world/conf/simple_mouse_world.yaml')


if __name__ == '__main__':
    main()
