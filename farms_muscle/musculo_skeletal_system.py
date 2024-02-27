""" MusculoSkeletalSystem class """

import os
import sys

import farms_pylog as pylog
import numpy as np
import yaml
from farms_core.io.yaml import read_yaml
from farms_core.model.options import MuscleOptions
from scipy.integrate import ode

from farms_muscle.muscle_system import MuscleSystemGenerator

pylog.set_level('debug')


class MusculoSkeletalSystem():
    """ Class to generate musculo-skeletal module.
    1. Muscle Generation
    2. Joint Generation
    3. Muscle-Joint Generation : Binding muscles and joints"""

    def __init__(self, container, time_step, muscles_options):
        """ Initialize the joints and muscles.
        Need to initialize the class with a valid json file"""

        # Create muscles namespace in the container
        self.container = container
        self.container.add_namespace('muscles')
        self.muscles_sys = None
        self.integrator = None

        # Generate the muscles in the system
        self.generate_muscles(muscles_options, time_step)

    @classmethod
    def from_file(cls, container, time_step, file_path):
        """ Load musculo_skeletal_system from config file """
        if not os.path.isfile(file_path):
            pylog.error('Wrong config path .....')
            raise RuntimeError()
        # Read data from the config file
        muscle_config = read_yaml(file_path)
        muscles_options = [
            MuscleOptions(
                muscle_name=name,
                muscle_type=muscle['model'],
                max_force=muscle['f_max'],
                max_velocity=muscle['v_max'],
                optimal_fiber=muscle['l_opt'],
                tendon_slack=muscle['l_slack'],
                pennation_angle=muscle['pennation'],
                waypoints=[
                    [waypoint[0]['link'], waypoint[1]['point']]
                    for waypoint in muscle['waypoints']
                ]
            )
            for name, muscle in muscle_config['muscles'].items()
        ]
        return cls(container, time_step, muscles_options)

    def generate_muscles(self, muscles_options, time_step):
        """This function creates muscle objects based on the config file.
        The function stores the created muscle objects in a dict."""
        num_muscles = len(muscles_options)
        self.muscle_sys = MuscleSystemGenerator(
            self.container, num_muscles
        )
        self.muscles = self.muscle_sys.generate_muscles(
            self.container, muscles_options, time_step
        )

    def setup_integrator(self, x0=None, integrator='dopri5', atol=1e-6,
                         rtol=1e-6, method='bdf'):
        """Setup system."""
        if self.muscle_sys.num_states > 0:
            self.integrator = ode(self.muscle_sys.ode).set_integrator(
                integrator,
                method=method,
                verbosity=0
            )
        else:
            self.integrator = None

        # Initialize DAE
        if x0 is None:
            x0 = np.ones((self.muscle_sys.num_states,))*0.05
            if self.muscle_sys.num_states == 2*len(self.muscles):
                for j, muscle in enumerate(self.muscles.values()):
                    x0[2*j] = muscle.compute_initial_l_ce()
            else:
                # TODO: Refactor the code
                # This is needed to initialize lmtu for rigid tendon
                # model.
                for j, muscle in enumerate(self.muscles.values()):
                    muscle.compute_initial_l_ce()
        self.integrator.set_initial_value(x0, 0.0)
        # Initialize the forces
        self.muscle_sys.py_update_outputs()

    def step(self, dt=1e-3):
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
        # Step the musculo_skeletal_system.
        if self.integrator:
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
