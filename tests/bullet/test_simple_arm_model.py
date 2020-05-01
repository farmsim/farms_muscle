""" Test human model in pybullet. """

from farms_models.utils import get_farms_model_path
from animals.generic.bullet_simulation import BulletSimulation
from IPython import embed
import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time
import farms_pylog as pylog
import numpy as np
import os
from farms_sdf.units import SimulationUnitScaling
from farms_container import Container
import pandas as pd
import matplotlib.pyplot as plt
from farms_muscle.bullet_interface import BulletInterface
pylog.set_level('debug')


class SimpleArmSimulation(BulletSimulation):
    """SimpleArm Simulation Class
    """

    def __init__(self, container, sim_options):
        super(SimpleArmSimulation, self).__init__(
            container, SimulationUnitScaling(), **sim_options
        )
        u = container.muscles.activations
        self.muscle_params = {}
        self.muscle_excitation = {}
        for muscle in self.muscles.muscles.keys():
            self.muscle_params[muscle] = u.get_parameter(
                'stim_{}'.format(muscle)
            )
            self.muscle_excitation[muscle] = p.addUserDebugParameter(
                "flexor {}".format(muscle), 0, 1, 0.05
            )

    def controller_to_actuator(self):
        """ Implementation of abstractmethod. """

        for muscle in self.muscles.muscles.keys():
            self.muscle_params[muscle].value = p.readUserDebugParameter(
                self.muscle_excitation[muscle]
            )

    def feedback_to_controller(self):
        """ Implementation of abstractmethod. """
        pass

    def update_parameters(self):
        """ Implementation of abstractmethod. """
        pass

    def optimization_check(self):
        """ Implementation of abstractmethod. """
        pass


def main():
    """ Main """

    sim_options = {"headless": False,
                   "model": os.path.join(
                       get_farms_model_path(),
                       "general/simple_arm/sdf/simple_arm.sdf"
                   ),
                   "model_offset": [0., 3., 0.],
                   "floor_offset": [0, 0., -3],
                   "run_time": 10.,
                   "time_step": 0.001,
                   "planar": False,
                   "muscles": os.path.join(
                       get_farms_model_path(),
                       "general/simple_arm/config/muscles.yaml"
                   ),
                   "track": False,
                   "slow_down": True,
                   "sleep_time": 0.001,
                   "base_link": "base",
                   "pose": os.path.join(
                       get_farms_model_path(),
                       "general/simple_arm/config/pose.yaml"
                   ),
                   }
    container = Container(max_iterations=int(
        sim_options['run_time']/sim_options['time_step']))
    animal = SimpleArmSimulation(container, sim_options)
    animal.run()
    container.dump(overwrite=True)

    #: Plot
    forces = pd.read_hdf('./Results/muscles/forces.h5')
    lengths = pd.read_hdf('./Results/muscles/parameters.h5')
    plt.figure()
    plt.plot(forces)
    plt.legend(forces.keys())
    plt.figure()
    plt.plot(lengths)
    plt.legend(lengths.keys())
    plt.show()


if __name__ == '__main__':
    main()
