""" Test mouse model in pybullet. """

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
pylog.set_level('error')


class MouseSimulation(BulletSimulation):
    """Mouse Simulation Class
    """

    def __init__(self, container, sim_options):
        super(MouseSimulation, self).__init__(
            container, SimulationUnitScaling(), **sim_options
        )
        self.test_muscles = ('LEFT_RF', 'LEFT_SM', 'LEFT_BFP_caudal')
        u = container.muscles.activations
        self.muscle_params = {}
        self.muscle_excitation = {}
        for muscle in self.test_muscles:
            self.muscle_params[muscle] = u.get_parameter(
                'stim_{}'.format(muscle)
            )
            self.muscle_excitation[muscle] = p.addUserDebugParameter(
                "flexor {}".format(muscle), 0, 1, 0.05
            )

    def controller_to_actuator(self):
        """ Implementation of abstractmethod. """
        for muscle in self.test_muscles:
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
                       "mouse_v1/design/sdf/mouse_locomotion.sdf"
                   ),
                   "model_offset": [0., 0., 0.1],
                   "run_time": 10.,
                   "planar": False,
                   "muscles": os.path.join(
                       get_farms_model_path(),
                       "mouse_v1/config/hind_muscles.yaml"
                   ),
                   "track": False,
                   "slow_down": True,
                   "sleep_time": 0.001,
                   "base_link": "Pelvis"
                   }
    container = Container(max_iterations=int(10/0.001))
    animal = MouseSimulation(container, sim_options)
    animal.run()
    continer.dump(overwrite=True)


if __name__ == '__main__':
    main()
