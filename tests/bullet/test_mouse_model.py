""" Test mouse model in pybullet. """

from farms_models.utils import (
    get_farms_model_path, get_sdf_path, get_model_path
)
from bullet_simulation import BulletSimulation
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
import matplotlib.pyplot as plt
pylog.set_level('error')


class MouseSimulation(BulletSimulation):
    """Mouse Simulation Class
    """

    def __init__(self, container, sim_options):
        super(MouseSimulation, self).__init__(
            container, SimulationUnitScaling(), **sim_options
        )
        self.connection_mode = p.getConnectionInfo(
            self.physics_id
        )['connectionMethod']
        if self.MUSCLES and (self.connection_mode == 1):
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
        if self.MUSCLES and (self.connection_mode == 1):
            for muscle in self.muscles.muscles.keys():
                self.muscle_params[muscle].value = p.readUserDebugParameter(
                    self.muscle_excitation[muscle]
                )
        # p.setJointMotorControl2(
        #     self.animal,
        #     self.joint_id['LHip'],
        #     p.POSITION_CONTROL,
        #     targetPosition=np.sin(2*np.pi*0.5*self.TIME)*0.8
        # )
        # p.setJointMotorControl2(
        #     self.animal,
        #     self.joint_id['RHip'],
        #     p.POSITION_CONTROL,
        #     targetPosition=np.sin(2*np.pi*0.5*self.TIME)*0.8
        # )
        # p.setJointMotorControl2(
        #     self.animal,
        #     self.joint_id['LKnee'],
        #     p.POSITION_CONTROL,
        #     targetPosition=0.0
        # )
        # p.setJointMotorControl2(
        #     self.animal,
        #     self.joint_id['RKnee'],
        #     p.POSITION_CONTROL,
        #     targetPosition=0.0
        # )

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
                   "model": get_sdf_path("mouse", "3"),
                   "model_offset": [0., 0., 0.1],
                   "run_time": 45.,
                   "planar": False,
                   "muscles": os.path.join(
                       get_model_path("mouse", "3"),
                       "config/hind_locomotion_muscles_scaled.yaml"
                   ),
                   "track": True,
                   "camera_yaw": 270,
                   "slow_down": False,
                   "sleep_time": 0.001,
                   "base_link": "Pelvis"
                   }
    container = Container(
        max_iterations=int(sim_options['run_time']/0.001
        )
    )
    animal = MouseSimulation(container, sim_options)
    animal.run()
    container.dump(overwrite=True)

if __name__ == '__main__':
    main()
