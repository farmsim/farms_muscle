""" Debug simple arm model in pybullet. """

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
        self.point_1 = np.asarray([0.0, -0.3, 0.0])
        self.point_2 = np.asarray([0.0, 0.0, 0.3])
        self.point_3 = np.asarray([0.0, 0.3, 0.0])
        self.point_4 = np.asarray([0.0, 0.0, -0.3])
        self.line_1 = p.addUserDebugLine(
            lineFromXYZ=self.point_1,
            lineToXYZ=self.point_2,
            lineColorRGB=[1, 0, 0],
            lineWidth=4,
            lifeTime=0)
        self.line_2 = p.addUserDebugLine(
            lineFromXYZ=self.point_3,
            lineToXYZ=self.point_4,
            lineColorRGB=[0, 0, 1],
            lineWidth=4,
            lifeTime=0)

        self.force_1 = p.addUserDebugLine(
            lineFromXYZ=self.point_1,
            lineToXYZ=self.point_2,
            lineColorRGB=[0, 1, 0],
            lineWidth=4,
            lifeTime=0)
        self.force_2 = p.addUserDebugLine(
            lineFromXYZ=self.point_3,
            lineToXYZ=self.point_4,
            lineColorRGB=[0, 1, 0],
            lineWidth=4,
            lifeTime=0)

        #: Joint damping
        p.changeDynamics(
            self.animal, self.link_id['arm'], jointDamping=10
        )

    def controller_to_actuator(self):
        """ Implementation of abstractmethod. """

        #: Draw
        world_point_1 = np.asarray(
            BulletInterface.compute_world_space_point_in_base(
                1, self.point_1
            ))
        world_point_2 = np.asarray(
            BulletInterface.compute_world_space_point_in_link(
                1, 0, self.point_2
            ))
        world_point_3 = np.asarray(
            BulletInterface.compute_world_space_point_in_base(
                1, self.point_3
            ))
        world_point_4 = np.asarray(
            BulletInterface.compute_world_space_point_in_link(
                1, 0, self.point_4
            ))

        f_mag = 5
        f_vec_1 = (world_point_1 - world_point_2) / \
            np.linalg.norm((world_point_1 - world_point_2))*f_mag*0.0
        p.applyExternalForce(
            self.animal,
            self.link_id['arm'],
            forceObj=f_vec_1,
            posObj=world_point_2,
            flags=p.WORLD_FRAME
        )
        f_vec_2_unit = (world_point_3 - world_point_4) / \
            np.linalg.norm((world_point_3 - world_point_4))
        f_vec_2 = f_vec_2_unit*f_mag * \
            np.linalg.norm((world_point_3 - world_point_4))
        p.applyExternalForce(
            self.animal,
            self.link_id['arm'],
            forceObj=f_vec_2,
            posObj=world_point_4,
            flags=p.WORLD_FRAME
        )

        #: Update debug lines
        p.addUserDebugLine(
            lineFromXYZ=world_point_1,
            lineToXYZ=world_point_2,
            lineColorRGB=[1, 0, 0],
            lineWidth=4,
            lifeTime=0,
            replaceItemUniqueId=self.line_1
        )
        p.addUserDebugLine(
            lineFromXYZ=world_point_3,
            lineToXYZ=world_point_4,
            lineColorRGB=[0, 0, 1],
            lineWidth=4,
            lifeTime=0,
            replaceItemUniqueId=self.line_2
        )

        p.addUserDebugLine(
            lineFromXYZ=world_point_2,
            lineToXYZ=f_vec_1+world_point_2,
            lineWidth=4,
            lineColorRGB=[0, 1, 0],
            lifeTime=0,
            replaceItemUniqueId=self.force_1
        )
        p.addUserDebugLine(
            lineFromXYZ=world_point_4,
            lineToXYZ=f_vec_2+world_point_4,
            lineWidth=4,
            lineColorRGB=[0, 1, 0],
            lifeTime=0,
            replaceItemUniqueId=self.force_2
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
                   "model_offset": [0., 0., 0.],
                   "gravity": [0, 0, 0],
                   "floor_offset": [0, 0., -3],
                   "run_time": 25.,
                   "time_step": 0.001,
                   "planar": False,
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


if __name__ == '__main__':
    main()
