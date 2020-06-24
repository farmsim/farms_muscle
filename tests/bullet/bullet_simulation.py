""" Class to run animal model. """
import abc
from tqdm import tqdm
import pybullet as p
import pybullet_data
import numpy as np
import farms_pylog as pylog
import os
import time
import matplotlib.pyplot as plt
from farms_container import Container
import yaml
try:
    from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
except ImportError:
    pylog.warning("farms-muscle not installed!")
try:
    from farms_network.neural_system import NeuralSystem
except ImportError:
    pylog.warning("farms-network not installed!")
pylog.set_level("debug")


class BulletSimulation(metaclass=abc.ABCMeta):
    """Methods to run bullet simulation.
    """

    def __init__(self, container, units, **kwargs):
        super(BulletSimulation, self).__init__()
        self.units = units
        self.container = container
        #: Simulation options
        self.GUI = p.DIRECT if kwargs["headless"] else p.GUI
        self.GRAVITY = np.array(
            kwargs.get("gravity", [0, 0, -9.81])
        )*self.units.gravity
        self.TIME_STEP = kwargs.get("time_step", 0.001)
        self.REAL_TIME = kwargs.get("real_time", 0)
        self.RUN_TIME = kwargs.get("run_time", 10)
        self.SOLVER_ITERATIONS = kwargs.get("solver_iterations", 50)
        self.MODEL = kwargs.get("model", None)
        self.MODEL_OFFSET = kwargs.get("model_offset", None)
        self.floor_offset = kwargs.get("floor_offset", [0.0, 0.0, 0.0])
        self.GROUND_CONTACTS = kwargs.get("ground_contacts", ())
        self.BASE_LINK = kwargs.get("base_link", None)
        self.CONTROLLER = kwargs.get("controller", None)
        self.POSE_FILE = kwargs.get("pose", None)
        self.MUSCLE_CONFIG_FILE = kwargs.get("muscles", None)
        self.camera_distance = kwargs.get('camera_distance', 0.1)
        self.camera_yaw = kwargs.get('camera_yaw', 90)
        self.camera_pitch = kwargs.get('camera_pitch', -10)
        self.track_animal = kwargs.get("track", True)
        self.slow_down = kwargs.get("slow_down", False)
        self.sleep_time = kwargs.get("sleep_time", 0.001)
        self.VIS_OPTIONS_BACKGROUND_COLOR_RED = kwargs.get(
            'background_color_red', 0.5)
        self.VIS_OPTIONS_BACKGROUND_COLOR_GREEN = kwargs.get(
            'background_color_GREEN', 0.5)
        self.VIS_OPTIONS_BACKGROUND_COLOR_BLUE = kwargs.get(
            'background_color_BLUE', 0.5)
        self.RECORD_MOVIE = kwargs.get('record', False)
        self.MOVIE_NAME = kwargs.get('moviename', 'default_movie.mp4')

        #: Init
        self.TIME = 0.0
        self.physics_id = -1
        self.plane = None
        self.animal = None
        self.control = None
        self.num_joints = 0
        self.joint_id = {}
        self.joint_type = {}
        self.link_id = {}
        self.ground_sensors = {}
        #: ADD PHYSICS SIMULATION namespace to container
        self.sim_data = self.container.add_namespace('physics')
        #: ADD Tables to physics container
        self.sim_data.add_table('base_position')
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocities')
        self.sim_data.add_table('joint_torques')
        self.sim_data.add_table('ground_contacts')

        self.ZEROS_3x1 = np.zeros((3,))

        #: Muscles
        if self.MUSCLE_CONFIG_FILE:
            self.MUSCLES = True
        else:
            self.MUSCLES = False

        #: Setup
        self.setup_simulation()

        #: Enable rendering
        self.rendering(1)

        # Initialize pose
        if self.POSE_FILE:
            self.initialize_position(self.POSE_FILE)

        #: Initialize simulation
        self.initialize_simulation()

        #: Camera
        if self.GUI == p.GUI:
            base = np.array(self.base_position)
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                self.camera_yaw,
                self.camera_pitch,
                base)

    def __del__(self):
        pylog.info("Disconnecting pybullet")
        try:
            p.disconnect()
        except p.error:
            pylog.warning("Not connected to any server to disconnect")

    def rendering(self, render=1):
        """Enable/disable rendering"""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, render)

    def setup_simulation(self):
        """ Setup the simulation. """
        ########## PYBULLET SETUP ##########
        if self.RECORD_MOVIE and self.GUI == p.GUI:
            self.physics_id = p.connect(
                self.GUI,
                options='--background_color_red={} --background_color_green={} --background_color_blue={} --mp4={}'.format(
                    self.VIS_OPTIONS_BACKGROUND_COLOR_RED,
                    self.VIS_OPTIONS_BACKGROUND_COLOR_GREEN,
                    self.VIS_OPTIONS_BACKGROUND_COLOR_RED,
                    self.MOVIE_NAME))
        elif self.GUI == p.GUI:
            if not p.getConnectionInfo(0)['isConnected']:
                self.physics_id = p.connect(
                    self.GUI,
                    options='--background_color_red={} --background_color_green={} --background_color_blue={}'.format(
                        self.VIS_OPTIONS_BACKGROUND_COLOR_RED,
                        self.VIS_OPTIONS_BACKGROUND_COLOR_GREEN,
                        self.VIS_OPTIONS_BACKGROUND_COLOR_RED))
            else:
                self.physics_id = 0
        else:
            self.physics_id = p.connect(self.GUI)
        # p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #: everything should fall down
        p.setGravity(self.GRAVITY[0], self.GRAVITY[1], self.GRAVITY[2])
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.TIME_STEP,
            numSolverIterations=100,
            enableFileCaching=0,
            numSubSteps=1,
            solverResidualThreshold=1e-10,
            erp=1e-1,
            contactERP=0.0,
            frictionERP=0.0,
        )
        #: Turn off rendering while loading the models
        self.rendering(0)

        ########## ADD FLOOR ##########
        self.plane = p.loadURDF(
            "plane.urdf", self.floor_offset,
            globalScaling=self.units.meters
        )

        ########## ADD ANIMAL #########
        if ".sdf" in self.MODEL:
            self.animal = p.loadSDF(self.MODEL)[0]
        elif ".urdf" in self.MODEL:
            self.animal = p.loadURDF(self.MODEL)
        if self.MODEL_OFFSET:
            p.resetBasePositionAndOrientation(
                self.animal, self.MODEL_OFFSET,
                p.getQuaternionFromEuler([0., 0., 0.]))
        self.num_joints = p.getNumJoints(self.animal)

        #: Generate joint_name to id dict
        #: FUCK : Need to clean this section
        self.link_id[p.getBodyInfo(self.animal)[0].decode('UTF-8')] = -1
        for n in range(self.num_joints):
            info = p.getJointInfo(self.animal, n)
            _id = info[0]
            joint_name = info[1].decode('UTF-8')
            link_name = info[12].decode('UTF-8')
            _type = info[2]
            self.joint_id[joint_name] = _id
            self.joint_type[joint_name] = _type
            self.link_id[link_name] = _id
            pylog.debug("Link name {} id {}".format(link_name, _id))

        ########## ADD GROUND_CONTACTS ##########
        for contact in self.GROUND_CONTACTS:
            self.add_ground_contact_sensor(contact)
            self.sim_data.ground_contacts.add_parameter(contact)

        ########## ADD MUSCLES ##########
        if self.MUSCLES:
            self.initialize_muscles()

        ########## ADD CONTROLLER ##########
        if self.CONTROLLER:
            self.controller = NeuralSystem(
                self.CONTROLLER, self.container)

        #: ADD base position parameters
        self.sim_data.base_position.add_parameter('x')
        self.sim_data.base_position.add_parameter('y')
        self.sim_data.base_position.add_parameter('z')

        #: ADD joint paramters
        for name, _ in self.joint_id.items():
            self.sim_data.joint_positions.add_parameter(name)
            self.sim_data.joint_velocities.add_parameter(name)
            self.sim_data.joint_torques.add_parameter(name)

        #: ADD applied torques parameter
        self.applied_torques = {
            key: 0.0 for key in self.joint_id.keys()
        }

        ########## DISABLE DEFAULT BULLET CONTROLLERS  ##########
        p.setJointMotorControlArray(
            self.animal,
            np.arange(self.num_joints),
            p.VELOCITY_CONTROL,
            targetVelocities=np.zeros((self.num_joints,)),
            forces=np.zeros((self.num_joints,))
        )
        p.setJointMotorControlArray(
            self.animal,
            np.arange(self.num_joints),
            p.POSITION_CONTROL,
            forces=np.zeros((self.num_joints,))
        )
        p.setJointMotorControlArray(
            self.animal,
            np.arange(self.num_joints),
            p.TORQUE_CONTROL,
            forces=np.zeros((self.num_joints,))
        )

        self.total_mass = 0.0

        for j in np.arange(-1, p.getNumJoints(self.animal)):
            self.total_mass += p.getDynamicsInfo(self.animal, j)[0]

        self.bodyweight = -1 * self.total_mass * self.GRAVITY[2]
        pylog.info("Total mass = {}".format(self.total_mass))
        if self.GUI == p.GUI:
            self.rendering(1)

    def initialize_simulation(self):
        """ Initialize simulation. """
        ########## INITIALIZE THE CONTAINER ##########
        self.container.initialize()

        ########## SETUP THE INTEGRATOR ##########
        if self.CONTROLLER:
            self.controller.setup_integrator()
        if self.MUSCLES:
            self.muscles.setup_integrator()

    def initialize_muscles(self):
        self.muscles = MusculoSkeletalSystem(
            self.container, self.MUSCLE_CONFIG_FILE
        )

    def initialize_position(self, pose_file=None):
        """Initialize the pose of the animal.
        Parameters:
        pose_file : <selftr>
             File path to the pose data
        """
        pylog.debug("Setting pose....")
        if pose_file:
            pylog.debug('Reading {}'.format(pose_file))
            try:
                with open(pose_file) as stream:
                    data = yaml.load(stream, Loader=yaml.SafeLoader)
                    data = {k.lower(): v for k, v in data.items()}
            except FileNotFoundError:
                pylog.error("Pose file {} not found".format(pose_file))
                return
            for joint, _id in self.joint_id.items():
                _pose = np.deg2rad(data['joints'].get(joint, 0))
                p.resetJointState(
                    self.animal, _id,
                    targetValue=_pose
                )
        else:
            return None

    def add_ground_contact_sensor(self, link):
        """Add new ground contact sensor

        Parameters
        ----------

        Returns
        -------
        out :

        """
        self.ground_sensors[link] = self.link_id[link]

    def _get_contact_force(self, link_id):
        c = p.getContactPoints(
            self.animal, self.plane,
            link_id, -1)
        self.contact_pos = np.sum(
            [pt[5] for pt in c], axis=0) / len(c) if c else self.ZEROS_3x1
        self.normal_dir = -1 * np.sum(
            [pt[7]for pt in c], axis=0) / len(c) if c else self.ZEROS_3x1
        self.normal = np.sum(
            [pt[9]for pt in c], axis=0) / self.bodyweight if c else self.ZEROS_3x1
        force = self.normal * self.normal_dir

        return force[2]

    def get_contact_friction(self, link_id):
        c = p.getContactPoints(
            self.animal, self.plane,
            link_id, -1)

        force1 = np.sum(
            [pt[10]*np.asarray(pt[11]) for pt in c], axis=0) if c else self.ZEROS_3x1
        force2 = np.sum(
            [pt[12]*np.asarray(pt[13]) for pt in c], axis=0) if c else self.ZEROS_3x1
        return force1, force2

    def is_contact(self, link_name):
        """ Check if link is in contact with floor. """
        return True if p.getContactPoints(
            self.animal, self.plane,
            self.link_id[link_name],
            -1
        ) else False

    @property
    def joint_states(self):
        """ Get all joint states  """
        return p.getJointStates(
            self.animal,
            np.arange(0, p.getNumJoints(self.animal))
        )

    @property
    def ground_reaction_forces(self):
        """Get the ground reaction forces.  """
        return list(
            map(self._get_contact_force, self.ground_sensors.values())
        )

    @property
    def base_position(self):
        """ Get the position of the animal  """
        if self.BASE_LINK and self.link_id[self.BASE_LINK] != -1:
            link_id = self.link_id[self.BASE_LINK]
            return (p.getLinkState(self.animal, link_id))[0]
        else:
            return (p.getBasePositionAndOrientation(self.animal))[0]

    @property
    def joint_positions(self):
        """ Get the joint positions in the animal  """
        return tuple(
            state[0] for state in p.getJointStates(
                self.animal,
                np.arange(0, p.getNumJoints(self.animal))
            )
        )

    @property
    def joint_torques(self):
        """ Get the joint torques in the animal  """
        return tuple(
            state[-1] for state in p.getJointStates(
                self.animal,
                np.arange(0, p.getNumJoints(self.animal))
            )
        )

    @property
    def joint_velocities(self):
        """ Get the joint velocities in the animal  """
        return tuple(
            state[1] for state in p.getJointStates(
                self.animal,
                np.arange(0, p.getNumJoints(self.animal))
            )
        )

    @property
    def distance_x(self):
        """ Distance the animal has travelled in x-direction. """
        return self.base_position[0]

    @property
    def distance_y(self):
        """ Distance the animal has travelled in y-direction. """
        return self.base_position[1]

    @property
    def distance_z(self):
        """ Distance the animal has travelled in z-direction. """
        return self.base_position[2]

    @property
    def mechanical_work(self):
        """ Mechanical work done by the animal. """
        return np.sum(np.sum(
            np.abs(np.asarray(self.sim_data.joint_torques.log)
                   * np.asarray(self.sim_data.joint_velocities.log))
        ))*self.TIME_STEP/self.RUN_TIME

    @property
    def thermal_loss(self):
        """ Thermal loss for the animal. """
        return np.sum(np.sum(
            np.asarray(self.sim_data.joint_torques.log)**2
        ))*self.TIME_STEP/self.RUN_TIME

    def update_logs(self):
        """ Update all the physics logs. """
        #: Update log
        self.sim_data.base_position.values = np.asarray(
            self.base_position)
        self.sim_data.joint_positions.values = np.asarray(
            self.joint_positions)
        self.sim_data.joint_velocities.values = np.asarray(
            self.joint_velocities)
        self.sim_data.joint_torques.values = np.asarray(
            self.joint_torques)
        self.sim_data.ground_contacts.values = np.asarray(
            self.ground_reaction_forces)

    @abc.abstractmethod
    def controller_to_actuator(self):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints

        Parameters
        ----------
        None

        Returns
        -------
        out :

        """
        pass

    @abc.abstractmethod
    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.

        Parameters
        ----------
        None

        Returns
        -------
        out:
        """
        pass

    @abc.abstractmethod
    def update_parameters(self, params):
        """ Update parameters. """
        pass

    @abc.abstractmethod
    def optimization_check(self):
        """ Optimization check. """
        pass

    def step(self, optimization=False):
        """ Step the simulation.

        Returns
        -------
        out :
        """
        #: Camera
        if self.GUI == p.GUI and self.track_animal:
            base = np.array(self.base_position)
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                self.camera_yaw,
                self.camera_pitch,
                base)
        #: update the feedback to controller
        self.feedback_to_controller()
        #: Step controller
        if self.CONTROLLER:
            self.controller.step(self.TIME_STEP)
        #: update the controller_to_actuator
        self.controller_to_actuator()
        #: Step muscles
        if self.MUSCLES:
            self.muscles.step()
        #: Step TIME
        self.TIME += self.TIME_STEP
        #: Update logs
        self.update_logs()
        #: Update container log
        self.container.update_log()
        #: Step physics
        p.stepSimulation()
        if self.slow_down:
            time.sleep(self.sleep_time)
        #: Check if optimization is to be killed
        if optimization:
            optimization_status = self.optimization_check()
            return optimization_status

        return True

    def run(self, optimization=False):
        """ Run the full simulation. """
        for t in tqdm(range(0, int(self.RUN_TIME / self.TIME_STEP))):
            status = self.step(optimization=optimization)
            if not status:
                return False


def main():
    """ Main """

    sim_options = {"headless": False,
                   "model": "../mouse/models/mouse_bullet/sdf/mouse_bullet.sdf",
                   "model_offset": [0., 0., 0.0],
                   "pose": "../mouse/config/mouse_rig_simple_default_pose.yml",
                   "run_time": 10.}

    animal = BulletSimulation(**sim_options)
    animal.run(optimization=True)
    # container = Container.get_instance()
    # animal.control.visualize_network(edge_labels=False)
    # plt.figure()
    # plt.plot(np.sin(container.neural.outputs.log))
    # plt.show()


if __name__ == '__main__':
    main()
