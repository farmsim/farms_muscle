from farms_casadi_dae.casadi_dae_generator import CasadiDaeGenerator
from farms_muscle.muscle_system import MuscleSystem
from controller import Robot
"""muscle_test controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, LED, DistanceSensor
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  led = robot.getLED('ledname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

dae = CasadiDaeGenerator()
muscles = MuscleSystem('conf/test_config.yml', dae)
muscles.dae.print_dae()
muscles.setup_integrator()
muscles['flexor'].initialize_muscle_length()

#: Get motor
actuator = robot.getMotor('muscle')
actuator_pos = robot.getPositionSensor('muscle_pos')
actuator_pos.enable(int(robot.getBasicTimeStep()))
pos = actuator_pos.getValue
delta_pos = {'flexor': pos()}

# Main loop:
# - perform simulation steps until Webots is stopping the controller
muscle_inputs = muscles.dae.u
while robot.step(timestep) != -1:
    muscle_inputs.set_val('l_delta_1', -1.*pos())
    muscle_inputs.set_val('stim_1', 0.95)
    res = muscles.step()
    force = float(muscles['flexor'].tendon_force)
    # print(force, res['xf'].full()[:, 0], )
    # print(muscles['flexor'].tendon_length
    #       + muscles['flexor'].fiber_length)
    # print(muscles['flexor'].tendon_length)
    # print(muscles['flexor'].fiber_length)

    actuator.setForce(force)
# Enter here exit cleanup code.
