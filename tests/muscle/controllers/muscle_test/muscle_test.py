from farms_casadi_dae.casadi_dae_generator import CasadiDaeGenerator
from farms_muscle import MuscleSystem
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
muscles.muscles['flexor'].initialize_muscle_length()

#: Get motor
actuator = robot.getMotor('muscle')
actuator_pos = robot.getPositionSensor('muscle_pos')
actuator_pos.enable(int(robot.getBasicTimeStep()))
pos = actuator_pos.getValue
delta_pos = {'flexor': pos()}
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    delta_pos['flexor'] = -1*pos()
    res = muscles.step(delta_pos)
    force = float(muscles.muscles['flexor'].tendon_force)
    print(force, res['xf'].full()[:, 0], )
    actuator.setForce(force)
# Enter here exit cleanup code.
