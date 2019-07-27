"""
The MIT License

Copyright(c) 2017 Stanford NMBL Group(http: // nmbl.stanford.edu/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
import numpy as np
import os
import opensim
import random

# OpenSim interface
# The amin purpose of this class is to provide wrap all
# the necessery elements of OpenSim in one place
# The actual RL environment then only needs to:
# - open a model
# - actuate
# - integrate
# - read the high level description of the state
# The objective, stop condition, and other gym-related
# methods are enclosed in the OsimEnv class


class OsimModel(object):
    """ Interface class to opensim api for integration. """
    # Initialize simulation
    stepsize = 0.001

    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0

    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize, integrator_accuracy=5e-5):
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)
        self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        if self.verbose:
            self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.noutput = self.muscleSet.getSize()

        self.model.addController(self.brain)
        self.model.initSystem()

    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i, self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i, self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i, self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i, self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i, self.markerSet.get(i).getName())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError(
                "NaN passed in the activation vector. Values in [0,1] interval are required.")

        # TODO: Check if actions within [0,1]
        self.last_action = action

        brain = opensim.PrescribedController.safeDownCast(
            self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(float(action[j]))

    """
    Directly modifies activations in the current state.
    """

    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError(
                "NaN passed in the activation vector. Values in [0,1] interval are required.")
        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])
        self.reset_manager()

    """
    Get activations in the given state.
    """

    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        # Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(
                self.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(
                self.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(
                self.state) for i in range(joint.numCoordinates())]

        # Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[
                i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(
                self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(
                self.state).get(1).get(i) for i in range(3)]

            res["body_pos_rot"][name] = [body.getTransformInGround(
                self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(
                self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(
                self.state).get(0).get(i) for i in range(3)]

        # Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        # Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(
                self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(
                self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(
                self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(
                self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html

        # Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[
                i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[
                i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[
                i] for i in range(3)]

        # Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[
            i] for i in range(2)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[
            i] for i in range(2)]
        res["misc"]["mass_center_acc"] = [
            self.model.calcMassCenterAcceleration(self.state)[i] for i in range(2)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(
                self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.model.initializeState()
        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

    def get_state(self):
        return opensim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        try:
            self.state = self.manager.integrate(self.stepsize * self.istep)
        except Exception as e:
            print (e)
