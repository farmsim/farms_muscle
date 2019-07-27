#!/usr/bin/env python
""" Generate the opensim-model of Drosophilla. """

import opensim as osim
import numpy as np
import farms_pylog as pylog

class OSIMModel(object):
    """Class to create the drosophila model.
    """

    def __init__(self, model=None, name='model', n_links=1, length=0.5):
        """ Parameters
            ----------
            model: <osim.Model>
                Opensim model instance
            name: <str>
                Name of the dorosphila model
        """
        super(OSIMModel, self).__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = osim.Model()
        self.model.setName(name)

        #: Set Gravity
        self.model.set_gravity = osim.Vec3(0, -9.81, 0.0)

        self.color = osim.Vec3(0,0,1)

        #: Methods
        self.generate_segments(n_links, length)

    def generate_segments(self, n_links, length, mass=2):
        """Generate n-link chain.
     
         Parameters
         ----------
         n_links : <int>    
         Number of links in the chain
        """

        #: Generate base
        self.add_segment('link_0', length)
        #: Base-link
        #: pylint: disable=no-member
        base_link = osim.WeldJoint(
            'base_link', self.model.getGround(),
            osim.Vec3(0, 2*length + n_links, 0),
            osim.Vec3(0),
            self.model.getBodySet().get('link_0'),
            osim.Vec3(0), osim.Vec3(0, 0, 0))

        self.model.addJoint(base_link)

        #: Generate n-links
        for n in range(n_links):
            #: Create link
            self.add_segment('link_' + str(n+1), length, mass)
            #: Attach link
            _link = osim.PinJoint(
                'link_'+str(n)+'_joint',
                self.model.getBodySet().get('link_'+ str(n)),
                osim.Vec3(0, -length*0.5, 0),
                osim.Vec3(0, 0, 0),
                self.model.getBodySet().get('link_'+str(n+1)),
                osim.Vec3(0, length*0.5, 0),
                osim.Vec3(0))
            self.model.addJoint(_link)

        #: Add muscles
        self.add_muscle()

    def add_segment(self, seg_name, length, mass=2):
        """
        Create link segment        
        Parameters
        ----------        
        """
        radius = 0.05*length
        
        inertia = osim.Vec3() 
        inertia[0] = (1/12.)*mass*(3*radius**2 + length**2)
        inertia[1] = (1/2.)*mass*radius**2
        inertia[2] = (1/12.)*mass*(3*radius**2 + length**2)
        pylog.debug('Inertia : {}'.format(inertia))
        #: Create segment
        segment = osim.Body(seg_name,
                            mass,
                            osim.Vec3(0),
                            osim.Inertia(inertia))
        shape = osim.Cylinder(radius, length*0.5)
        appearance = osim.Appearance()
        appearance.set_opacity = 0.8
        appearance.set_color(self.color)
        self.color[1] = not(self.color[1])
        self.color[2] = not(self.color[2])
        shape.set_Appearance(appearance)
        segment.attachGeometry(shape)
        self.model.addBody(segment)

    def add_muscle(self, muscle_name='muscle', links=[0,1]):
        """Add muscle and wrapping objects"""

        m1 = osim.Millard2012EquilibriumMuscle()
        m1.setName(muscle_name + '_1')
        # Muscle attachment
        m1.addNewPathPoint(
            'origin',
            self.model.getBodySet().get('link_'+str(links[0])),
            osim.Vec3(0.025, -0.15,0))
        m1.addNewPathPoint(
            'mid',
            self.model.getBodySet().get('link_'+str(links[0])),
            osim.Vec3(0.025, -0.25, 0))
        m1.addNewPathPoint(
            'insertion',
            self.model.getBodySet().get('link_'+str(links[1])),
            osim.Vec3(
                0.025, 0.1, 0.0))
        # Muscle parameters
        m1.setOptimalFiberLength(0.11)
        m1.setTendonSlackLength(0.13)
        m1.setMaxIsometricForce(1500)
        self.model.addForce(m1)

        m2 = osim.Millard2012EquilibriumMuscle()
        m2.setName(muscle_name + '_2')
        # Muscle attachment
        m2.addNewPathPoint(
            'origin',
            self.model.getBodySet().get('link_'+str(links[0])),
            osim.Vec3(-0.025, -0.15,0))
        m2.addNewPathPoint(
            'mid',
            self.model.getBodySet().get('link_'+str(links[0])),
            osim.Vec3(-0.025, -0.25, 0))
        m2.addNewPathPoint(
            'insertion',
            self.model.getBodySet().get('link_'+str(links[1])),
            osim.Vec3(
                -0.025, 0.1, 0.0))
        # Muscle parameters
        m2.setOptimalFiberLength(0.11)
        m2.setTendonSlackLength(0.13)
        m2.setMaxIsometricForce(1500)
        self.model.addForce(m2)


def main():
    """ Main function. """

    #: osim model
    model=osim.Model()
    gravity=osim.Vec3(0, -9.81, 0)
    model.set_gravity(gravity)
    OSIMModel(model, 'pendulum', n_links=1)
    # fly.add_muscle()
    #: Visualization
    #: pylint: disable=no-member
    # model.setUseVisualizer(True)
    # state = model.initSystem()
    # model.equilibrateMuscles(state)
    # manager = osim.Manager(model)
    # model.getMuscles().get(
    #     'fore_right_protraction').setActivation(state, 1)
    # model.getMuscles().get(
    #     'fore_right_retraction').setActivation(state, 0.05)
    # state.setTime(0)
    # manager.initialize(state)
    # state = manager.integrate(5)

    #: Dump osim file
    model.finalizeConnections()
    file_name='pendulum.osim'
    model.printToXML(file_name)


if __name__ == '__main__':
    main()
