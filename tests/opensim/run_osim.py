#!/usr/bin/env python

""" Main file for drosophila simulation in opensim. """
import numpy as np
import farms_pylog as pylog
from osim_rl import OsimModel

def main():
    """ Main file. """
    # osim model
    model = OsimModel('./pendulum.osim',
                    visualize=True)
    # Initialize
    model.reset()
    model.reset_manager()

    # List model components
    model.list_elements()

    # Integrate
    for j in range(0, 5000):
        model.integrate()
        # model.actuate([np.abs(np.sin(2*np.pi*j/1000*0.25)),
        #                np.abs(np.sin(2*np.pi*j*0.001*0.25 + np.pi))])
        model.actuate([0.05,0.5])
        res = model.compute_state_desc()
        # biolog.info('Time {} \t Act : {} \t TC2 : {}' .format(
        #     j*0.001,
        #     abs(np.sin(2*np.pi*j*0.001*0.1)),
        #     res['forces']['fore_right_retraction']))
    print(res['joint_pos']['link_0_joint'])

if __name__ == '__main__':
    main()
