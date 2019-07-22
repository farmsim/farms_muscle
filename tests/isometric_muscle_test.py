#!/usr/bin/python
""" Generate Isomertic Muscle Data."""
from farms_dae_generator.dae_generator import DaeGenerator
from farms_muscle.musculo_skeletal_parameters import MuscleParameters
from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
import farms_pylog as pylog
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def isometric_contraction():
    #: DAE
    muscles = MusculoSkeletalSystem('../farms_muscle/conf/muscles.yaml')

    #: Initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    _time = np.arange(0, 100, dt)  #: Time

    #: Integrate the network
    muscle_act = {'m1': 0.75}

    #: Initialize DAE
    muscles.dae.initialize_dae()

    x0 = np.array([0.11, 0.0])

    #: integrator
    muscles.setup_integrator(x0)

    start = time.time()
    N = 10000
    dl = np.linspace(-0.1, 0.1, N/100.0)
    count = -1
    time_vec = np.linspace(0., N/1000., N)
    for j in range(0, N):
        if j % 100 == 0:
            count += 1
        muscles.dae.u.values = np.array([dl[count], 1.0])
        muscles.step()
        muscles.dae.update_log()
        muscles.muscle_sys.py_update_outputs()
    end = time.time()
    print('TIME {}'.format(end-start))

    #: PLOTTING
    data_y = muscles.dae.y.log
    print(muscles.dae.y.name_idx)
    print(data_y)
    plt.figure(1)
    plt.title('Muscle Tendon Length')
    plt.plot(time_vec[0:N:100], data_y[0:N:100, 0])
    plt.plot(time_vec[0:N:100], data_y[0:N:100, 0], '*')
    plt.grid(True)
    plt.show()

    # plt.figure(2)
    # plt.title('Muscle Length')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 1])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 1], '*')
    # plt.grid(True)

    # plt.figure(3)
    # plt.title('Belly Force')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 2])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 2], '*')
    # plt.grid(True)

    # plt.figure(4)
    # plt.title('Parallel Force')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 3])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 3], '*')
    # plt.grid(True)

    # plt.figure(5)
    # plt.title('Force-Length')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 4])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 4], '*')
    # plt.grid(True)

    # plt.figure(6)
    # plt.title('Force-Velocity')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 5])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 5], '*')
    # plt.grid(True)

    # plt.figure(7)
    # plt.title('Contractile Force')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 6])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 6], '*')
    # plt.grid(True)

    # plt.figure(8)
    # plt.title('Tendon Force')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 7])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 7], '*')
    # plt.grid(True)

    # plt.figure(9)
    # plt.title('Force-Profile')
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 6])
    # plt.plot(np.linspace(0, N*0.001, N/100.0), data_y[0:N:100, 6], '*')
    # passive_force = data_y[0:N:100, 2] + data_y[0:N:100, 3]
    # plt.plot(np.linspace(0, N*0.001, N/100.0), passive_force)
    # plt.plot(np.linspace(0, N*0.001, N/100.0), passive_force, '*')
    # total_force = data_y[0:N:100, 6] + passive_force
    # plt.plot(np.linspace(0, N*0.001, N/100.0), total_force)
    # plt.plot(np.linspace(0, N*0.001, N/100.0), total_force, '*')
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    isometric_contraction()
