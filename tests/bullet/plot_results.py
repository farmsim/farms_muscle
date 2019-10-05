""" Plot results from dae. """

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import pandas as pd

def main(FILE_PATH):
    #: Logging
    musculo_x = pd.read_hdf(os.path.join(FILE_PATH, 'musculo_x.h5'))
    musculo_xdot = pd.read_hdf(os.path.join(FILE_PATH, 'musculo_xdot.h5'))
    musculo_y = pd.read_hdf(os.path.join(FILE_PATH, 'musculo_y.h5'))
    musculo_p = pd.read_hdf(os.path.join(FILE_PATH, 'musculo_p.h5'))
    musculo_u = pd.read_hdf(os.path.join(FILE_PATH, 'musculo_u.h5'))


    plt.figure()
    plt.title('musculo_l_ce')
    _names = [key for key in musculo_x.keys() if 'l_ce' in key]
    plt.plot(musculo_x[_names])
    plt.legend(tuple(_names))
    plt.grid(True)
    
    plt.figure()
    plt.title('musculo_v_ce')
    _names = [key for key in musculo_xdot.keys() if 'v_ce' in key]
    plt.plot(musculo_xdot[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    plt.figure()
    plt.title('musculo_act')
    _names = [key for key in musculo_x.keys() if 'activation' in key]
    plt.plot(musculo_x[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    # plt.figure()
    # plt.title('musculo_xdot')
    # plt.plot(musculo_xdot)
    # plt.legend(tuple(musculo_xdot.keys()))
    # plt.grid(True)

    # plt.figure()
    # plt.title('Force-Profile')
    # lmtu = musculo_u['lmtu_m1']
    # active_force = musculo_y['active_force_m1']
    # plt.plot(lmtu, active_force)
    # passive_force = musculo_y['belly_force_m1'] + musculo_y['parallel_force_m1']
    # plt.plot(lmtu, passive_force)
    # total_force = active_force + passive_force
    # plt.plot(lmtu, total_force)
    # plt.grid(True)

    plt.figure()
    plt.subplot(411)
    plt.title('Tendon Force')
    _names = [key for key in musculo_y.keys() if 'tendon_force' in key]
    plt.plot(musculo_y[_names])
    plt.legend(tuple(_names))
    plt.grid(True)
    plt.subplot(412)
    plt.title('Active Force')
    _names = [key for key in musculo_y.keys() if 'active_force' in key]
    plt.plot(musculo_y[_names])
    plt.legend(tuple(_names))
    plt.grid(True)
    plt.subplot(413)
    plt.title('Parallel Force')
    _names = [key for key in musculo_y.keys() if 'parallel_force' in key]
    plt.plot(musculo_y[_names])
    plt.legend(tuple(_names))
    plt.grid(True)
    plt.subplot(414)
    plt.title('Belly Force')
    _names = [key for key in musculo_y.keys() if 'belly_force' in key]
    plt.plot(musculo_y[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    # plt.figure()
    # plt.title('Tendon Length')
    # _names = [key for key in musculo_y.keys() if 'tendon_length' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

    plt.figure()
    plt.title('Length')
    _names = [key for key in musculo_u.keys() if 'lmtu_' in key]
    plt.plot(musculo_u[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    plt.figure()
    plt.title('Ia Afferents')
    _names = [key for key in musculo_y.keys() if 'Ia' in key]
    plt.plot(musculo_y[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    plt.figure()
    plt.title('Ib Afferents')
    _names = [key for key in musculo_y.keys() if 'Ib' in key]
    plt.plot(musculo_y[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    plt.figure()
    plt.title('II Afferents')
    _names = [key for key in musculo_y.keys() if 'II' in key]
    plt.plot(musculo_y[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    # plt.figure()
    # plt.title('Parallel Force')
    # _names = [key for key in musculo_y.keys() if 'parallel_force' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

    plt.show()


if __name__ == '__main__':
    #: Argparser
    parser = ArgumentParser()

    parser.add_argument('--results_path', '-r',
                        dest='results_path', type=str,
                        help="Path containing results")

    args = parser.parse_args()

    FILE_PATH = args.results_path
    
    main(FILE_PATH)
