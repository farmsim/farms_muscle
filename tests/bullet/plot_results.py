""" Plot results from dae. """

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import pandas as pd

def main(FILE_PATH):
    #: Logging
    musculo_x = pd.read_hdf(os.path.join(FILE_PATH, 'states.h5'))
    musculo_xdot = pd.read_hdf(os.path.join(FILE_PATH, 'dstates.h5'))
    musculo_y = pd.read_hdf(os.path.join(FILE_PATH, 'outputs.h5'))
    musculo_p = pd.read_hdf(os.path.join(FILE_PATH, 'parameters.h5'))
    musculo_u = pd.read_hdf(os.path.join(FILE_PATH, 'activations.h5'))
    musculo_f = pd.read_hdf(os.path.join(FILE_PATH, 'forces.h5'))
    musculo_c = pd.read_hdf(os.path.join(FILE_PATH, 'constants.h5'))


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

    # plt.figure()
    # plt.title('musculo_act')
    # _names = [key for key in musculo_x.keys() if 'activation' in key]
    # plt.plot(musculo_x[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

    # plt.figure()
    # plt.title('musculo_xdot')
    # plt.plot(musculo_xdot)
    # plt.legend(tuple(musculo_xdot.keys()))
    # plt.grid(True)

    plt.figure()
    plt.title('Force-Length-Curve')
    lmtu = musculo_p['lmtu_m1']
    lce = musculo_x['l_ce_m1']
    l_opt = musculo_c['l_opt_m1']
    f_max = musculo_c['f_max_m1']
    force_length = musculo_y['force_length_m1']
    passive_force = (musculo_y['parallel_force_m1'])/f_max[0]
    total_force = force_length + passive_force
    plt.plot(lce/l_opt[0], force_length, 'b')
    plt.plot(lce/l_opt[0], passive_force, 'r')
    plt.plot(lce/l_opt[0], total_force, 'g')
    plt.legend(('force_length', 'passive', 'total'))
    plt.grid(True)

    plt.figure()
    plt.title('Force-Velocity-Curve')
    vce = musculo_xdot['v_ce_m1']
    v_max = musculo_c['v_max_m1']
    force_velocity = musculo_y['force_velocity_m1']
    plt.plot(vce, force_velocity, 'b')
    plt.legend(('force_velocity',))
    plt.grid(True)

    # plt.figure()
    # plt.subplot(411)
    # plt.title('Tendon Force')
    # _names = [key for key in musculo_f.keys() if 'tendon_force' in key]
    # plt.plot(musculo_f[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)
    # plt.subplot(412)
    # plt.title('Active Force')
    # _names = [key for key in musculo_y.keys() if 'force_length' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)
    # plt.subplot(413)
    # plt.title('Parallel Force')
    # _names = [key for key in musculo_y.keys() if 'parallel_force' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)
    # plt.subplot(414)
    # plt.title('Belly Force')
    # _names = [key for key in musculo_y.keys() if 'belly_force' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

    # plt.figure()
    # plt.title('Tendon Length')
    # _names = [key for key in musculo_y.keys() if 'tendon_length' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

    plt.figure()
    plt.title('Length')
    _names = [key for key in musculo_p.keys() if 'lmtu_' in key]
    plt.plot(musculo_p[_names])
    plt.legend(tuple(_names))
    plt.grid(True)

    # plt.figure()
    # plt.title('Velocity')
    # _names = [key for key in musculo_p.keys() if 'lmtu_' in key]
    # plt.plot(np.diff(musculo_p[_names[0]]))
    # plt.plot(np.diff(musculo_p[_names[1]]))
    # plt.legend(tuple(_names))
    # plt.grid(True)

    # plt.figure()
    # plt.title('Ia Afferents')
    # _names = [key for key in musculo_y.keys() if 'Ia' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

    # plt.figure()
    # plt.title('Ib Afferents')
    # _names = [key for key in musculo_y.keys() if 'Ib' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

    # plt.figure()
    # plt.title('II Afferents')
    # _names = [key for key in musculo_y.keys() if 'II' in key]
    # plt.plot(musculo_y[_names])
    # plt.legend(tuple(_names))
    # plt.grid(True)

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
