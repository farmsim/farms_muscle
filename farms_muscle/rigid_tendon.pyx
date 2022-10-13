# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

""" Call back functions for mujoco """

from libc.math cimport acos as cacos
from libc.math cimport acosh as cacosh
from libc.math cimport atan as catan
from libc.math cimport cos as ccos
from libc.math cimport exp as cexp
from libc.math cimport fabs as cfabs
from libc.math cimport fmax as cfmax
from libc.math cimport log as clog
from libc.math cimport pow as cpow
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.stdio cimport printf

import numpy as np

cimport numpy as cnp


def mjcb_muscle_gain(mj_model, mj_data, mj_id):
    """ mujoco callback function interface for actuator gain.
    In the hill-type formulation the active force component is represented
    as the actuator gain in mujoco
    """
    l_mtu = mj_data.actuator_length[mj_id]
    v_mtu = mj_data.actuator_velocity[mj_id]
    gainprm = mj_model.actuator_gainprm[mj_id]
    f_max = gainprm[0]
    l_opt = gainprm[1]
    l_slack = gainprm[2]
    v_max = gainprm[3]
    alpha_opt = gainprm[4]
    alpha = c_pennation_angle(l_mtu, l_opt, l_slack, alpha_opt)
    l_ce_norm = c_fiber_length(l_mtu, l_slack, alpha)/l_opt
    v_ce_norm = c_fiber_velocity(v_mtu, alpha)/v_max
    return -f_max*c_active_force(l_ce_norm, v_ce_norm, alpha)


def mjcb_muscle_bias(mj_model, mj_data, mj_id):
    """ mujoco callback function interface for actuator bias.
    In the hill-type formulation the passive force component is represented
    as the actuator bias in mujoco
    """
    l_mtu = mj_data.actuator_length[mj_id]
    v_mtu = mj_data.actuator_velocity[mj_id]
    gainprm = mj_model.actuator_gainprm[mj_id]
    f_max = gainprm[0]
    l_opt = gainprm[1]
    l_slack = gainprm[2]
    v_max = gainprm[3]
    alpha_opt = gainprm[4]
    alpha = c_pennation_angle(l_mtu, l_opt, l_slack, alpha_opt)
    l_ce_norm = c_fiber_length(l_mtu, l_slack, alpha)/l_opt
    v_ce_norm = c_fiber_velocity(v_mtu, alpha)/v_max
    pf = c_passive_force(l_ce_norm, v_ce_norm, alpha)
    damping = c_damping_force(l_ce_norm, v_ce_norm, alpha)
    return -f_max*(pf + damping)


cpdef inline double c_active_force(
    double l_ce_norm, double v_ce_norm, double alpha,
) nogil:
    """ Compute the active force. """
    return ccos(alpha)*c_force_length(l_ce_norm)*c_force_velocity(v_ce_norm)


cpdef inline double c_passive_force(
    double l_ce_norm, double v_ce_norm, double alpha
) nogil:
    """ passive-force computation """

    # passive-force constants
    kpe = 4.0
    e0 = 0.6
    cdef double _den = cexp(kpe) - 1.0
    cdef double _num = cexp((kpe*(l_ce_norm) - kpe)/e0) - 1.0
    return ccos(alpha)*(_num/_den) if l_ce_norm >= 1.0 else 0.0


cpdef inline double c_damping_force(
    double l_ce_norm, double v_ce_norm, double alpha
) nogil:
    """ Muscle damping """
    return 1e-1*ccos(alpha)*v_ce_norm


cpdef inline double c_force_length(double l_ce) nogil:
    """ force-length computation. """

    # force-length constants
    cdef double[3] b1, b2, b3, b4
    b1[0] = 0.815
    b2[0] = 1.055
    b3[0] = 0.162
    b4[0] = 0.063

    b1[1] = 0.433
    b2[1] = 0.717
    b3[1] = -0.030
    b4[1] = 0.2

    b1[2] = 0.1
    b2[2] = 1.0
    b3[2] = 0.354
    b4[2] = 0.0

    cdef double _force_length = 0.0
    cdef unsigned int j = 0

    for j in range(3):
        _force_length += b1[j]*cexp(
            (-0.5*(l_ce - b2[j])**2)/((b3[j] + b4[j]*l_ce)**2)
        )
    return _force_length


cpdef inline double c_force_velocity(double v_ce) nogil:
    """ force-velocity computation """

    # force-velocity constants
    d1 = -0.318
    d2 = -8.149
    d3 = -0.374
    d4 = 0.886

    cdef double exp1 = d2*v_ce + d3
    cdef double exp2 = ((d2*v_ce + d3)**2) + 1.
    return d1*clog(exp1 + csqrt(exp2)) + d4


cpdef inline double c_pennation_angle(
    double l_mtu, double l_opt, double l_slack, double alpha_opt
) nogil:
    """ Calculate pennation angle """
    # Not sure if this method is better over the other implementation??
    # cdef double parallelogram_height = l_opt*csin(alpha_opt)
    # cdef double zero_pennate_fiber_length = (l_mtu - l_slack)
    # zero_pennate_fiber_length *= (zero_pennate_fiber_length > 0.0)
    # # compute angle
    # cdef double fiber_length = (
    #     csqrt(zero_pennate_fiber_length**2 + parallelogram_height**2)
    # )
    # return cacos(zero_pennate_fiber_length/fiber_length)
    return catan((l_opt*csin(alpha_opt))/(l_mtu-l_slack))


cpdef inline double c_fiber_velocity(double v_mtu, double alpha) nogil:
    """ Compute the fiber velocity. """
    return v_mtu*ccos(alpha)


cpdef inline double c_fiber_length(double l_mtu, double l_slack, double alpha) nogil:
    """ Compute the fiber length. """
    return (l_mtu - l_slack)/ccos(alpha)
