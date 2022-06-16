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

from libc.stdio cimport printf
from libc.math cimport sqrt as csqrt
from libc.math cimport sin as csin
from libc.math cimport pow as cpow
from libc.math cimport cos as ccos
from libc.math cimport atan as catan
from libc.math cimport acos as cacos
from libc.math cimport fmax as cfmax
from libc.math cimport fabs as cfabs
from libc.math cimport log as clog
from libc.math cimport exp as cexp
from libc.math cimport acosh as cacosh
import numpy as np
cimport numpy as cnp


def mjcb_muscle_gain(mj_model, mj_data, mj_id):
    l_mtu = mj_data.actuator_length[mj_id]
    v_mtu = mj_data.actuator_velocity[mj_id]
    gainprm = mj_model.actuator_gainprm[mj_id]
    f_max = gainprm[0]
    l_opt = gainprm[1]
    l_slack = gainprm[2]
    v_max = gainprm[3]
    alpha = gainprm[4]
    return c_active_force(l_mtu, v_mtu, l_opt, l_slack, alpha, f_max, v_max)


def mjcb_muscle_bias(mj_model, mj_data, mj_id):
    l_mtu = mj_data.actuator_length[mj_id]
    v_mtu = mj_data.actuator_velocity[mj_id]
    gainprm = mj_model.actuator_gainprm[mj_id]
    f_max = gainprm[0]
    l_opt = gainprm[1]
    l_slack = gainprm[2]
    v_max = gainprm[3]
    alpha = gainprm[4]
    pf = c_passive_force(l_mtu, v_mtu, l_opt, l_slack, alpha, f_max, v_max)
    # print(pf)
    return pf


cdef inline double c_active_force(
    double l_mtu, double v_mtu, double l_opt, double l_slack,
    double alpha, double f_max, double v_max
) nogil:
    """ Compute the active force. """
    cdef double l_ce_norm = c_fiber_length(l_mtu, l_slack, alpha)/l_opt
    cdef double v_ce_norm = c_fiber_velocity(v_mtu, alpha)/v_max
    return -1*f_max*c_force_length(l_ce_norm)*c_force_velocity(v_ce_norm)


cdef inline double c_passive_force(
    double l_mtu, double v_mtu, double l_opt, double l_slack,
    double alpha, double f_max, double v_max
) nogil:
    """ passive-force computation """

    # passive-force constants
    kpe = 4.0
    e0 = 0.6

    cdef double l_ce_norm = c_fiber_length(l_mtu, l_slack, alpha)/l_opt
    # printf("%f \n", l_ce_norm)
    cdef double _den = cexp(kpe) - 1.0
    cdef double _num = cexp((kpe*(l_ce_norm) - kpe)/e0) - 1.0
    return -1*(_num/_den) if l_ce_norm >= 1.0 else 0.0


cdef inline double c_force_length(double l_ce) nogil:
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


cdef inline double c_force_velocity(double v_ce) nogil:
    """ force-velocity computation """

    # force-velocity constants
    d1 = -0.318
    d2 = -8.149
    d3 = -0.374
    d4 = 0.886

    cdef double exp1 = d2*v_ce + d3
    cdef double exp2 = ((d2*v_ce + d3)**2) + 1.
    return d1*clog(exp1 + csqrt(exp2)) + d4


cdef inline double c_fiber_velocity(double v_mtu, double alpha) nogil:
    """ Compute the fiber velocity. """
    return v_mtu/ccos(2*alpha)


cdef inline double c_fiber_length(double l_mtu, double l_slack, double alpha) nogil:
    """ Compute the fiber length. """
    return (l_mtu - l_slack)/ccos(alpha)
