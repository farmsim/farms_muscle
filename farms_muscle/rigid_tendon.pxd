cpdef double c_active_force(
    double l_ce_norm, double v_ce_norm, double alpha,
)
cpdef double c_passive_force(
    double l_ce_norm, double v_ce_norm, double alpha
)
cpdef double c_damping_force(
    double l_ce_norm, double v_ce_norm, double alpha, double damping_factor
)
cpdef double c_force_length(double l_ce)
cpdef double c_force_velocity(double v_ce_norm)
cpdef double c_pennation_angle(
    double l_mtu, double l_opt, double l_slack, double alpha_opt
)
cpdef double c_fiber_velocity(double v_mtu, double alpha)
cpdef double c_fiber_length(double l_mtu, double l_slack, double alpha)
