"""Unit test of Geyer muscle model."""

import unittest
from farms_muscle.geyer_muscle import GeyerMuscle
from farms_muscle.parameters import MuscleParameters
from farms_casadi_dae.casadi_dae_generator import CasadiDaeGenerator
import numpy as np


class TestGeyerMuscleForces(unittest.TestCase):
    """Test for different internal muscle forces of the geyer model.
    """

    def setUp(self):
        """ Set up the muscle model. """
        self.parameters = MuscleParameters()
        self.dae = CasadiDaeGenerator()
        self.muscle = GeyerMuscle(self.dae, self.parameters)

    def test_tendon_force_default(self):
        """ Test the default tendon force. """
        tendon_force = self.muscle._tendon_force(0.0)
        self.assertEqual(tendon_force, 0.0)

    def test_tendon_force_l_se_equal_slack(self):
        """ Test the tendon force when length is equal to slack"""
        tendon_force = self.muscle._tendon_force(
            self.muscle._l_slack.val)
        self.assertEqual(tendon_force, 0.0)

    def test_tendon_force_l_se_greater_slack_non_zero(self):
        """ Test the tendon force is non zero when 
        length is greater than slack."""
        tendon_force = self.muscle._tendon_force(
            self.muscle._l_slack.val + 0.1)
        self.assertGreater(tendon_force, 0.0)

    def test_tendon_force_l_se_greater_slack_value(self):
        """ Test the tendon force when length is greater than slack."""
        _l_slack = self.muscle._l_slack.val
        _l_se = _l_slack + _l_slack*0.1
        _strain = (_l_se - _l_slack)/_l_slack
        _f_max = self.muscle._f_max.val
        _computed_tendon_force = _f_max*(_strain/self.muscle.e_ref)**2
        tendon_force = float(self.muscle._tendon_force(_l_se))
        self.assertAlmostEqual(tendon_force, _computed_tendon_force)

    def test_parallel_star_force_default(self):
        """ Test the default parallel start force. """
        parallel_force = float(self.muscle._parallel_star_force(0.0))
        self.assertEqual(parallel_force, 0.0)

    def test_parallel_star_force_l_ce_equal_l_opt(self):
        """ Test the parallel force when l_ce and l_opt are equal."""
        _l_ce = self.muscle._l_opt.val
        parallel_force = float(self.muscle._parallel_star_force(_l_ce))
        self.assertEqual(parallel_force, 0.0)

    def test_parallel_star_force_l_ce_greater_l_opt_non_zero(self):
        """ Test parallel force is non zero when l_ce is 
        greater than l_opt """
        _l_ce = self.muscle._l_opt.val + self.muscle._l_opt.val*0.1
        parallel_force = float(self.muscle._parallel_star_force(_l_ce))
        self.assertGreater(parallel_force, 0.0)

    def test_parallel_star_force_l_ce_greater_l_opt_value(self):
        """ Test parallel force value when l_ce is greater than l_opt"""
        _l_opt = self.muscle._l_opt.val
        _l_ce = _l_opt + _l_opt*0.1
        _f_max = self.muscle._f_max.val
        _num = _l_ce - _l_opt
        _den = _l_opt * self.muscle.w
        parallel_force = float(self.muscle._parallel_star_force(_l_ce))
        self.assertAlmostEqual(parallel_force, _f_max*(_num/_den)**2)

    def test_belly_force_default(self):
        """ Test default belly force. """
        belly_force = float(self.muscle._belly_force(0.0))
        self.assertNotEqual(belly_force, 0.0)

    def test_belly_force_l_ce_equal_l_opt(self):
        """ Test belly force when l_ce is equal to l_opt """
        _l_ce = round(self.muscle._l_opt.val*(1.0 - self.muscle.w), 3)
        belly_force = float(self.muscle._belly_force(_l_ce))
        self.assertNotEqual(belly_force, 0.0)

    def test_belly_force_l_ce_greater_l_opt_zero(self):
        """ Test that belly force is zero when l_ce greater than l_opt."""
        _l_opt = self.muscle._l_opt.val
        _l_ce = _l_opt + _l_opt*0.1
        belly_force = float(self.muscle._belly_force(_l_ce))
        self.assertEqual(belly_force, 0.0)

    def test_belly_force_l_ce_lesser_l_opt_value(self):
        """ Test the belly force when l_ce is lesser than l_opt. """
        _l_opt = self.muscle._l_opt.val
        _w = self.muscle.w
        _l_ce = _l_opt*(1.0 - _w) - _l_opt*(1.0 - _w)*0.1
        _f_max = self.muscle._f_max.val
        _num = _l_ce - _l_opt*(1.0 - _w)
        _den = 0.5*_l_opt*_w
        belly_force = float(self.muscle._belly_force(_l_ce))
        self.assertAlmostEqual(belly_force, _f_max*(_num/_den)**2)

    def test_force_length_default(self):
        """ Test the force length relationship """
        _l_opt = self.muscle._l_opt.val
        _l_ce = self.muscle._l_opt.val
        _val = abs(
            (_l_ce - _l_opt) / (_l_opt * self.muscle.w))
        _exposant = GeyerMuscle.c * _val**3
        _f_l = self.muscle._force_length(_l_ce)
        self.assertAlmostEqual(_f_l, np.exp(_exposant))

    def test_force_velocity_default(self):
        """ Test the default force velocity. """
        _v_ce = 0.0
        _f_v = self.muscle._force_velocity(_v_ce)
        self.assertNotEqual(_f_v, 0.0)

    def test_force_velocity_v_ce_positive(self):
        """ Test force velocity when v_ce is positive. """
        _v_ce = 0.1
        _v_max = self.muscle._v_max.val
        _f_v_ce = (_v_max - _v_ce)/(_v_max + GeyerMuscle.K*_v_ce)
        _f_v = float(self.muscle._force_velocity(_v_ce))
        self.assertEqual(_f_v, _f_v_ce)

    def test_force_velocity_v_ce_negative(self):
        """ Test force velocity when v_ce is negative. """
        _v_ce = -0.1
        _v_max = self.muscle._v_max.val
        _N = GeyerMuscle.N
        _f_v_ce = _N + ((_N - 1)*(
            _v_max + _v_ce)/(7.56*GeyerMuscle.K*_v_ce - _v_max))
        _f_v = float(self.muscle._force_velocity(_v_ce))
        self.assertEqual(_f_v, _f_v_ce)

    def test_force_velocity_v_ce_zero(self):
        """ Test force velocity when v_ce is zero. """
        _v_ce = 0.0
        _v_max = self.muscle._v_max.val
        _f_v_ce = (_v_max - _v_ce)/(_v_max + GeyerMuscle.K*_v_ce)
        _f_v = float(self.muscle._force_velocity(_v_ce))
        self.assertEqual(_f_v, _f_v_ce)


if __name__ == '__main__':
    unittest.main()
