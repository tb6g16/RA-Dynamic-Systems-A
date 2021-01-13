# This file contains the unit test for the optimise file that initialises the
# objective function, constraints, and all their associated gradients.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach A")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
from System import System
import trajectory_functions as traj_funcs
import residual_functions as res_funcs
import traj2vec as t2v
import optimise as opt
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

class TestOptimise(unittest.TestCase):

    def setUp(self):
        self.traj1 = Trajectory(uc.x, disc = 64)
        self.freq1 = rand.uniform(-10, 10)
        self.traj1_vec = t2v.traj2vec(self.traj1, self.freq1)
        self.traj2 = Trajectory(elps.x, disc = 64)
        self.freq2 = rand.uniform(-10, 10)
        self.traj2_vec = t2v.traj2vec(self.traj2, self.freq2)
        self.sys1 = System(vpd)
        self.sys2 = System(vis)

    def tearDown(self):
        del self.traj1
        del self.traj1_vec
        del self.traj2
        del self.traj2_vec
        del self.sys1
        del self.sys2

    def test_traj_global_res(self):
        res_func_s1, _ = opt.init_opt_funcs(self.sys1, 2, np.zeros([2, 1]))
        res_func_s2, _ = opt.init_opt_funcs(self.sys2, 2, np.zeros([2, 1]))
        gr_t1s1 = res_func_s1(self.traj1_vec)
        gr_t2s1 = res_func_s1(self.traj2_vec)
        gr_t1s2 = res_func_s2(self.traj1_vec)
        gr_t2s2 = res_func_s2(self.traj2_vec)
        
        # real number
        temp = True
        if type(gr_t1s1) != np.float64 and type(gr_t1s1) != float:
            temp = False
        if type(gr_t2s1) != np.float64 and type(gr_t2s1) != float:
            temp = False
        if type(gr_t1s2) != np.float64 and type(gr_t1s2) != float:
            temp = False
        if type(gr_t2s2) != np.float64 and type(gr_t2s2) != float:
            temp = False
        self.assertTrue(temp) 

        # correct value
        gr_t1s1_true = res_funcs.global_residual(self.traj1, self.sys1, self.freq1, np.zeros([2, 1]))
        gr_t2s1_true = res_funcs.global_residual(self.traj2, self.sys1, self.freq2, np.zeros([2, 1]))
        gr_t1s2_true = res_funcs.global_residual(self.traj1, self.sys2, self.freq1, np.zeros([2, 1]))
        gr_t2s2_true = res_funcs.global_residual(self.traj2, self.sys2, self.freq2, np.zeros([2, 1]))
        self.assertEqual(gr_t1s1, gr_t1s1_true)
        self.assertEqual(gr_t2s1, gr_t2s1_true)
        self.assertEqual(gr_t1s2, gr_t1s2_true)
        self.assertEqual(gr_t2s2, gr_t2s2_true)

    def test_traj_global_res_jac(self):
        _, res_grad_func_s1 = opt.init_opt_funcs(self.sys1, 2, np.zeros([2, 1]))
        _, res_grad_func_s2 = opt.init_opt_funcs(self.sys2, 2, np.zeros([2, 1]))
        gr_traj_t1s1, gr_freq_t1s1 = t2v.vec2traj(res_grad_func_s1(self.traj1_vec), 2)
        gr_traj_t2s1, gr_freq_t2s1 = t2v.vec2traj(res_grad_func_s1(self.traj2_vec), 2)
        gr_traj_t1s2, gr_freq_t1s2 = t2v.vec2traj(res_grad_func_s2(self.traj1_vec), 2)
        gr_traj_t2s2, gr_freq_t2s2 = t2v.vec2traj(res_grad_func_s2(self.traj2_vec), 2)

        # correct values
        gr_traj_t1s1_true, gr_freq_t1s1_true = res_funcs.global_residual_grad(self.traj1, self.sys1, self.freq1, np.zeros([2, 1]))
        gr_traj_t2s1_true, gr_freq_t2s1_true = res_funcs.global_residual_grad(self.traj2, self.sys1, self.freq2, np.zeros([2, 1]))
        gr_traj_t1s2_true, gr_freq_t1s2_true = res_funcs.global_residual_grad(self.traj1, self.sys2, self.freq1, np.zeros([2, 1]))
        gr_traj_t2s2_true, gr_freq_t2s2_true = res_funcs.global_residual_grad(self.traj2, self.sys2, self.freq2, np.zeros([2, 1]))
        self.assertEqual(gr_traj_t1s1, gr_traj_t1s1_true)
        self.assertEqual(gr_traj_t2s1, gr_traj_t2s1_true)
        self.assertEqual(gr_traj_t1s2, gr_traj_t1s2_true)
        self.assertEqual(gr_traj_t2s2, gr_traj_t2s2_true)
        self.assertEqual(gr_freq_t1s1, gr_freq_t1s1_true)
        self.assertEqual(gr_freq_t2s1, gr_freq_t2s1_true)
        self.assertEqual(gr_freq_t1s2, gr_freq_t1s2_true)
        self.assertEqual(gr_freq_t2s2, gr_freq_t2s2_true)

    def test_constraints(self):
        pass

    def test_constraints_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()