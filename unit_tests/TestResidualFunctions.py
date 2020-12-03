# This file contains the tests for the residual calculation functions defined
# in the residual_functions file

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
from System import System
import residual_functions as res_funcs
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

class TestResidualFunctions(unittest.TestCase):
    
    def setUp(self):
        self.traj1 = Trajectory(uc.x)
        self.freq1 = 1
        self.traj2 = Trajectory(elps.x)
        self.freq2 = 1
        self.sys1 = System(vpd)
        self.sys2 = System(vis)

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.sys1
        del self.sys2

    def test_local_residual(self):
        # NEED TO DO EQUIVALENT FOR ELLIPSE TRAJECTORY

        # generating random frequency and system parameters
        freq1 = rand.uniform(-10, 10)
        # freq2 = rand.uniform(-10, 10)
        mu1 = rand.uniform(-10, 10)
        mu2 = rand.uniform(-10, 10)
        r = rand.uniform(-10, 10)

        # apply parameters
        self.sys1.parameters['mu'] = mu1
        self.sys2.parameters['mu'] = mu2
        self.sys2.parameters['r'] = r

        # generate local residual trajectories
        lr_traj1_sys1 = res_funcs.local_residual(self.traj1, self.sys1, freq1)
        # lr_traj2_sys1 = res_funcs.local_residual(self.traj2, self.sys1, freq2)
        lr_traj1_sys2 = res_funcs.local_residual(self.traj1, self.sys2, freq1)
        # lr_traj2_sys2 = res_funcs.local_residual(self.traj2, self.sys2, freq2)

        # output is of Trajectory class
        self.assertIsInstance(lr_traj1_sys1, Trajectory)
        # self.assertIsInstance(lr_traj2_sys1, Trajectory)
        self.assertIsInstance(lr_traj1_sys2, Trajectory)
        # self.assertIsInstance(lr_traj2_sys2, Trajectory)

        # output is of correct shape
        self.assertEqual(lr_traj1_sys1.curve_array.shape, self.traj1.curve_array.shape)
        # self.assertEqual(lr_traj2_sys1.curve_array.shape, self.traj2.curve_array.shape)
        self.assertEqual(lr_traj1_sys2.curve_array.shape, self.traj1.curve_array.shape)
        # self.assertEqual(lr_traj2_sys2.curve_array.shape, self.traj2.curve_array.shape)

        # outputs are numbers
        temp = True
        if lr_traj1_sys1.curve_array.dtype != np.int64 and lr_traj1_sys1.curve_array.dtype != np.float64:
            temp = False
        # if lr_traj2_sys1.curve_array.dtype != np.int64 and lr_traj2_sys1.curve_array.dtype != np.float64:
            # temp = False
        if lr_traj1_sys2.curve_array.dtype != np.int64 and lr_traj1_sys2.curve_array.dtype != np.float64:
            temp = False
        # if lr_traj2_sys2.curve_array.dtype != np.int64 and lr_traj2_sys2.curve_array.dtype != np.float64:
            # temp = False
        self.assertTrue(temp)

        # correct values
        lr_traj1_sys1_true = np.zeros(self.traj1.curve_array.shape)
        # lr_traj2_sys1_true = np.zeros(self.traj2.curve_array.shape)
        lr_traj1_sys2_true = np.zeros(self.traj1.curve_array.shape)
        # lr_traj2_sys2_true = np.zeros(self.traj2.curve_array.shape)
        for i in range(np.shape(self.traj1.curve_array)[1]):
            s = ((2*np.pi)/np.shape(self.traj1.curve_array)[1])*i
            lr_traj1_sys1_true[0, i] = (1 - freq1)*np.sin(s)
            lr_traj1_sys1_true[1, i] = (mu1*(1 - (np.cos(s)**2))*np.sin(s)) + ((1 - freq1)*np.cos(s))
            lr_traj1_sys2_true[0, i] = ((1 - freq1)*np.sin(s)) - (mu2*np.cos(s)*((r**2) - 1))
            lr_traj1_sys2_true[1, i] = ((1 - freq1)*np.cos(s)) + (mu2*np.sin(s)*((r**2) - 1))
        lr_traj1_sys1_true = Trajectory(lr_traj1_sys1_true)
        lr_traj1_sys2_true = Trajectory(lr_traj1_sys2_true)
        # for i in range(np.shape(self.traj2.curve_array)[1]):
            # lr_traj2_sys1_true[0, i] = 1 PLACEHOLDER FOR FUNCTION
            # lr_traj2_sys1_true[1, i] = 1 PLACEHOLDER FOR FUNCTION
            # lr_traj2_sys2_true[0, i] = 1 PLACEHOLDER FOR FUNCTION
            # lr_traj2_sys2_true[1, i] = 1 PLACEHOLDER FOR FUNCTION
        self.assertEqual(lr_traj1_sys1, lr_traj1_sys1_true)
        # self.assertEqual(lr_traj2_sys1, lr_traj2_sys1_true)
        self.assertEqual(lr_traj1_sys2, lr_traj1_sys2_true)
        # self.assertEqual(lr_traj2_sys2, lr_traj2_sys2_true)

    def test_global_residual(self):
        pass
        # gr_traj1_sys1 = res_funcs.global_residual(self.traj1, self.sys1, self.freq1)
        # gr_traj2_sys1 = res_funcs.global_residual(self.traj2, self.sys1, self.freq2)
        # gr_traj1_sys2 = res_funcs.global_residual(self.traj1, self.sys2, self.freq1)
        # gr_traj2_sys2 = res_funcs.global_residual(self.traj2, self.sys2, self.freq2)

        # output is a positive number

        # correct values

    def test_global_residual_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()