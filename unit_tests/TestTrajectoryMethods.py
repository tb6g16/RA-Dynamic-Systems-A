# This file contains the tests for the Trajectory class and its associated
# methods

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
import trajectory_functions as traj_funcs
from System import System
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

class TestTrajectoryMethods(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_func2array(self):
        pass
        # output correct size
        # self.assertEqual(self.traj1.curve_array.shape, (2, 64))
        # self.assertEqual(self.traj2.curve_array.shape, (2, 64))

        # # outputs are numbers
        # self.assertTrue(self.traj1.curve_array.dtype == np.int64 or \
        #     self.traj1.curve_array.dtype == np.float64)
        # self.assertTrue(self.traj2.curve_array.dtype == np.int64 or \
        #     self.traj2.curve_array.dtype == np.float64)

        # # correct values
        # rindex1 = int(rand.random()*np.shape(self.traj1.curve_array)[1])
        # rindex2 = int(rand.random()*np.shape(self.traj2.curve_array)[1])
        # rs1 = ((2*np.pi)/(np.shape(self.traj1.curve_array)[1]))*rindex1
        # rs2 = ((2*np.pi)/(np.shape(self.traj2.curve_array)[1]))*rindex2
        # traj1_val = self.traj1.curve_array[:, rindex1]
        # traj2_val = self.traj2.curve_array[:, rindex2]
        # traj1_true = uc.x(rs1)
        # traj2_true = elps.x(rs2)
        # self.assertTrue(np.allclose(traj1_val, traj1_true))
        # self.assertTrue(np.allclose(traj2_val, traj2_true))

    def test_summing(self):
        # test both __add__ and __sub__
        pass

    def test_mul(self):
        # test both __mul__ and __rmul__
        pass

    def test_pow(self):
        pass

    def test_eq(self):
        pass
