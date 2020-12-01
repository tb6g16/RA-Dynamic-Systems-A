# This file contains the tests for the residual calculation functions defined
# in the residual_functions file

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
import unittest
import numpy as np
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
        lr_traj1_sys1 = res_funcs.local_residual(self.traj1, self.sys1, self.freq1)
        lr_traj2_sys1 = res_funcs.local_residual(self.traj2, self.sys1, self.freq2)
        lr_traj1_sys2 = res_funcs.local_residual(self.traj1, self.sys2, self.freq1)
        lr_traj2_sys2 = res_funcs.local_residual(self.traj2, self.sys2, self.freq2)

        # output is of Trajectory class

        # output is of correct shape

        # outputs are numbers

        # correct values
        # NEED TO COMPUTE DIFFERENCE ARRAYS FOR THE TRUE AND CALCULATED ANSWERS
        # FOR ALL THE DIFFERENT PARAMETERS

    def test_global_residual(self):
        # NEED TO COMPUTE DIFFERENCE ARRAYS FOR THE TRUE AND CALCULATED ANSWERS
        # FOR ALL THE DIFFERENT PARAMETERS
        gr_traj1_sys1 = res_funcs.global_residual(self.traj1, self.sys1, self.freq1)
        gr_traj2_sys1 = res_funcs.global_residual(self.traj2, self.sys1, self.freq2)
        gr_traj1_sys2 = res_funcs.global_residual(self.traj1, self.sys2, self.freq1)
        gr_traj2_sys2 = res_funcs.global_residual(self.traj2, self.sys2, self.freq2)

        # output is a positive number

        # correct values

    def test_global_residual_grad(self):
        pass