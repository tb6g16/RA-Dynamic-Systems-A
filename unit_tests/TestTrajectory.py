# This file contains the testing methods for the Trajectory class and its
# contained methods.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
import unittest
import numpy as np
from Trajectory import Trajectory
from System import System
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd

class TestTrajectory(unittest.TestCase):
    
    def test_traj_prod(self):
        traj1 = Trajectory(uc.x)
        traj2 = Trajectory(elps.x)
        new_traj1 = traj1.traj_prod(traj2)
        new_traj2 = traj2.traj_prod(traj1)

        # new trajectories are instances of Trajectory class
        self.assertTrue(isinstance(new_traj1, Trajectory))
        self.assertTrue(isinstance(new_traj2, Trajectory))

        # does the operation commute
        self.assertEqual(new_traj1, new_traj2)

        # inner product equal to norm
        self.assertEqual(traj1.normed_traj(), traj1.traj_prod(traj1))

        # single number at each index
        temp = True
        for i in range(np.shape(new_traj1.curve_array)[1]):
            if new_traj1.curve_array[:, i].shape[0] != 1:
                temp = False
        self.assertTrue(temp)

    def test_gradient(self):
        pass

    def test_traj_response(self):
        pass

    def test_traj_nl_response(self):
        pass

    def test_jacob_init(self):
        pass

    def test_normed_traj(self):
        pass

    def test_average_over_s(self):
        pass

if __name__ == "__main__":
    unittest.main()