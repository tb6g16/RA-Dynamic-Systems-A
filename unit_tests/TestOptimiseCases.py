# This file contains the unit tests designed to see whether this approach
# correctly finds the expected solutions to certain test cases/systems.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach A")
import unittest
import numpy as np
import scipy.integrate as integ
import random as rand
from Trajectory import Trajectory
from System import System
import residual_functions as res_funcs
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

class TestOptimiseCases(unittest.TestCase):

    def test_case1(self):
        """
            Perturbed point
        """
        pass

    def test_case2(self):
        """
            Random noise
        """
        pass

    def test_case3(self):
        """
            Solve VPD from unit circle
        """
        pass

    def test_case4(self):
        """
            Solve Viswanath from VPD limit cycle solution
        """
        pass

    def test_case5(self):
        """
            Lorenz/Rossler system?
        """
        pass


if __name__ == "__main__":
    unittest.main()
