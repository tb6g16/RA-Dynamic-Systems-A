# This file contains the unit tests for my_fft functions.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach A")
import unittest
import numpy as np
from Trajectory import Trajectory
from my_fft import my_fft, my_ifft
from test_cases import unit_circle as uc
from test_cases import ellipse as elps

class TestMyFFT(unittest.TestCase):

    def setUp(self):
        self.traj1 = Trajectory(uc.x)
        self.traj2 = Trajectory(elps.x)
        self.traj_rand = Trajectory(np.random.rand(3, 65))
        self.rand_time = np.random.rand(4, 32)

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.traj_rand
        del self.rand_time

    def test_my_fft_functions(self):
        # do the random noises convert back and forth properly
        self.assertTrue(np.allclose(self.traj_rand.modes, my_fft(my_ifft(self.traj_rand.modes))))
        self.assertTrue(np.allclose(self.rand_time, my_ifft(my_fft(self.rand_time))))

        # correct modes for unit circle
        traj1_modes_true = np.zeros(self.traj1.shape, dtype = complex)
        traj1_modes_true[0, 1] = 0.5
        traj1_modes_true[1, 1] = 1j*0.5
        self.assertTrue(np.allclose(self.traj1.modes, traj1_modes_true))

        # correct values for ellipse
        traj2_modes_true = np.zeros(self.traj1.shape, dtype = complex)
        traj2_modes_true[0, 1] = 1
        traj2_modes_true[1, 1] = 1j*0.5
        self.assertTrue(np.allclose(self.traj2.modes, traj2_modes_true))


if __name__ == "__main__":
    unittest.main()
