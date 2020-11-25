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
from test_cases import viswanath as vis

class TestTrajectory(unittest.TestCase):
    
    def test_traj_prod(self):
        traj1 = Trajectory(uc.x)
        traj2 = Trajectory(elps.x)
        new_traj1 = traj1.traj_prod(traj2)
        new_traj2 = traj2.traj_prod(traj1)

        # output is of the Trajectory class
        self.assertIsInstance(new_traj1, Trajectory)
        self.assertIsInstance(new_traj2, Trajectory)

        # does the operation commute
        self.assertEqual(new_traj1, new_traj2)

        # inner product equal to norm
        self.assertEqual(traj1.normed_traj()**2, traj1.traj_prod(traj1))
        self.assertEqual(traj2.normed_traj()**2, traj2.traj_prod(traj2))

        # single number at each index
        temp = True
        for i in range(np.shape(new_traj1.curve_array)[1]):
            if new_traj1.curve_array[:, i].shape[0] != 1:
                temp = False
        self.assertTrue(temp)

    def test_gradient(self):
        traj1 = Trajectory(uc.x)
        traj2 = Trajectory(elps.x)

        # same shape as original trajectories
        self.assertEqual(traj1.curve_array.shape, traj1.grad.curve_array.shape)
        self.assertEqual(traj2.curve_array.shape, traj2.grad.curve_array.shape)

        # correct values
        traj1_grad = np.zeros(traj1.curve_array.shape)
        traj2_grad = np.zeros(traj2.curve_array.shape)
        for i in range(traj2.curve_array.shape[1]):
            s = ((2*np.pi)/traj2.curve_array.shape[1])*i
            traj1_grad[0, i] = -np.sin(s)
            traj1_grad[1, i] = -np.cos(s)
            traj2_grad[0, i] = -2*np.sin(s)
            traj2_grad[1, i] = -np.cos(s)
        traj1_grad = Trajectory(traj1_grad)
        traj2_grad = Trajectory(traj2_grad)
        self.assertEqual(traj1_grad, traj1.grad)
        self.assertEqual(traj2_grad, traj2.grad)

    def test_traj_response(self):
        traj1 = Trajectory(uc.x)
        traj2 = Trajectory(elps.x)
        sys1 = System(vpd)
        sys2 = System(vis)
        traj1_response1 = traj1.traj_response(sys1)
        traj1_response2 = traj1.traj_response(sys2)
        traj2_response1 = traj2.traj_response(sys1)
        traj2_response2 = traj2.traj_response(sys2)
        
        # output is of the Trajectory class
        self.assertIsInstance(traj1_response1, Trajectory)
        self.assertIsInstance(traj1_response2, Trajectory)
        self.assertIsInstance(traj2_response1, Trajectory)
        self.assertIsInstance(traj2_response2, Trajectory)

        # same response for trajectories at crossing points
        cross_i1 = int(((traj1.curve_array.shape[1])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((traj1.curve_array.shape[1])/(2*np.pi))*((3*np.pi)/2))
        traj1_cross1_resp1 = traj1_response1.curve_array[:, cross_i1]
        traj2_cross1_resp1 = traj2_response1.curve_array[:, cross_i1]
        traj1_cross2_resp1 = traj1_response1.curve_array[:, cross_i2]
        traj2_cross2_resp1 = traj2_response1.curve_array[:, cross_i2]
        traj1_cross1_resp2 = traj1_response2.curve_array[:, cross_i1]
        traj2_cross1_resp2 = traj2_response2.curve_array[:, cross_i1]
        traj1_cross2_resp2 = traj1_response2.curve_array[:, cross_i2]
        traj2_cross2_resp2 = traj2_response2.curve_array[:, cross_i2]
        self.assertTrue(np.allclose(traj1_cross1_resp1, traj2_cross1_resp1))
        self.assertTrue(np.allclose(traj1_cross2_resp1, traj2_cross2_resp1))
        self.assertTrue(np.allclose(traj1_cross1_resp2, traj2_cross1_resp2))
        self.assertTrue(np.allclose(traj1_cross2_resp2, traj2_cross2_resp2))

        # gradient of solution equal to response
        sys2.parameters['mu'] = 1
        self.assertEqual(traj1.grad, traj1_response1)
        self.assertEqual(traj1.grad, traj1_response2)

    def test_traj_nl_response(self):
        traj1 = Trajectory(uc.x)
        traj2 = Trajectory(elps.x)
        sys1 = System(vpd)
        sys2 = System(vis)
        traj1_response1 = traj1.traj_nl_response(sys1)
        traj1_response2 = traj1.traj_nl_response(sys2)
        traj2_response1 = traj2.traj_nl_response(sys1)
        traj2_response2 = traj2.traj_nl_response(sys2)

        # output is of the Trajectory class
        self.assertIsInstance(traj1_response1, Trajectory)
        self.assertIsInstance(traj1_response2, Trajectory)
        self.assertIsInstance(traj2_response1, Trajectory)
        self.assertIsInstance(traj2_response2, Trajectory)

        # same response for trajectories at crossing points
        cross_i1 = int(((traj1.curve_array.shape[1])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((traj1.curve_array.shape[1])/(2*np.pi))*((3*np.pi)/2))
        traj1_cross1_resp1 = traj1_response1.curve_array[:, cross_i1]
        traj2_cross1_resp1 = traj2_response1.curve_array[:, cross_i1]
        traj1_cross2_resp1 = traj1_response1.curve_array[:, cross_i2]
        traj2_cross2_resp1 = traj2_response1.curve_array[:, cross_i2]
        traj1_cross1_resp2 = traj1_response2.curve_array[:, cross_i1]
        traj2_cross1_resp2 = traj2_response2.curve_array[:, cross_i1]
        traj1_cross2_resp2 = traj1_response2.curve_array[:, cross_i2]
        traj2_cross2_resp2 = traj2_response2.curve_array[:, cross_i2]
        self.assertTrue(np.allclose(traj1_cross1_resp1, traj2_cross1_resp1))
        self.assertTrue(np.allclose(traj1_cross2_resp1, traj2_cross2_resp1))
        self.assertTrue(np.allclose(traj1_cross1_resp2, traj2_cross1_resp2))
        self.assertTrue(np.allclose(traj1_cross2_resp2, traj2_cross2_resp2))

    def test_jacob_init(self):
        pass

    def test_normed_traj(self):
        pass

    def test_average_over_s(self):
        pass

if __name__ == "__main__":
    unittest.main()
