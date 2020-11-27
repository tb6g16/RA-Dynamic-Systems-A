# This file contains the testing methods for the Trajectory class and its
# contained methods.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\RA Dynamical System")
import unittest
import numpy as np
import random as rand
from Trajectory import Trajectory
from System import System
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

class TestTrajectory(unittest.TestCase):

    def setUp(self):
        a = 1
        self.traj1 = Trajectory(uc.x)
        self.traj2 = Trajectory(elps.x)
        self.traj3 = Trajectory(self.traj1.curve_array + a)
        self.sys1 = System(vpd)
        self.sys1.parameters['mu'] = 0
        self.sys2 = System(vis)
        self.sys2.parameters['mu'] = 0

    def tearDown(self):
        self.sys1.parameters['mu'] = 0
        self.sys2.parameters['mu'] = 0

    def test_func2array(self):
        pass

    def test_traj_prod(self):
        new_traj1 = self.traj1.traj_prod(self.traj2)
        new_traj2 = self.traj2.traj_prod(self.traj1)

        # output is of the Trajectory class
        self.assertIsInstance(new_traj1, Trajectory)
        self.assertIsInstance(new_traj2, Trajectory)

        # does the operation commute
        self.assertEqual(new_traj1, new_traj2)

        # inner product equal to norm
        self.assertEqual(self.traj1.normed_traj()**2, \
            self.traj1.traj_prod(self.traj1))
        self.assertEqual(self.traj2.normed_traj()**2, \
            self.traj2.traj_prod(self.traj2))

        # single number at each index
        temp1 = True
        for i in range(np.shape(new_traj1.curve_array)[1]):
            if new_traj1.curve_array[:, i].shape[0] != 1:
                temp1 = False
        for i in range(new_traj2.curve_array.shape[1]):
            if new_traj2.curve_array[:, i].shape[0] != 1:
                temp1 = False
        self.assertTrue(temp1)

        # outputs are numbers
        temp2 = True
        if new_traj1.curve_array.dtype != np.int64 and \
            new_traj1.curve_array.dtype != np.float64:
            temp2 = False
        if new_traj2.curve_array.dtype != np.int64 and \
            new_traj2.curve_array.dtype != np.float64:
            temp2 = False
        self.assertTrue(temp2)

    def test_gradient(self):
        # same shape as original trajectories
        self.assertEqual(self.traj1.curve_array.shape, \
            self.traj1.grad.curve_array.shape)
        self.assertEqual(self.traj2.curve_array.shape, \
            self.traj2.grad.curve_array.shape)

        # outputs are real numbers
        temp = True
        if self.traj1.grad.curve_array.dtype != np.int64 and \
            self.traj1.grad.curve_array.dtype != np.float64:
            temp = False
        if self.traj2.grad.curve_array.dtype != np.int64 and \
            self.traj2.grad.curve_array.dtype != np.float64:
            temp = False
        self.assertTrue(temp)

        # correct values
        traj1_grad = np.zeros(self.traj1.curve_array.shape)
        traj2_grad = np.zeros(self.traj2.curve_array.shape)
        for i in range(self.traj2.curve_array.shape[1]):
            s = ((2*np.pi)/self.traj2.curve_array.shape[1])*i
            traj1_grad[0, i] = -np.sin(s)
            traj1_grad[1, i] = -np.cos(s)
            traj2_grad[0, i] = -2*np.sin(s)
            traj2_grad[1, i] = -np.cos(s)
        traj1_grad = Trajectory(traj1_grad)
        traj2_grad = Trajectory(traj2_grad)
        self.assertEqual(traj1_grad, self.traj1.grad)
        self.assertEqual(traj2_grad, self.traj2.grad)

    def test_traj_response(self):
        traj1_response1 = self.traj1.traj_response(self.sys1)
        traj1_response2 = self.traj1.traj_response(self.sys2)
        traj2_response1 = self.traj2.traj_response(self.sys1)
        traj2_response2 = self.traj2.traj_response(self.sys2)
        
        # output is of the Trajectory class
        self.assertIsInstance(traj1_response1, Trajectory)
        self.assertIsInstance(traj1_response2, Trajectory)
        self.assertIsInstance(traj2_response1, Trajectory)
        self.assertIsInstance(traj2_response2, Trajectory)

        # outputs are numbers
        temp = True
        if traj1_response1.curve_array.dtype != np.int64 and \
            traj1_response1.curve_array.dtype != np.float64:
            temp = False
        if traj1_response2.curve_array.dtype != np.int64 and \
            traj1_response2.curve_array.dtype != np.float64:
            temp = False
        if traj1_response2.curve_array.dtype != np.int64 and \
            traj1_response2.curve_array.dtype != np.float64:
            temp = False
        if traj2_response2.curve_array.dtype != np.int64 and \
            traj2_response2.curve_array.dtype != np.float64:
            temp = False
        self.assertTrue(temp)

        # same response for trajectories at crossing points
        cross_i1 = int(((self.traj1.curve_array.shape[1])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((self.traj1.curve_array.shape[1])/(2*np.pi))*((3*np.pi)/2))
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
        self.sys2.parameters['mu'] = 1
        self.assertEqual(self.traj1.grad, traj1_response1)
        self.assertEqual(self.traj1.grad, traj1_response2)

    def test_traj_nl_response(self):
        traj1_response1 = self.traj1.traj_nl_response(self.sys1)
        traj1_response2 = self.traj1.traj_nl_response(self.sys2)
        traj2_response1 = self.traj2.traj_nl_response(self.sys1)
        traj2_response2 = self.traj2.traj_nl_response(self.sys2)

        # output is of the Trajectory class
        self.assertIsInstance(traj1_response1, Trajectory)
        self.assertIsInstance(traj1_response2, Trajectory)
        self.assertIsInstance(traj2_response1, Trajectory)
        self.assertIsInstance(traj2_response2, Trajectory)

        # outputs are numbers
        temp = True
        if traj1_response1.curve_array.dtype != np.int64 and \
            traj1_response1.curve_array.dtype != np.float64:
            temp = False
        if traj1_response2.curve_array.dtype != np.int64 and \
            traj1_response2.curve_array.dtype != np.float64:
            temp = False
        if traj1_response2.curve_array.dtype != np.int64 and \
            traj1_response2.curve_array.dtype != np.float64:
            temp = False
        if traj2_response2.curve_array.dtype != np.int64 and \
            traj2_response2.curve_array.dtype != np.float64:
            temp = False
        self.assertTrue(temp)

        # same response for trajectories at crossing points
        cross_i1 = int(((self.traj1.curve_array.shape[1])/(2*np.pi))*(np.pi/2))
        cross_i2 = int(((self.traj1.curve_array.shape[1])/(2*np.pi))*((3*np.pi)/2))
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
        self.sys1.parameters['mu'] = 1
        self.sys2.parameters['mu'] = 1
        sys1_jac = self.traj1.jacob_init(self.sys1)
        sys2_jac = self.traj2.jacob_init(self.sys2)

        # outputs are numbers
        temp1 = True
        rindex1 = int(rand.random()*(np.shape(self.traj1.curve_array)[1]))
        rindex2 = int(rand.random()*(np.shape(self.traj2.curve_array)[1]))
        output1 = sys1_jac(rindex1)
        output2 = sys2_jac(rindex2)
        if output1.dtype != np.int64 and output1.dtype != np.float64:
            temp1 = False
        if output2.dtype != np.int64 and output2.dtype != np.float64:
            temp1 = False
        self.assertTrue(temp1)

        # output is correct size
        temp2 = True
        if output1.shape != (2, 2):
            temp2 = False
        if output2.shape != (2, 2):
            temp2 = False
        self.assertTrue(temp2)

        # correct values
        rstate1 = self.traj1.curve_array[:, rindex1]
        rstate2 = self.traj2.curve_array[:, rindex2]
        sys1_jac_true = vpd.jacobian(rstate1)
        sys2_jac_true = vis.jacobian(rstate2)
        self.assertTrue(np.allclose(output1, sys1_jac_true))
        self.assertTrue(np.allclose(output2, sys2_jac_true))

    def test_normed_traj(self):
        traj1_normed = self.traj1.normed_traj()
        traj2_normed = self.traj2.normed_traj()

        # output is of the Trajectory class
        self.assertIsInstance(traj1_normed, Trajectory)
        self.assertIsInstance(traj2_normed, Trajectory)

        # single number at each index
        temp1 = True
        for i in range(traj1_normed.curve_array.shape[1]):
            if traj1_normed.curve_array[:, i].shape[0] != 1:
                temp1 = False
        for i in range(traj2_normed.curve_array.shape[1]):
            if traj2_normed.curve_array[:, i].shape[0] != 1:
                temp1 = False
        self.assertTrue(temp1)
        
        # outputs are numbers
        temp2 = True
        if traj1_normed.curve_array.dtype != np.int64 and \
            traj1_normed.curve_array.dtype != np.float64:
            temp2 = False
        if traj2_normed.curve_array.dtype != np.int64 and \
            traj2_normed.curve_array.dtype != np.float64:
            temp2 = False
        self.assertTrue(temp2)

        # correct values for known cases
        traj1_norm_true = Trajectory(np.ones(traj1_normed.curve_array.shape))
        traj2_norm_true = np.zeros(traj2_normed.curve_array.shape)
        for i in range(np.shape(traj2_norm_true)[1]):
            traj2_norm_true[:, i] = np.sqrt((self.traj2.curve_array[0, i]**2) \
                + (self.traj2.curve_array[1, i]**2))
        traj2_norm_true = Trajectory(traj2_norm_true)
        self.assertEqual(traj1_normed, traj1_norm_true)
        self.assertEqual(traj2_normed, traj2_norm_true)

    def test_average_over_s(self):
        traj1_av = self.traj1.average_over_s()
        traj2_av = self.traj2.average_over_s()
        traj3_av = self.traj3.average_over_s()

        # outputs are numbers
        self.assertEqual(traj1_av.shape, (2,))
        self.assertEqual(traj2_av.shape, (2,))
        self.assertEqual(traj3_av.shape, (2,))

        # zero average for circle and ellipse
        self.assertTrue(np.allclose(traj1_av, np.zeros(traj1_av.shape)))
        self.assertTrue(np.allclose(traj2_av, np.zeros(traj2_av.shape)))

        # non-zero average for offset circle
        self.assertTrue(np.allclose(traj3_av, np.ones(traj3_av.shape)))

    def test_mul(self):
        pass

if __name__ == "__main__":
    unittest.main()
