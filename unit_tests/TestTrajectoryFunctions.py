# This file contains the testing methods for the functions defined in
# trajectory_functions

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

class TestTrajectoryFunctions(unittest.TestCase):

    def setUp(self):
        a = 1
        self.traj1 = Trajectory(uc.x, disc = 64)
        self.traj1_grad = traj_funcs.traj_grad(self.traj1)
        self.traj2 = Trajectory(elps.x, disc = 64)
        self.traj2_grad = traj_funcs.traj_grad(self.traj2)
        self.traj3 = Trajectory(self.traj1.curve_array + a)
        self.traj3_grad = traj_funcs.traj_grad(self.traj3)
        self.sys1 = System(vpd)
        self.sys2 = System(vis)

    def tearDown(self):
        del self.traj1
        del self.traj2
        del self.traj3
        del self.sys1
        del self.sys2

    def test_traj_prod(self):
        traj1_traj1_prod = traj_funcs.traj_inner_prod(self.traj1, self.traj1)
        traj2_traj2_prod = traj_funcs.traj_inner_prod(self.traj2, self.traj2)
        traj1_traj2_prod = traj_funcs.traj_inner_prod(self.traj1, self.traj2)
        traj2_traj1_prod = traj_funcs.traj_inner_prod(self.traj2, self.traj1)

        # output is of the Trajectory class
        self.assertIsInstance(traj1_traj2_prod, Trajectory)
        self.assertIsInstance(traj2_traj1_prod, Trajectory)

        # does the operation commute
        self.assertEqual(traj1_traj2_prod, traj2_traj1_prod)

        # inner product equal to norm
        traj1_norm = np.ones([1, np.shape(self.traj1.curve_array)[1]])
        traj1_norm = Trajectory(traj1_norm)
        traj2_norm = np.zeros([1, np.shape(self.traj2.curve_array)[1]])
        for i in range(np.shape(self.traj2.curve_array)[1]):
            s = ((2*np.pi)/np.shape(self.traj2.curve_array)[1])*i
            traj2_norm[0, i] = (4*(np.cos(s)**2)) + (np.sin(s)**2)
        traj2_norm = Trajectory(traj2_norm)
        self.assertEqual(traj1_norm, traj1_traj1_prod)
        self.assertEqual(traj2_norm, traj2_traj2_prod)

        # single number at each index
        temp1 = True
        for i in range(np.shape(traj1_traj2_prod.curve_array)[1]):
            if traj1_traj2_prod.curve_array[:, i].shape[0] != 1:
                temp1 = False
        for i in range(traj2_traj1_prod.curve_array.shape[1]):
            if traj2_traj1_prod.curve_array[:, i].shape[0] != 1:
                temp1 = False
        self.assertTrue(temp1)

        # outputs are numbers
        temp2 = True
        if traj1_traj2_prod.curve_array.dtype != np.int64 and \
            traj1_traj2_prod.curve_array.dtype != np.float64:
            temp2 = False
        if traj2_traj1_prod.curve_array.dtype != np.int64 and \
            traj2_traj1_prod.curve_array.dtype != np.float64:
            temp2 = False
        self.assertTrue(temp2)

    def test_gradient(self):
        # same shape as original trajectories
        self.assertEqual(self.traj1.curve_array.shape, \
            self.traj1_grad.curve_array.shape)
        self.assertEqual(self.traj2.curve_array.shape, \
            self.traj2_grad.curve_array.shape)

        # outputs are real numbers
        temp = True
        if self.traj1_grad.curve_array.dtype != np.int64 and \
            self.traj1_grad.curve_array.dtype != np.float64:
            temp = False
        if self.traj2_grad.curve_array.dtype != np.int64 and \
            self.traj2_grad.curve_array.dtype != np.float64:
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
        self.assertEqual(traj1_grad, self.traj1_grad)
        self.assertEqual(traj2_grad, self.traj2_grad)

    def test_traj_response(self):
        # response to full system
        traj1_response1 = traj_funcs.traj_response(self.traj1, \
            self.sys1.response)
        traj1_response2 = traj_funcs.traj_response(self.traj1, \
            self.sys2.response)
        traj2_response1 = traj_funcs.traj_response(self.traj2, \
            self.sys1.response)
        traj2_response2 = traj_funcs.traj_response(self.traj2, \
            self.sys2.response)
        traj1_nl1 = traj_funcs.traj_response(self.traj1, \
            self.sys1.nl_factor)
        traj1_nl2 = traj_funcs.traj_response(self.traj1, \
            self.sys2.nl_factor)
        traj2_nl1 = traj_funcs.traj_response(self.traj2, \
            self.sys1.nl_factor)
        traj2_nl2 = traj_funcs.traj_response(self.traj2, \
            self.sys2.nl_factor)
        
        # output is of the Trajectory class
        self.assertIsInstance(traj1_response1, Trajectory)
        self.assertIsInstance(traj1_response2, Trajectory)
        self.assertIsInstance(traj2_response1, Trajectory)
        self.assertIsInstance(traj2_response2, Trajectory)
        self.assertIsInstance(traj1_nl1, Trajectory)
        self.assertIsInstance(traj1_nl2, Trajectory)
        self.assertIsInstance(traj2_nl1, Trajectory)
        self.assertIsInstance(traj2_nl2, Trajectory)

        # outputs are numbers
        temp1 = True
        if traj1_response1.curve_array.dtype != np.int64 and \
            traj1_response1.curve_array.dtype != np.float64:
            temp1 = False
        if traj1_response2.curve_array.dtype != np.int64 and \
            traj1_response2.curve_array.dtype != np.float64:
            temp1 = False
        if traj2_response1.curve_array.dtype != np.int64 and \
            traj2_response1.curve_array.dtype != np.float64:
            temp1 = False
        if traj2_response2.curve_array.dtype != np.int64 and \
            traj2_response2.curve_array.dtype != np.float64:
            temp1 = False
        temp2 = True
        if traj1_nl1.curve_array.dtype != np.int64 and \
            traj1_nl1.curve_array.dtype != np.float64:
            temp2 = False
        if traj1_nl2.curve_array.dtype != np.int64 and \
            traj1_nl2.curve_array.dtype != np.float64:
            temp2 = False
        if traj2_nl1.curve_array.dtype != np.int64 and \
            traj2_nl1.curve_array.dtype != np.float64:
            temp2 = False
        if traj2_nl2.curve_array.dtype != np.int64 and \
            traj2_nl2.curve_array.dtype != np.float64:
            temp2 = False
        self.assertTrue(temp1)
        self.assertTrue(temp2)

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
        traj1_cross1_nl1 = traj1_nl1.curve_array[:, cross_i1]
        traj2_cross1_nl1 = traj2_nl1.curve_array[:, cross_i1]
        traj1_cross2_nl1 = traj1_nl1.curve_array[:, cross_i2]
        traj2_cross2_nl1 = traj2_nl1.curve_array[:, cross_i2]
        traj1_cross1_nl2 = traj1_nl2.curve_array[:, cross_i1]
        traj2_cross1_nl2 = traj2_nl2.curve_array[:, cross_i1]
        traj1_cross2_nl2 = traj1_nl2.curve_array[:, cross_i2]
        traj2_cross2_nl2 = traj2_nl2.curve_array[:, cross_i2]
        self.assertTrue(np.allclose(traj1_cross1_nl1, traj2_cross1_nl1))
        self.assertTrue(np.allclose(traj1_cross2_nl1, traj2_cross2_nl1))
        self.assertTrue(np.allclose(traj1_cross1_nl2, traj2_cross1_nl2))
        self.assertTrue(np.allclose(traj1_cross2_nl2, traj2_cross2_nl2))

    def test_jacob_init(self):
        self.sys1.parameters['mu'] = 1
        self.sys2.parameters['mu'] = 1
        sys1_jac = traj_funcs.jacob_init(self.traj1, self.sys1)
        sys2_jac = traj_funcs.jacob_init(self.traj2, self.sys2)

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

    def test_average_over_s(self):
        traj1_av = traj_funcs.average_over_s(self.traj1)
        traj2_av = traj_funcs.average_over_s(self.traj2)
        traj3_av = traj_funcs.average_over_s(self.traj3)

        # outputs are numbers
        self.assertEqual(traj1_av.shape, (2,))
        self.assertEqual(traj2_av.shape, (2,))
        self.assertEqual(traj3_av.shape, (2,))

        # zero average for circle and ellipse
        self.assertTrue(np.allclose(traj1_av, np.zeros(traj1_av.shape)))
        self.assertTrue(np.allclose(traj2_av, np.zeros(traj2_av.shape)))

        # non-zero average for offset circle
        self.assertTrue(np.allclose(traj3_av, np.ones(traj3_av.shape)))


if __name__ == "__main__":
    unittest.main()