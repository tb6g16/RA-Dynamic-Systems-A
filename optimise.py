# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

# Thomas Burton - November 2020

import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System
from Problem import Problem
from traj2vec import traj2vec, vec2traj

def init_opt_funcs(sys, dim):
    """
        This functions initialises the optimisation vectors for a specific
        system.
    """
    def traj_global_res(opt_vector):
        """
            This function calculates the global residual for a given vector
            that defines a trajectory frequency pair, for the purpose of
            optimisation.
        """
        # unpack vector
        traj, freq = vec2traj(opt_vector, dim)

        # initialise problem class
        current = Problem(traj, sys, freq)

        return current.global_residual[0]

    def traj_global_res_jac(opt_vector):
        """
            This function calculates the gradient of the global residual for a
            given vector that defines a trajectory frequency pair, for the
            purpose of optimisation.
        """
        # unpack vector
        traj, freq = vec2traj(opt_vector, dim)

        # initialise problem class
        current = Problem(traj, sys, freq)

        # calculate gradient values
        global_res_wrt_traj = current.dglobal_res_dtraj()
        global_res_wrt_freq = current.dglobal_res_dfreq()

        return traj2vec(global_res_wrt_traj, global_res_wrt_freq)
    return traj_global_res, traj_global_res_jac

def init_constraints(sys, dim, mean):
    """
        This function intialises the nonlinear constraints imposed on the
        optimisation, and returns the instance of the NonlinearConstraint class
        that is passed as an argument to the optimisation.
    """
    def constraints(opt_vector):
        """
            This function is the input to the NonlinearConstraint class to
            intialise it.
        """
        traj, _ = vec2traj(opt_vector, dim)
        traj_fluc = Trajectory(traj.curve_array - mean)
        mean_fluc_const = traj_fluc.average_over_s()
        traj_fluc_nl = traj_fluc.traj_nl_response(sys)
        nl_fluc_const = np.squeeze(sys.response(mean)) + traj_fluc_nl.average_over_s()
        return np.concatenate((mean_fluc_const, nl_fluc_const), axis = 0)
    eq = np.zeros([2*dim])
    return opt.NonlinearConstraint(constraints, eq, eq)

if __name__ == "__main__":
    from test_cases import unit_circle as uc
    from test_cases import van_der_pol as vpd

    sys = System(vpd)
    sys.parameters['mu'] = 1
    circle = Trajectory(uc.x)
    freq = 1
    dim = 2

    res_func, jac_func = init_opt_funcs(sys, dim)

    constraint = init_constraints(sys, dim, np.zeros([2, 1]))

    # op_vec = opt.minimize(res_func, traj2vec(circle, freq), options = {'maxiter': 1})
    op_vec = opt.minimize(res_func, traj2vec(circle, freq), jac = jac_func) # , method = 'L-BFGS-B')
    # op_vec = opt.minimize(res_func, traj2vec(circle, freq), jac = jac_func,\
    #     constraints = constraint)
    print(op_vec.message)
    print("Number of iterations: " + str(op_vec.nit))
    op_traj, op_freq = vec2traj(op_vec.x, dim)

    op_traj.plot(gradient = True, gradient_density = 32/256)
    traj_diff = op_traj - circle
    traj_diff.plot(gradient = True, gradient_density = 32/256)

    print(res_func(traj2vec(circle, freq)))
    print(res_func(traj2vec(op_traj, op_freq)))

    # test jacbian is zero also
    # print(jac_func(traj2vec(circle, freq)))
    # print(jac_func(traj2vec(op_traj, op_freq)))
