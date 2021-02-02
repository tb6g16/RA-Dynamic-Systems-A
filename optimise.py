# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

import numpy as np
import scipy.optimize as opt
from Trajectory import Trajectory
from System import System
from traj2vec import traj2vec, vec2traj
import trajectory_functions as traj_funcs
import residual_functions as res_funcs

def init_opt_funcs(sys, dim, mean):
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
        # unpack trajectory
        traj, freq = vec2traj(opt_vector, dim)

        # calculate global residual and return
        return res_funcs.global_residual(traj, sys, freq, mean, with_zero = True)

    def traj_global_res_jac(opt_vector):
        """
            This function calculates the gradient of the global residual for a
            given vector that defines a trajectory frequency pair, for the
            purpose of optimisation.
        """
        # unpack trajectory
        traj, freq = vec2traj(opt_vector, dim)

        # calculate global residual gradients
        gr_traj, gr_freq = res_funcs.global_residual_grad(traj, sys, freq, mean)

        # convert back to vector and return
        return traj2vec(gr_traj, gr_freq)
    
    return traj_global_res, traj_global_res_jac

if __name__ == "__main__":
    from test_cases import unit_circle as uc
    from test_cases import van_der_pol as vpd
    from test_cases import viswanath as vis

    sys = System(vpd)
    sys.parameters['mu'] = 2
    # sys = System(vis)
    # sys.parameters['mu'] = 1
    circle = 2*Trajectory(uc.x, modes = 65)
    freq = 1
    dim = 2

    res_func, jac_func = init_opt_funcs(sys, dim, np.zeros(2))

    op_vec = opt.minimize(res_func, traj2vec(circle, freq), jac = jac_func, method = 'L-BFGS-B')
    # op_vec = opt.minimize(res_func, traj2vec(circle, freq), method = 'L-BFGS-B')

    print(op_vec.message)
    print("Number of iterations: " + str(op_vec.nit))
    op_traj, op_freq = vec2traj(op_vec.x, dim)

    print("Period of orbit: " + str((2*np.pi)/op_freq))
    print("Global residual before: " + str(res_func(traj2vec(circle, freq))))
    print("Global residual after: " + str(res_func(traj2vec(op_traj, op_freq))))
    op_traj.plot(gradient = 1/4)

    print("Trajectory zero mode: " + str(op_traj[:, 0]))
    print("Local residual zero mode: " + str(res_funcs.local_residual(op_traj, sys, op_freq, np.zeros(2))[:, 0]))
