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
