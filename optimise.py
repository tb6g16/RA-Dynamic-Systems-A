# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

# Thomas Burton - November 2020

import numpy as np
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

def init_optimise_time(sys):
    """
        This function initialises an optimisation function (in time domain) for
        a specific dynamical system.

        Parameters
        ----------
        sys: System
            the dynamical system to optimise with respect to
        
        Returns
        -------
        optimise_time: function
            the function that optimises an initial trajectory and frequency to
            minimise the global residual with respect to the given dynamical
            system
    """
    def optimise_time(init_traj, init_freq, iter_max = 100):
        """
            This function takes an initial trajectory and frequency and optimises
            them to minimise a global residual with respect to a given dynamical
            system.

            Parameters
            ----------
            init_traj: Trajectory
                the initial trajectory from which to start optimising
            init_freq: float
                the initial frequency from which to start optimisaing
            iter_max: positive integer
                the maximum number of iterations of the optimisation to perform
                before forcibly terminating
            
            Returns
            -------
            opt_traj: Trajectory
                the optimal trajectory to produce a minimum in the global
                residual
            opt_freq: float
                the optimal frequency to produce a minimum in the global
                residual
        """
        return None
    return optimise_time

if __name__ == "__main__":
    from test_cases import unit_circle as uc
    from test_cases import van_der_pol as vpd

    sys = System(vpd)
    circle = Trajectory(uc.x)
    freq = 1

    res_func, jac_func = init_opt_funcs(sys, 2)

    print(res_func(traj2vec(circle, freq)))
