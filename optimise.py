# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

# Thomas Burton - November 2020

import numpy as np
from Trajectory import Trajectory
from System import System

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
