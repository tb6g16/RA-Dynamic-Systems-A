# This file contains the function to calculate the global residual for a given dynamical system
# and an arbitrary solution function.

# Thomas Burton - October 2020

import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt

def tangent_vector(curve, s):
    """
        This function estimates the tangent vector to a curve in state space using
        finite differencing.

        Parameters
        ----------
        curve: function
            defines the curve in state space, taking in a float
            and returnin a vector
        s: float
            location at which the tangent vector needs to estimated
        
        Returns
        -------
        tangent: vector
            the tangent vector of the state space curve at the
            given location on the curve, s
    """

    # define step size
    step_size = 1e-6

    # estimate the tangent vector using central differencing and return
    return (curve(s + step_size) - curve(s - step_size))/step_size

def init_global_residual_time(dynamics):
    """
        This function initialises the global residual function in the time domain for a given
        dynamical system.

        Parameters
        ----------
        dynamics: function
            the function that defines the response of the system
            at each state
        
        Returns
        -------
        global_residual_time: function
            instance of the function that calculates the global
            residual for a given solution curve and frequency
    """
    def global_residual_time(solution_curve, fundamental_frequency):
        """
            This function calculates the global residual of a solution curve with a given
            frequency. This function is initialised with a specific dynamics specified.

            Parameters
            ----------
            solution_curve: function
                the proposed curve in state space that solves the
                dynamics, assumed periodic
            fundamental_frequency: float
                frequency of the periodic solution curve
        """

        s = np.linspace(0, 2*np.pi, 500)

        # initialise vectors
        local_residual_norm = np.zeros(np.shape(s)[0])

        # calculate local residual norms along solution curve
        for i in range(np.shape(s)[0]):
            local_residual_norm[i] = np.linalg.norm((fundamental_frequency*tangent_vector(solution_curve, s[i])) - dynamics(solution_curve(s[i])), 2)

        # integrate the residual norm to get the global residual and return
        return (1/(4*np.pi))*integ.simps(local_residual_norm, s)
    return global_residual_time

# run file to test residual function
if __name__ == "__main__":

    from test_case import solution_curve, dynamical_system
    fundamental_frequency = np.logspace(-6, 0, 100)

    global_residual_time = init_global_residual_time(dynamical_system)
    global_residual = np.zeros(np.shape(fundamental_frequency)[0])
    for i in range(np.shape(fundamental_frequency)[0]):
        global_residual[i] = global_residual_time(solution_curve, fundamental_frequency[i])

    plt.semilogx(fundamental_frequency, global_residual)
    plt.xlabel(r"$\omega_0$"), plt.ylabel("Global Residual")
    plt.grid()
    plt.show()

    print(fundamental_frequency[np.argmin(global_residual)])
    print(np.min(global_residual))