# This file will hold the functions that define dynamical system and solution curve
# for testing purposes

import numpy as np

# define a test solution
def solution_curve(s):
    """
        Solution function to test the operation of the global_residual_time
        function, defined over the range of 0 <= s >= 2*pi.

        Paramters
        ---------
        s: float
            parametric distance along the solution function
        
        Returns
        -------
        state: array
            the state of the system at the parametric
            distance along the curve, s
    """

    # initialise vectors
    state = np.zeros([2])

    # define function behaviour
    state[0] = np.cos(s)
    state[1] = -np.sin(s)

    return state

# define test dynamical system
def dynamical_system(state):
    """
        Dynamical system function to test the operation of the global_residual_time
        function, giving the rate of change of the system at each location in the
        given state-space.

        Parameters
        ----------
        state: vector
            the state of the system, given as a vector of the
            same length as dimensions of the state-space
        
        Returns
        -------
        tangent: vector
            the direction of change at a given state, called the
            tangent because it is a vector tangent to curve
            in the state-space at that state
    """

    # define constants
    frequency = 1

    # initialise vectors
    tangent = np.zeros(np.shape(state))

    # calculate tangent vector
    tangent[0] = state[1]
    tangent[1] = -state[0]*(frequency**2)

    return tangent