# This file contains the function definitions for some trajectory-trajectory
# and trajectory-system interactions.

import numpy as np
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System
from my_fft import my_fft, my_ifft

# def swap_tf(object):
#     if hasattr(object, 'modes'):
#         return np.fft.irfft(object.modes, axis = 1)
#     elif type(object) == np.ndarray:
#         return np.fft.rfft(object, axis = 1)
#     else:
#         raise TypeError("Input is not of correct type!")

def traj_grad(traj):
    """
        This function calculates the gradient vectors of a given trajectory and
        returns it as an instance of the Trajectory class. The algorithm used
        is based off of the RFFT

        Parameters
        ----------
        traj: Trajectory object
            the trajectory which we will calculate the gradient for
        
        Returns
        -------
        grad: Trajectory object
            the gradient of the input trajectory
    """
    # initialise array for new modes
    new_modes = np.zeros(traj.shape, dtype = np.complex)

    # loop over time and multiply modes by modifiers
    for k in range(traj.shape[1]):
        new_modes[:, k] = 1j*k*traj.modes[:, k]
    
    # force zero mode if symmetric
    if traj.shape[1] % 2 == 0:
        new_modes[:, traj.shape[1]//2] = 0

    return Trajectory(new_modes)

def traj_inner_prod(traj1, traj2):
    """
        This function calculates the Euclidean inner product of two
        trajectories at each location along their domains, s.

        Parameters
        ----------
        traj1: Trajectory object
            the first trajectory in the inner product
        traj2: Trajectory object
            the second trajectory in the inner product
        
        Returns
        -------
        traj_prod: Trajectory object
            the inner product of the two trajectories at each location along
            their domains, s
    """
    # convert to time domain
    # curve1 = swap_tf(traj1)
    # curve2 = swap_tf(traj2)
    curve1 = my_ifft(traj1.modes)
    curve2 = my_ifft(traj2.modes)

    # initialise new array
    prod_curve = np.zeros([1, np.shape(curve1)[1]])

    # evaluate inner product in time domain
    for i in range(np.shape(curve1)[1]):
        prod_curve[:, i] = np.dot(curve1[:, i], curve2[:, i])

    # convert back to frequency domain and return
    # return Trajectory(swap_tf(prod_curve))
    return Trajectory(my_fft(prod_curve))

def traj_response(traj, func):
    """
        This function evaluates the response over the domain of a given
        trajectory due to a given dyanmical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory over which the response will be evaluated
        func: function
            the function that defines the response at each location along the
            trajectory, must be a function that inputs a vector and outputs a
            vector of the same size
        
        Returns
        -------
        response_traj: Trajectory object
            the response at each location of the trajectory, given as an
            instance of the Trajectory class
    """
    # convert trajectory to time domain
    # curve = swap_tf(traj)
    curve = my_ifft(traj.modes)

    # evaluate response in time domain
    for i in range(np.shape(curve)[1]):
        curve[:, i] = func(curve[:, i])

    # convert back to frequency domain and return
    # return Trajectory(swap_tf(curve))
    return Trajectory(my_fft(curve))

def jacob_init(traj, sys, if_transp = False):
    """
        This function initialises a jacobian function that returns the jacobian
        matrix for the given system at each location along the domain of the
        trajectory (through the indexing of the array, i).

        Parameters
        ----------
        traj: Trajectory object
            the trajectory over which the jacobian matrix will be evaluated
        sys: System object
            the system for which the jacobian matrix is for
        
        Returns
        -------
        jacob: function
            the function that returns the jacobian matrix for a given index of
            the array defining the trajectory in state-space
    """
    def jacobian(i):
        """
            This function returns the jacobian matrix for a dynamical system at
            a specified location along a trajectory through the associated
            state-space, indexed by i.

            Parameters
            ----------
            i: positive integer
                the discretised location along the trajectory in state-space
            
            Returns
            -------
            jacobian_matrix: numpy array
                the 2D numpy array for the jacobian of a dynamical system given
                at a specified location of the trajectory
        """
        # convert to time domain
        # curve = swap_tf(traj)
        curve = my_ifft(traj.modes)
        
        # test for input
        if i%1 != 0:
            raise TypeError("Inputs are not of the correct type!")
        if i >= np.shape(curve)[1]:
            raise ValueError("Input index is too large!")

        # make sure index is integer
        i = int(i)

        # state at index
        state = curve[:, i]

        # the jocobian for that state
        if if_transp == True:
            return np.transpose(sys.jacobian(state))
        elif if_transp == False:
            return sys.jacobian(state)
    return jacobian
