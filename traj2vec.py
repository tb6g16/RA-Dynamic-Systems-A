# This file contains the functions that convert a trajectory frequency pair to
# a vector for optimisation purposes, and also the inverse conversion.

# Thomas Burton - November 2020

import numpy as np
from Trajectory import Trajectory

def traj2vec(traj, freq):
    """
        This function takes in a trajectory and frequency and returns a vector
        that will be used for the optimisation.

        Parameters
        ----------
        traj: Trajectory
            the trajectory that makes up most of the optimisation vector
        freq: float
            the fundamental frequency of the associated trajectory, the last
            element of the optimisation vector
        
        Returns
        -------
        opt_vector: numpy array
            the optimisation vector defined by the trajectory frequency pair
    """
    array_size_x = np.shape(traj.curve_array)[0]
    array_size_y = np.shape(traj.curve_array)[1]
    dofs = (array_size_x*array_size_y) + 1
    vector = np.zeros([dofs])
    for j in range(array_size_y):
        for i in range(array_size_x):
            vector[i + j*array_size_x] = traj.curve_array[i, j]
    vector[-1] = freq
    return vector

def vec2traj(opt_vector, dim):
    """
        This function converts an optimisation variable back into its
        corresponding trajectory frequency pair.

        Parameters
        ----------
        opt_vector: numpy array
            the optimisation vector
        dim: positive integer
            the dimension of the state-space through which the trajectory
            travels
        
        Returns
        -------
        traj: Trajectory
            the corresponding trajectory
        freq: float
            the corresponding frequency
    """
    vec_size = np.shape(opt_vector)[0]
    if (vec_size - 1)/dim % 1 != 0:
        raise ValueError("Vector length not compatible with dim!")
    traj_array = np.zeros([dim, int((vec_size - 1)/dim)])
    for i in range(vec_size - 1):
        traj_array[i - dim*int(i/dim), int(i/dim)] = opt_vector[i]
    return Trajectory(traj_array), opt_vector[-1]

if __name__ == "__main__":
    from test_cases import unit_circle as uc

    traj1 = Trajectory(uc.x)
    vec = traj2vec(traj1, 1)
    traj2, freq = vec2traj(vec, 2)

    traj1.plot(gradient = True, gradient_density = 32/256)
    traj2.plot(gradient = True, gradient_density = 32/256)
    print(freq)
