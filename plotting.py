# This file contains the function to plot state-space solution curves with a given
# dynamical system in 2D or 3D.

# Thomas Burton - October 2020

import numpy as np
import matplotlib.pyplot as plt

def plot(dynamics, curve, dim = 2):
    """
        This function plots the given dynamical system and trajectory in the same space, as a vector
        field and as a curve respectively.

        Parameters
        ----------
        dynamics: function
            the dynamical system defining the system response at
            each state
        curve: function
            a given trajectory in state-space
        dim: {2, 3}
            the dimension of the state-space being plotted in, only
            permitted to be 2- or 3-dimensional
    """

    if dim not in (2, 3):
        raise ValueError("Dimension given not a valid option, choose from 2 or 3!")

    # discretise domain on trajectory
    s = np.linspace(0, 2*np.pi, 200)

    if dim == 2:
        # discretise domain of dynamical system
        X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
        
    else:
        s[0] = 1

    return None