# This file contains the class definition for a general trajectory in some
# vector space. This will most commonly be a periodic state-space trajectory.

# Thomas Burton - October 2020

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Trajectory:
    """
        A trajectory in some finite-dimensional vector space parameterised with
        respect to a standard 'time' unit assumed to range from 0 to 2*pi.

        Attributes
        ----------
        curve_array: numpy.ndarray
            the discretised trajectory, number of rows equal to the dimension
            of the vector space and columns equal to the number of 'time' steps
            taken
        curve_fun: function
            function defining the trajectory, given as an input to __init__ to
            generate curve_array attribute
        closed: bool
            prescribing whether the trajectory is closed, i.e. is it periodic
            in 'time'
        
        Methods
        -------
        func2array(curve_func, time_disc = 200)
        plot()
    """
    
    __slots__ = ['curve_array', 'curve_func', 'closed']

    def __init__(self, curve, closed = True):
        """
            DOCSTRING NEEDED HERE
        """
        if type(curve) == np.ndarray:
            if len(np.shape(curve)) == 2:
                self.curve_array = curve
                self.closed = closed
                self.curve_func = None
            else:
                raise AttributeError("The trajectory array has to 2D (only \
                rows and columns)!")
        elif hasattr(curve, '__call__'):    
            self.curve_array = self.func2array(curve)
            self.closed = closed
            self.curve_func = curve
        else:
            raise TypeError("Curve variable has to be either a function or a \
            2D numpy array!")
    
    def func2array(self, curve_func, time_disc = 200):
        """
            DOCSTRING NEEDED HERE
        """
        curve_array = np.zeros([np.shape(curve_func(0))[0], time_disc])
        t = np.linspace(0, 2*np.pi, time_disc)
        for i in range(time_disc):
                curve_array[:, i] = curve_func(t[i])
        return curve_array
    
    def plot(self):
        """
            DOCSTRING NEEDED HERE
        """
        # check if vector space is 2D or 3D
        if np.shape(self.curve_array)[0] == 1:
            t = np.linspace(0, 2*np.pi, np.shape(self.curve_array)[1])
            # plot state against parametric time
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(t, self.curve_array[0])
            plt.show()
        elif np.shape(self.curve_array)[0] == 2:
            # plot in 2D vector space
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.curve_array[0], self.curve_array[1])
            ax.set_aspect('equal')
            plt.show()
        elif np.shape(self.curve_array)[0] == 3:
            # plot in 3D vector space
            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            ax.plot(self.curve_array[0], self.curve_array[1], \
            self.curve_array[2])
            plt.show()
        else:
            raise ValueError("Cannot plot trajectories in higher dimensions!")
        return None

if __name__ == '__main__':
    from test_cases import harmonic_oscillator as harm

    unit_circle = Trajectory(harm.solution_curve)
    unit_circle.plot()
