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
        gradient()
        plot()
    """
    
    __slots__ = ['curve_array', 'curve_func', 'grad']

    def __init__(self, curve):
        """
            Initialise an instance of the Trajectory object, with either a
            continuous of discrete time function.

            Parameters
            ----------
            curve: function or numpy.ndarray
                function defining the trajectory, either given by a python
                function (continuous) or a numpy array (discrete)
        """
        if type(curve) == np.ndarray:
            if len(np.shape(curve)) == 2:
                self.curve_array = curve
                self.curve_func = None
                self.grad = None
            else:
                raise AttributeError("The trajectory array has to 2D (only \
                rows and columns)!")
        elif hasattr(curve, '__call__'):    
            self.curve_array = self.func2array(curve)
            self.curve_func = curve
            self.grad = None
        else:
            raise TypeError("Curve variable has to be either a function or a \
            2D numpy array!")
    
    # TIME_DISC HAS TO BE EVEN BECAUSE OF RFFT ALGORITHM
    def func2array(self, curve_func, time_disc = 256):
        """
            Discretise a continuous time representation of a function (given
            as a python function) to a discrete time representation (as a
            numpy array).

            Parameters
            ----------
            curve_func: function
                python function that defines the continuous time representation
                of the trajectory
            time_disc: positive integer
                number of discrete time locations to use
        """
        curve_array = np.zeros([np.shape(curve_func(0))[0], time_disc])
        t = np.linspace(0, 2*np.pi - (2*np.pi/(time_disc - 1)), time_disc)
        for i in range(time_disc):
                curve_array[:, i] = curve_func(t[i])
        return curve_array
    
    def add_traj(self, other_traj):
        """
            Sum two trajectories together, assuming they have the same
            discretisation.

            Parameters
            ----------
            other_traj: Trajectory
                the other trajectory to be summed to this instace
            
            Returns
            -------
            new_traj: Trajectory
                a new Trajectory instance which is the sum of the two previous
        """
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.curve_array + other_traj.curve_array)

    def sub_traj(self, other_traj):
        """
            Take the difference of two trajectories, assuming they have the
            same discretisation.

            Parameters
            ----------
            other_traj: Trajectory
                the other trajectory to be differenced with this instance

            Returns
            -------
            new_traj: Trajectory
                a new Trajectory instance which is the difference of the two
                previous
        """
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.curve_array + other_traj.curve_array)

    def gradient(self):
        """
            Calculate the gradient of the trajectory (tangent vector) at the
            time locations given by the discrete time representation of the
            Trajectory. The method used is Spectral Differentiation.
        """
        # number of discretised time locations
        time_disc = np.shape(self.curve_array)[1]
        # FFT along the time dimension
        mode_array = np.fft.rfft(self.curve_array, axis = 1)
        # loop over time and multiply modes by modifiers
        for k in range(time_disc//2):
            mode_array[:, k] *= 1j*k
        # force zero mode if symmetric
        if time_disc % 2 == 0:
            mode_array[:, time_disc//2] = 0
        # IFFT to get discrete time gradients
        self.grad = Trajectory(np.fft.irfft(mode_array, axis = 1))

    def plot(self, gradient = False, gradient_density = None):
        """
            Plot 1D, 2D, or 3D trajectories or gradients.

            Paramters
            ---------
            gradient: bool
                boolean to decide whether to plot the gradient (tangent
                vectors) with the trajectory curve
            gradient_density: float between 0 and 1
                amount of gradient vectors to show on the trajectory curve
        """
        # check if gradient density is between o and 1
        if gradient_density < 0 or gradient_density > 1:
            raise ValueError("gradient_density should be between 0 and 1 \
            inclusive!")
        # check if gradient attribute has value None
        if gradient == True and self.grad is None:
            self.gradient()
        # check dimension of plotting space and then plot (if possible)
        if np.shape(self.curve_array)[0] == 1:
            t = np.linspace(0, 2*np.pi, np.shape(self.curve_array)[1])
            # plot state against parametric time
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(t, self.curve_array[0])

            # NEEDS TO IMPLEMENT HERE ON X-AXIS

            plt.show()
        elif np.shape(self.curve_array)[0] == 2:
            # plot in 2D vector space
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(np.append(self.curve_array[0], self.curve_array[0, 0]), \
                np.append(self.curve_array[1], self.curve_array[1, 0]))
            ax.set_aspect('equal')
            if gradient:
                for i in range(0, np.shape(self.curve_array)[1], \
                int(1/gradient_density)):
                    ax.quiver(self.curve_array[0, i], self.curve_array[1, i], \
                    self.grad.curve_array[0, i], self.grad.curve_array[1, i])
            plt.show()
        elif np.shape(self.curve_array)[0] == 3:
            # plot in 3D vector space
            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            ax.plot(np.append(self.curve_array[0], self.curve_array[0, 0]), \
                np.append(self.curve_array[1], self.curve_array[1, 0]), \
                np.append(self.curve_array[2], self.curve_array[2, 0]))
            # NEED TO DO GRADIENT PLOT FOR 3D
            plt.show()
        else:
            raise ValueError("Cannot plot trajectories in higher dimensions!")
        return None


if __name__ == '__main__':
    from test_cases import unit_circle as circ

    unit_circle = Trajectory(circ.x)
    unit_circle.plot(gradient = True, gradient_density = 32/256)
    
