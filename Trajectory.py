# This file contains the class definition for a general trajectory in some
# vector space. This will most commonly be a periodic state-space trajectory.

import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from System import System
from my_fft import my_fft, my_ifft

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

    __slots__ = ['modes', 'shape']
    __array_priority__ = 1e16

    def __init__(self, curve, modes = 33):
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
            if len(np.shape(curve)) == 1:
                curve = np.expand_dims(curve, axis = 0)
            if len(np.shape(curve)) == 2:
                self.modes = curve
                self.shape = np.shape(curve)
            else:
                raise AttributeError("The mode array has to 2D (only rows and columns)!")
        elif hasattr(curve, '__call__'):    
            self.modes = self.func2modes(curve, modes)
            self.shape = np.shape(self.modes)
        else:
            raise TypeError("Curve variable has to be either a function or a \
            2D numpy array!")

    def func2modes(self, curve_func, modes):
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
        disc = 2*(modes - 1)
        curve_array = np.zeros([np.shape(curve_func(0))[0], disc])
        t = np.linspace(0, 2*np.pi*(1 - 1/disc), disc)
        for i in range(disc):
            curve_array[:, i] = curve_func(t[i])
        return my_fft(curve_array)

    def __add__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.modes + other_traj.modes)

    def __sub__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.modes - other_traj.modes)

    def __mul__(self, factor):
        # scalar multiplication
        if type(factor) == float or type(factor) == int or \
            type(factor) == np.float64 or type(factor) == np.int64:
            return Trajectory(factor*self.modes)
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmul__(self, factor):
        return self.__mul__(factor)
    
    def __matmul__(self, factor):
        if type(factor) == np.ndarray:
            return Trajectory(np.matmul(factor, self.modes))
        elif hasattr(factor, '__call__'):
            curve = my_ifft(self.modes)
            for i in range(np.shape(curve)[1]):
                curve[:, i] = np.matmul(factor(i), curve[:, i])
            return Trajectory(my_fft(curve))
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmatmul__(self, factor):
        return self.__matmul__(factor)

    def __pow__(self, exponent):
        # perform element-by-element exponentiation
        curve = my_ifft(self.modes)
        return Trajectory(my_fft(curve**exponent))

    def __eq__(self, other_traj, rtol = 1e-6, atol = 1e-6):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return np.allclose(self.modes, other_traj.modes, rtol = rtol, atol = atol)

    def __getitem__(self, key):
        i, j = key
        return self.modes[i, j]

    def __setitem__(self, key, value):
        i, j = key
        self.modes[i, j] = value

    def plot(self, **kwargs):
        """
            This function is a placeholder and will be used for plotting
            purposes.
        """
        import trajectory_functions as traj_funcs

        # unpack keyword arguments
        gradient = kwargs.get('gradient', None)
        title = kwargs.get('title', None)
        time_disc = kwargs.get('disc', None)
        mean = kwargs.get('mean', None)
        show = kwargs.get('show', True)
        proj = kwargs.get('proj', None)
        aspect = kwargs.get('aspect', None)

        # calcualte gradient
        grad = traj_funcs.traj_grad(self)

        # pad with zeros to increase resolution
        if time_disc != None:
            tot_modes = int(time_disc/2) + 1
            pad_len = tot_modes - self.shape[1]
            if pad_len >= 0:
                modes_padded = np.pad(self.modes, ((0, 0), (0, pad_len)), 'constant')
                grad_padded = np.pad(grad.modes, ((0, 0), (0, pad_len)), 'constant')
            else:
                modes_padded = self.modes[:, 0:(tot_modes + 1)]
                grad_padded = grad.modes[:, 0:(tot_modes + 1)]
        else:
            modes_padded = self.modes
            grad_padded = grad.modes

        # adding in mean
        if type(mean) == np.ndarray:
            modes_padded[:, 0] = mean

        # convert to time domain
        curve = my_ifft(modes_padded)
        grad = my_ifft(grad_padded)

        if self.shape[0] == 2:
            # plotting trajectory
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(np.append(curve[0], curve[0, 0]), np.append(curve[1], curve[1, 0]))
            if aspect != None:
                ax.set_aspect(aspect)

            # add gradient
            if gradient != None:
                for i in range(0, curve.shape[1], int(1/gradient)):
                    ax.quiver(curve[0, i], curve[1, i], grad[0, i], grad[1, i])

            # plt.xlabel("$x$")
            # plt.ylabel("$\dot{x}$")
            # plt.xlim([-2.2, 2.2])
            # plt.ylim([-4, 4])
            # plt.grid()

        elif self.shape[0] == 3:
            # plotting trajectory
            if proj == None:
                fig = plt.figure()
                ax = fig.gca(projection = "3d")
                ax.plot(np.append(curve[0], curve[0, 0]), np.append(curve[1], curve[1, 0]), np.append(curve[2], curve[2, 0]))
                ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                fig.suptitle(title)
            elif proj == 'xy' or proj == 'yx':
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(np.append(curve[0], curve[0, 0]), np.append(curve[1], curve[1, 0]))
                if aspect != None:
                    ax.set_aspect(aspect)
            elif proj == 'xz' or proj == 'zx':
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(np.append(curve[0], curve[0, 0]), np.append(curve[2], curve[2, 0]))
                if aspect != None:
                    ax.set_aspect(aspect)
            elif proj == 'yz' or proj == 'zy':
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(np.append(curve[1], curve[1, 0]), np.append(curve[2], curve[2, 0]))
                if aspect != None:
                    ax.set_aspect(aspect)
        else:
            raise ValueError("Can't plot!")

        if show == True:
            plt.show()

if __name__ == '__main__':
    from test_cases import unit_circle as circ
    from test_cases import ellipse as elps

    unit_circle1 = Trajectory(circ.x)
    unit_circle2 = 0.5*Trajectory(circ.x)

    unit_circle3 = np.pi*unit_circle1 + unit_circle2

    unit_circle1.plot(gradient = 1/4, aspect = 1)
    unit_circle3.plot(gradient = 1, time_disc = 16, aspect = 1)
    
    ellipse = Trajectory(elps.x)
    ellipse.plot(gradient = 1/4, mean = np.ones(2), aspect = 1)
