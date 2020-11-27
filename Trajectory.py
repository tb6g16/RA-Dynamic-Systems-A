# This file contains the class definition for a general trajectory in some
# vector space. This will most commonly be a periodic state-space trajectory.

# Thomas Burton - October 2020

import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from System import System

# GRADIENT ATTRIBUTE REQUIRES GETTER AND SETTER METHODS OR JUST CALCULATE AT
# INIT

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

    def __init__(self, curve, comp_grad = True, disc = 64):
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
                self.curve_array = curve
                self.curve_func = None
                if comp_grad == True:
                    self.gradient()
                else:
                    self.grad = None
            else:
                raise AttributeError("The trajectory array has to 2D (only \
                rows and columns)!")
        elif hasattr(curve, '__call__'):    
            self.curve_array = self.func2array(curve, time_disc = disc)
            self.curve_func = curve
            if comp_grad == True:
                self.gradient()
            else:
                self.grad = None
        else:
            raise TypeError("Curve variable has to be either a function or a \
            2D numpy array!")

    # TIME_DISC HAS TO BE EVEN BECAUSE OF RFFT ALGORITHM
    def func2array(self, curve_func, time_disc = 64):
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
        t = np.linspace(0, 2*np.pi*(1 - 1/time_disc), time_disc)
        for i in range(time_disc):
                curve_array[:, i] = curve_func(t[i])
        return curve_array

    def __add__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.curve_array + other_traj.curve_array)

    def __sub__(self, other_traj):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return Trajectory(self.curve_array - other_traj.curve_array)

    def __mul__(self, factor):
        # scalar multiplication
        if type(factor) == float or type(factor) == int:
            return Trajectory(factor*self.curve_array)
        # variable matrix multiplication
        elif hasattr(factor, '__call__'):
            s_disc = np.shape(self.curve_array)
            new_traj = np.zeros(s_disc)
            for i in range(s_disc[1]):
                # s = self.gen_s(i)
                new_traj[:, i] = np.matmul(factor(i), self.curve_array[:, i])
            return Trajectory(new_traj)
        # constant matrix multiplication
        elif type(factor) == np.ndarray:
            return Trajectory(np.matmul(factor, self.curve_array))
        else:
            raise TypeError("Inputs are not of the correct type!")

    def __rmul__(self, factor):
        # scalar multiplication
        # if type(factor) == float or type(factor) == int:
        #     return Trajectory(factor*self.curve_array)
        # # variable matrix multiplication
        # elif hasattr(factor, '__call__'):
        #     s_disc = np.shape(self.curve_array)
        #     new_traj = np.zeros(s_disc)
        #     for i in range(s_disc[1]):
        #         # s = self.gen_s(i)
        #         new_traj[:, i] = np.matmul(factor(i), self.curve_array[:, i])
        #     return Trajectory(new_traj)
        # # constant matrix multiplication
        # elif type(factor) == np.ndarray:
        #     return Trajectory(np.matmul(factor, self.curve_array))
        # else:
        #     raise TypeError("Inputs are not of the correct type!")
        return self.__mul__(factor)

    def __pow__(self, exponent):
        # perform element-by-element exponentiation
        return Trajectory(self.curve_array ** exponent)

    def __eq__(self, other_traj, rtol = 1e-6):
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        return np.allclose(self.curve_array, other_traj.curve_array, \
            rtol = rtol)

    def traj_prod(self, other_traj):
        """
            This performs a multiplication of the current instance with another
            instance of the Trajectory class in a sense of an Euclidean inner
            product for every location s.
        """
        if not isinstance(other_traj, Trajectory):
            raise TypeError("Inputs are not of the correct type!")
        disc_size = np.shape(self.curve_array)[1]
        product_array = np.zeros([1, disc_size])
        for i in range(disc_size):
            product_array[0, i] = np.dot(self.curve_array[:, i],\
                other_traj.curve_array[:, i])
        return Trajectory(product_array)

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
        self.grad = Trajectory(np.fft.irfft(mode_array, axis = 1), \
            comp_grad = False)

    def traj_response(self, sys):
        """
            This method takes in a trajectory in the state-space defined by the
            dynamical system of the instance of the System class and returns an
            instance of the Trajectory class which is the response of the
            system at each state over the length of the tracectory

            Parameters
            ----------
            traj: Trajectory object
                the trajectory through state-space
            
            Returns
            -------
            response_traj: Trajectory object
                the response at each state over the length of the given
                trajectory (given as an instance of the Trajectory object)
        """
        # checks
        if not isinstance(sys, System):
            raise TypeError("Inputs are not of the correct type!")

        # initialise arrays
        array_size = np.shape(self.curve_array)
        response_array = np.zeros(array_size)

        # calculate gradient of trajectory
        if self.grad is None:
            self.gradient()
        
        # evaluate response
        for i in range(array_size[1]):
            response_array[:, i] = sys.response(self.curve_array[:, i])
        
        return Trajectory(response_array)

    def traj_nl_response(self, sys):
        """
            This method returns an instance of the Trajectory class for the
            response at each location over the original trajectory of the non-
            linear factor of the dynamical system.
        """
        # checks
        if not isinstance(sys, System):
            raise TypeError("Inputs are not of the correct type!")

        # initialise arrays
        array_size = np.shape(self.curve_array)
        nl_response_array = np.zeros(array_size)

        # calculate gradient of trajectory
        if self.grad is None:
            self.gradient()
        
        # evaluate response
        for i in range(array_size[1]):
            nl_response_array[:, i] = sys.nl_factor(self.curve_array[:, i])
        
        return Trajectory(nl_response_array)

    def jacob_init(self, sys):
        """
            Initialise a function that returns the jacobian of this dynamical
            system for a given scalar between 0 and 2*pi, which is the distance
            along a given trajectory.

            Parameters
            ----------
            traj: Trajectory
                the particular trajectory for which the jacobian will be
                evaluated over

            Returns
            -------
            jacobian: function
                the function that returns the jacobian of this instance of a
                dynamical system
        """
        def jacobian(i):
            """
                Return a (square) jacobian matrix for a given positive integer
                being the discretised location along the trajectory.

                Parameter
                ---------
                i: positive integer
                    the discretised location on the trajectory
            """
            # test for index, and input as index instead of s
            if i%1 != 0:
                raise TypeError("Inputs are not of the correct type!")
            i = int(i)
            state = self.curve_array[:, i]
            return sys.jacobian(state)
        return jacobian

    def normed_traj(self):
        """
            Calculate the norm of the vector-valued trajectory at each of its
            discretised locations, and return as another trajectory.

            Returns
            norm_traj: Trajectory
                the norm of the vector-valued trajectory at each location it is
                defined at
        """
        # norm_traj = np.zeros([1, np.shape(self.curve_array)[1]])
        # for i in range(np.shape(self.curve_array)[1]):
        #     norm_traj[0, i] = np.linalg.norm(self.curve_array[:, i])
        # return Trajectory(norm_traj)
        temp = np.linalg.norm(self.curve_array, axis = 0)
        return Trajectory(temp)

    def average_over_s(self):
        """
            This method performs integration of the trajectory over the domain
            of s (0 to 2*pi), which is analogous to the L^2 norm of a vector-
            valued function.
        """
        integ_traj = np.concatenate((self.curve_array, \
            self.curve_array[:, 0:1]), axis = 1)
        traj_disc = np.linspace(0, 2*np.pi, np.shape(integ_traj)[1])
        integ_vec = np.zeros([np.shape(self.curve_array)[0]])
        for i in range(np.shape(integ_traj)[0]):
            integ_vec[i] = (1/(2*np.pi))*integ.trapz(integ_traj[i, :], \
                traj_disc)
        return integ_vec

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
    from test_cases import ellipse as elps

    unit_circle1 = Trajectory(circ.x)
    unit_circle2 = 0.5*Trajectory(circ.x)

    unit_circle3 = np.pi*unit_circle1 + unit_circle2

    unit_circle1.plot(gradient = True, gradient_density = 32/256)
    unit_circle3.plot(gradient = True, gradient_density = 32/256)
    
    ellipse = Trajectory(elps.x)
    ellipse.plot(gradient = True, gradient_density = 32/256)