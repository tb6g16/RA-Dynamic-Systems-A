# This file contains the definition of the class that defines the dynamical
# system in state-space.

# Thomas Burton - October 2020

import numpy as np
import matplotlib.pyplot as plt

class System:
    """
        This class defines the system of equations, defining a dynamical system
        (a state-space vector field). This class allows the modification of any
        other associated parameters that define the behaviour of the dynamical
        system to modified dynamically (without having to generate a new
        instance of the class).

        Attributes
        ----------
        response: function
            the function defining response of the system for a given state 
            (and optional parameters)
        optionals: dict
            a dictionary defining all the optional parameters used to modify
            the behaviour of the dynamical system
        
        Methods
        -------
        plot()
    """

    __slots__ = ['response', 'jacobian', 'parameters']

    def __init__(self, function_file):
        """
            Initialise instance of the System class with a specific dynamical
            system described as function in a separate python file.

            Parameters
            ----------
            function_file: module
                the file containing the definition of the function defining the
                behaviour of the dynamical system, with optional parameters
                given as a separate deictionary called "defaults"
        """
        if ('response' or 'jacobian') not in dir(function_file):
            raise AttributeError("The file does not contain the required \
                functions!")
        self.response = function_file.response
        self.jacobian = function_file.jacobian
        if 'defaults' in dir(function_file):
            self.parameters = function_file.defaults
        else:
            self.parameters = None
    
    def traj_response(self, traj):
        return None
    
    def jacob_init(self, traj):
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
        def jacobian(s):
            """
                Return a (square) jacobian matrix for a given scalar s between
                0 and 2*pi.
            """
            if type(s) != float and type(s) != int:
                raise TypeError("Inputs are not of the correct type!")
            i = int((256/(2*np.pi))*s)
            state = traj.curve_array[:, i]
            return self.jacobian(state, defaults = self.parameters)
        return jacobian

    def plot(self, domain=[[-1, 1], [-1, 1]], disc = [20, 20]):
        """
            This method plots the state-space response of the dynamical system
            given by the particular instance of the System class over a
            cartesian grid (in 2D).

            Parameters
            ----------
            domain: 2-by-2 array of floats
                the boundaries of the domain
            disc: list of floats
                the number of discrete points to take inside the domain, for
                the respective directions
        """
        # discretise domain
        x_dir_discretised = np.linspace(domain[0][0], domain[0][1], disc[0])
        y_dir_discretised = np.linspace(domain[1][0], domain[1][1], disc[1])
        X_discretised, Y_discretised = np.meshgrid(x_dir_discretised, \
            y_dir_discretised)
        
        # initialise arrays
        domain_array_size_list = list(np.shape(X_discretised))
        domain_array_size_list.append(2) # NOT GENERAL
        response_array = np.zeros(domain_array_size_list)
        
        # response vectors on domain
        for i in range(disc[0]):
            for j in range(disc[1]):
                response_array[i, j, :] = self.response([X_discretised[i, j], \
                    Y_discretised[i, j]])
        
        # plot vector field
        plt.figure()
        ax = plt.gca()
        ax.quiver(X_discretised, Y_discretised, response_array[:, :, 0], \
            response_array[:, :, 1])
        plt.xlabel(r"$x$"), plt.ylabel(r"$\dot{x}$")
        ax.set_aspect("equal")
        plt.show()

        return None

if __name__ == "__main__":
    from test_cases import van_der_pol as vpd

    system = System(vpd)
    
    system.plot(domain = [[-2, 2], [-2, 2]])
    system.parameters['mu'] = 1
    system.plot(domain = [[-2, 2], [-2, 2]])
    system.parameters['mu'] = 2
    system.plot(domain = [[-2, 2], [-2, 2]])

# TRAJECTORY RESPONSE AND JACOBIAN METHODS