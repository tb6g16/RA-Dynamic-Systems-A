# This file contains the object that holds all the important information about
# the optimisation problem. It will have the functionality to optimise the
# global residual in all the domains, and analyse the data too.

# Thomas Burton - October 2020

import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt

class Problem:
    """
        This class holds the required information to solve the RA otpimisation
        problem. Specifically: the dynamical system, history of Trajectories
        at each iteration, and the methods required for the residuals and
        post-processing. Consider this the master class of the numerical
        method.

        Attributes
        ----------
        trajectories: list of instances of Trajectory class
            list of all the saved trajectories, where the trajectory history
            will be utilised for post-processing
        dynamical_system: instance of System class
            the system for which we are trying the find the best periodic
            solution to
        fundamental_freq: list of floats
            the number of revolutions of the trajectory completed for a given
            unit of time, stored as list with a direct correspondence to the
            trajectory list
        global_residual: float
            the global 'error' between a trajectory and the given dynamical
            system
        
        Methods
        -------
        local_residual(trajectory = -1)
        global_residual(trajectory = -1)
        plot()
    """

    __slots__ = ['trajectories', 'dynamical_system', 'fundamental_freq', \
        'global_residual']

    def __init__(self, initial_trajectory, dynamical_system, fundamental_freq):
        self.trajectories = []
        self.trajectories.append(initial_trajectory)
        self.dynamical_system = dynamical_system
        self.fundamental_freq = []
        self.fundamental_freq.append(fundamental_freq)
        self.global_residual = []
        self.global_residual.append(self.compute_global_residual())
    
    # NEEDS VALIDATION
    def local_residual(self, trajectory = -1):
        """
            This function computes the local residual vector along the whole
            trajectory (according the discretisation prescribed by the
            trajectory class instance).

            Parameters
            ----------
            trajectory: integer
                the index of the trajectory list (picking a specific
                trajectory), defaults the last trajectory in the list
            
            Returns
            -------
            residual_array: numpy array
                the residual vector between the trajectory and the dynamical
                system at the given location along the trajectory
        """
        # initialise arrays
        trajectory_size = np.shape(self.trajectories[trajectory].curve_array)
        response_array = np.zeros(trajectory_size)

        # compute gradient of trajectory
        self.trajectories[trajectory].gradient()

        # evaluate system response at the states of the trajectory
        for i in range(np.shape(self.trajectories[trajectory].curve_array)[1]):
            state_now = self.trajectories[trajectory].curve_array[:, i]
            response_array[:, i] = self.dynamical_system.response(state_now)

        # compute value of residual vector
        residual_array = (self.fundamental_freq[trajectory]*\
            self.trajectories[trajectory].grad.curve_array) - response_array
        return residual_array

    # NEEDS VALIDATION
    def compute_global_residual(self, trajectory = -1):
        """
            This method computes the global residual between an instance of the
            Trajectory class and an instance of the System class.

            Parameters
            ----------
            trajectory: integer
                the index of the trajectory list (picking a specific
                trajectory), defaults the last trajectory in the list
        """
        # obtain set of local residual vectors
        local_residual_array = self.local_residual(trajectory = trajectory)

        # take norm of the local residual vectors
        local_residual_norm_vector = np.linalg.norm(local_residual_array, 2, \
            axis = 0)
        
        # integrate over the discretised time
        trajectory_discretisation = np.linspace(0, 2*np.pi, \
            np.shape(self.trajectories[trajectory].curve_array)[1])
        global_residual = (1/(4*np.pi))*integ.trapz(local_residual_norm_vector\
            , trajectory_discretisation)

        return global_residual

    def dglobal_res_dtraj(self, trajectory = -1):
        """
            This function calculates the gradient of the global residual with
            respect to its discretised trajectory.
        """
        freq = self.fundamental_freq[trajectory]
        traj = self.trajectories[trajectory]
        local_res = Trajectory(self.local_residual(trajectory = trajectory))
        local_res.gradient()
        jacob_func = traj.jacob_init(self.dynamical_system)
        return (-freq*local_res.grad) + (jacob_func*local_res)
    
    def dglobal_res_dfreq(self, trajectory = -1):
        """
            This function calculates the gradient of the global residual with
            respect to the frequency of its associated frequency.
        """
        freq = self.fundamental_freq[trajectory]
        traj = self.trajectories[trajectory]
        traj.gradient()
        traj_norm = traj.normed_traj()
        traj_response = traj.traj_response(self.dynamical_system)
        integrand = (2*freq*(traj_norm.curve_array**2)) - \
            (2*traj_response.traj_prod(traj.grad))
        traj_disc = np.linspace(0, 2*np.pi, np.shape(traj.curve_array)[1])
        return (1/(2*np.pi))*integ.trapz(integrand, traj_disc)

    def plot(self, trajectory = -1):
        return None

if __name__ == "__main__":
    # import trajectory and system functions
    from test_cases import van_der_pol as vpd
    from test_cases import unit_circle as circ

    # import class definitions
    from Trajectory import Trajectory
    from System import System

    # initialise above classes
    initial_trajectory = Trajectory(circ.x)
    dynamical_system = System(vpd)

    # initialise problem class and calculate global residuals
    test_problem = Problem(initial_trajectory, dynamical_system, 1)
    test_problem.compute_global_residual()
    print(test_problem.global_residual)

    # CHANGING PARAMETERS HAS NO EFFECT
    test_problem.dynamical_system.parameters['mu'] = 50
    test_problem.compute_global_residual()
    print(test_problem.global_residual)