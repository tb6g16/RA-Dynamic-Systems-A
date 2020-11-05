# This file contains the function definitions that will optimise a given
# trajectory and fundamental frequency to find the lowest global residual for
# a given dynamical system.

# Thomas Burton - November 2020

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
    return None

def vec2traj(opt_vector):
    """
        This function converts an optimisation variable back into its
        corresponding trajectory frequency pair.

        Parameters
        ----------
        opt_vector: numpy array
            the optimisation vector
        
        Returns
        -------
        traj: Trajectory
            the corresponding trajectory
        freq: float
            the corresponding frequency
    """
    return None

def init_optimise_time(sys):
    """
        This function initialises an optimisation function (in time domain) for
        a specific dynamical system.

        Parameters
        ----------
        sys: System
            the dynamical system to optimise with respect to
        
        Returns
        -------
        optimise_time: function
            the function that optimises an initial trajectory and frequency to
            minimise the global residual with respect to the given dynamical
            system
    """
    def optimise_time(init_traj, init_freq, iter_max = 100):
        """
            This function takes an initial trajectory and frequency and optimises
            them to minimise a global residual with respect to a given dynamical
            system.

            Parameters
            ----------
            init_traj: Trajectory
                the initial trajectory from which to start optimising
            init_freq: float
                the initial frequency from which to start optimisaing
            iter_max: positive integer
                the maximum number of iterations of the optimisation to perform
                before forcibly terminating
            
            Returns
            -------
            opt_traj: Trajectory
                the optimal trajectory to produce a minimum in the global
                residual
            opt_freq: float
                the optimal frequency to produce a minimum in the global
                residual
        """
        return None
    return optimise_time
