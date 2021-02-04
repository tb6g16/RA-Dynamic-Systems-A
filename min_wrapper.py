# This file defines a wrapper function for scipy.optimize.minimize to allow for
# easier analysis of the common points of interest for the RA minimisation.

from scipy.optimize import minimize
from optimise import init_opt_funcs
from residual_functions import local_residual
from traj2vec import traj2vec, vec2traj
import matplotlib.pyplot as plt

def my_min(traj, sys, freq, mean, **kwargs):
    """
        This function works as a wrapper for the minimization function provided
        by scipy. The point of this function is to reduce the amount of code
        required to set-up and analyse a solution.
    """
    # unpack keyword arguments
    my_method = kwargs.get('method', 'L-BFGS-B')
    if_quiet = kwargs.get('quiet', False)

    # setup the problem
    dim = traj.shape[0]
    res_func, jac_func = init_opt_funcs(sys, dim, mean)

    # define varaibles to be tracked using callback
    traj_trace = []
    freq_trace = []
    lr_trace = []
    gr_trace = []
    gr_grad_trace = []

    # define callback function
    def callback(x):
        cur_traj, cur_freq = vec2traj(x, dim)
        cur_gr = res_func(x)

        # print("-----------------------------------")
        # print("Global residual: " + str(cur_gr))

        traj_trace.append(cur_traj)
        freq_trace.append(cur_freq)
        lr_trace.append(local_residual(cur_traj, sys, cur_freq, mean))
        gr_trace.append(cur_gr)
        gr_grad_trace.append(jac_func(x))

    # convert trajectory to vector of optimisation variables
    traj_vec = traj2vec(traj, freq)

    # define options
    if if_quiet == True:
        options = {'disp': False}
    else:
        options = {'disp': True}

    # perform optimisation
    sol = minimize(res_func, traj_vec, jac = jac_func, method = my_method, callback = callback, options = options)

    # pack traces into a dictionary
    traces = {'traj': traj_trace, 'freq': freq_trace, 'lr': lr_trace, 'gr': gr_trace, 'gr_grad': gr_grad_trace}

    return sol, traces

if __name__ == "__main__":
    import numpy as np
    from test_cases import unit_circle as uc
    from test_cases import van_der_pol as vpd
    from Trajectory import Trajectory
    from System import System

    traj = 2*Trajectory(uc.x)
    sys = System(vpd)
    sys.parameters['mu'] = 2
    freq = 1
    mean = np.zeros(2)

    sol, traces = my_min(traj, sys, freq, mean)

    print(sol.message)
    op_traj, op_freq = vec2traj(sol.x, 2)
    op_traj.plot(gradient = 1/8, time_disc = 256)
    plt.plot(traces['gr'])
    plt.show()
