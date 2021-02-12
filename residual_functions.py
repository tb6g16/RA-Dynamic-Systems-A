# This file contains the function definitions that calculate the residuals and
# their associated gradients.

import numpy as np
import scipy.integrate as integ
from Trajectory import Trajectory
from System import System
from my_fft import my_fft, my_ifft
import trajectory_functions as traj_funcs

def local_residual(traj, sys, freq, mean):
    """
        This function calculates the local residual of a trajectory through a
        state-space defined by a given dynamical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory through state-space
        sys: System object
            the dynamical system defining the state-space
        freq: float
            the fundamental frequency of the trajectory
        
        Returns
        -------
        residual_traj: Trajectory object
            the local residual of the trajectory with respect to the dynamical
            system, given as an instance of the Trajectory class
    """
    # compute gradient of trajectory
    grad = traj_funcs.traj_grad(traj)
    grad_time = my_ifft(grad.modes)

    # evaluate system response at the states of the trajectory
    full_traj = traj
    full_traj[:, 0] = mean
    response = traj_funcs.traj_response(full_traj, sys.response)
    response_time = my_ifft(response.modes)

    # evaluate local residual, convert to frequency domain and return
    return Trajectory(my_fft((freq*grad_time) - response_time))

# ADD ON ZERO MODE OF RESIDUAL???
def global_residual(traj, sys, freq, mean, with_zero = False):
    """
        This function calculates the global residual of a trajectory through a
        state-space defined by a given dynamical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory through state-space
        sys: System object
            the dynamical system defining the state-space
        freq: float
            the fundamental frequency of the trajectory
        
        Returns
        -------
        global_res: float
            the global residual of the trajectory-system pair
    """
    # obtain set of local residual vectors
    local_res = local_residual(traj, sys, freq, mean)

    # take norm of the local residual vectors
    if with_zero == True:
        local_res_norm_sq = traj_funcs.traj_inner_prod(local_res, local_res)
    else:
        local_res_zero = local_res
        local_res_zero[:, 0] = 0
        local_res_norm_sq = traj_funcs.traj_inner_prod(local_res_zero, local_res_zero)

    # integrate over the discretised time
    return 0.5*np.real(local_res_norm_sq[0, 0])

def global_residual_grad(traj, sys, freq, mean):
    """
        This function calculates the gradient of the global residual with
        respect to the trajectory and the associated fundamental frequency for
        a trajectory through a state-space defined by a given dynamical system.

        Parameters
        ----------
        traj: Trajectory object
            the trajectory through state-space
        sys: System object
            the dynamical system defining the state-space
        freq: float
            the fundamental frequency of the trajectory
        
        Returns
        -------
        d_gr_wrt_traj: Trajectory object
            the gradient of the global residual with respect to the trajectory,
            given as an instance of the Trajectory class
        d_gr_wrt_freq: float
            the gradient of the global residual with respect to the trajectory
    """
    # calculate residual and gradient
    lr = local_residual(traj, sys, freq, mean)
    lr_grad = traj_funcs.traj_grad(lr)

    # perform multiplication with system jacobian (transpose)
    jac_transp = traj_funcs.jacob_init(traj, sys, if_transp = True)
    jac_lr_prod = jac_transp @ lr

    # convert to time domain
    lr_grad_time = my_ifft(lr_grad.modes)
    jac_lr_prod_time = my_ifft(jac_lr_prod.modes)

    # evaluate gradient w.r.t trajectory in time domain
    gr_traj_grad_time = -(freq*lr_grad_time) - jac_lr_prod_time

    # convert to frequency domain
    gr_traj_grad = Trajectory(my_fft(gr_traj_grad_time))

    # square norm of gradient of trajectory
    traj_grad = traj_funcs.traj_grad(traj)
    traj_grad_norm_sq = traj_funcs.traj_inner_prod(traj_grad, traj_grad)

    # response of full trajectory to system and inner product with trajectory gradient
    full_traj = traj
    full_traj[:, 0] = mean
    traj_resp = traj_funcs.traj_response(full_traj, sys.response)
    traj_grad_sys_prod = traj_funcs.traj_inner_prod(traj_grad, traj_resp)

    # combine into full "integrand"
    gr_freq_grad = np.real((freq*traj_grad_norm_sq[0, 0]) - traj_grad_sys_prod[0, 0])

    # return and don't forget to normalise zero mode
    return gr_traj_grad, gr_freq_grad
