# This file is here for initial tests of whether the code can solve the Lorenz
# system.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach A")
import numpy as np
import random as rand
from Trajectory import Trajectory
from System import System
from traj2vec import traj2vec, vec2traj
import trajectory_functions as traj_funcs
import residual_functions as res_funcs
import optimise as opt
from scipy.optimize import minimize
from test_cases import lorenz
from test_cases import unit_circle_3d as uc3

# define parameters and system
dim = 3
mean = np.zeros(3)
mean[2] = 25.55
sys = System(lorenz)

# define initial guess
init_traj = np.random.rand(3, 520)
init_traj = traj_funcs.swap_tf(init_traj)
init_traj = Trajectory(init_traj)
modes = 261
init_traj = Trajectory(uc3.x, modes = modes)
# for i in range(3):
#     for j in range(modes):
#         init_traj[i, j] = init_traj[i, j] + rand.uniform(-0.01, 0.01)
init_freq = (2*np.pi)/1.6
init_traj_vec = traj2vec(init_traj, init_freq)

# initialise optimisation functions
res_func, jac_func = opt.init_opt_funcs(sys, dim, mean)

# perform optimisation
sol = minimize(res_func, init_traj_vec, jac = jac_func, method = 'L-BFGS-B')

# unpack results
print(sol.message)
print("Number of iterations: " + str(sol.nit))
op_traj, op_freq = vec2traj(sol.x, dim)
# op_traj_full = op_traj
# op_traj_full[:, 0] = mean*(2*(modes - 1))

print("Period of orbit: " + str((2*np.pi)/op_freq))
print("Global residual before: " + str(res_func(init_traj_vec)))
print("Global residual after: " + str(res_func(traj2vec(op_traj, op_freq))))
# op_traj.plot(gradient = 1/4)
op_traj.plot(title = r"Guess: 3D UC (z = 0), Period: 1.6s -> 1.96s, GR: 361 -> 189")
# op_traj.plot(title = r"Guess: 3D UC (z = 0) w\ noise, Period: 1.6s -> 3.39s, GR: 361 -> 29")
# op_traj.plot(title = r"Guess: random time domain (inside unit spehere), Period: 1.6s -> 422s, GR: 39583 -> 18")
# op_traj.plot(title = r"Guess: random frequency domain (inside unit spehere), Period: 1.6s -> 1.96s, GR: 361 -> 189")

print("Trajectory zero mode: " + str(op_traj[:, 0]))
print("Local residual zero mode: " + str(res_funcs.local_residual(op_traj, sys, op_freq, np.zeros(3))[:, 0]))
