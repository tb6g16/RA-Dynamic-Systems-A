# This file is here for initial tests of whether the code can solve the Lorenz
# system.

import numpy as np
import random as rand
from Trajectory import Trajectory
from System import System
from traj2vec import traj2vec, vec2traj
import trajectory_functions as traj_funcs
import matplotlib.pyplot as plt
from min_wrapper import my_min
from test_cases import lorenz
from test_cases import unit_circle_3d as uc3

# define parameters and system
mean = np.zeros(3)
mean[2] = 25.55
sys = System(lorenz)

# define initial guess
modes = 10
# init_traj = np.random.rand(3, modes)
# init_traj = traj_funcs.swap_tf(init_traj)
# init_traj = Trajectory(init_traj)
# modes = 250
init_traj = Trajectory(uc3.x, modes = modes)*2
# for i in range(3):
#     for j in range(modes):
#         init_traj[i, j] = init_traj[i, j] + rand.uniform(-0.01, 0.01)
init_freq = (2*np.pi)/1.6
init_traj_vec = traj2vec(init_traj, init_freq)

# perform optimisation
op_traj, op_freq, traces, sol = my_min(init_traj, sys, init_freq, mean)

# results
print(sol.message)
print("Period of orbit: " + str((2*np.pi)/op_freq))
op_traj.plot(gradient = 1/4, disc = 256, mean = mean)
# plt.figure(1)
# plt.plot(traces['gr'])
# plt.show()

# lr_zero_mode = np.zeros([3, len(traces['lr'])])
# for i in range(len(traces['lr'])):
#     lr_zero_mode[:, i] = traces['lr'][i][:, 0]

# _, (ax1, ax2, ax3) = plt.subplots(figsize = (12, 5), nrows = 3)
# ax1.plot(np.abs(lr_zero_mode[0, :]))
# ax2.plot(np.abs(lr_zero_mode[1, :]))
# ax3.plot(np.abs(lr_zero_mode[2, :]))
# plt.show()
