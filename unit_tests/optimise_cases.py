# This file contains a number of test cases for the optimisation methods
# implemented, such as to validate the behaviour.

import sys
sys.path.append(r"C:\Users\user\Desktop\PhD\Bruno Paper\Code\Approach A")
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Trajectory import Trajectory
from System import System
import optimise as my_opt
from traj2vec import traj2vec, vec2traj
from test_cases import unit_circle as uc
from test_cases import ellipse as elps
from test_cases import van_der_pol as vpd
from test_cases import viswanath as vis

res_trace = []
traj_trace = []

def callback(x):
    res_trace.append(res_func(x))
    current_traj, _ = vec2traj(x, dim)
    traj_trace.append(current_traj)

# Case 1: circle to van-der-pol oscillator
n = 1
i = 2
a = 0
m = 1.2
freq = 1
dim = 2
mean = np.zeros(2)
circle = 2*Trajectory(uc.x, modes = 65)
# circle_almost = circle.curve_array
# circle_almost[n, i] = m*circle.curve_array[n, i] + a
# circle_almost = Trajectory(circle_almost)
sys = System(vpd)
sys.parameters['mu'] = 2

res_func, jac_func = my_opt.init_opt_funcs(sys, dim, mean)

op_vec = opt.minimize(res_func, traj2vec(circle, freq), jac = jac_func, method = 'L-BFGS-B', callback = callback)
print(op_vec.message)
print("Number of iterations: " + str(op_vec.nit))

op_traj, op_freq = vec2traj(op_vec.x, dim)

print("Period of orbit: " + str((2*np.pi)/op_freq))
print("Global residual before: " + str(res_func(traj2vec(circle, freq))))
print("Global residual after: " + str(res_func(traj2vec(op_traj, op_freq))))

traj_trace.insert(0, circle)
for i in range(op_vec.nit):
    fname = "vpd_convergence_it" + str(i) + ".jpg"
    traj_trace[i].plot(gradient = 1/4)
    plt.savefig(r'C:\Users\user\Desktop\temp' + '\\' + fname, bbox_inches='tight')
    plt.close()

# circle_almost.plot(gradient = True, gradient_density = 32/256)
op_traj.plot(gradient = 1/4)

# plt.figure()
# plt.plot(res_trace)
# plt.show()

# Case 2: non-unit circle with viswanath system

# Case 3: circle input to nonlinear vpd and result input to linear vpd

# Case 4: vpd result input to viswanath system to get circle back out
