# This file contains the function to plot state-space solution curves with a given
# dynamical system in 2D or 3D.

# Thomas Burton - October 2020

import numpy as np
import matplotlib.pyplot as plt

def plot(dynamics, curve):
    """
        This function plots the given dynamical system and trajectory in the same space, as a vector
        field and as a curve respectively.

        Parameters
        ----------
        dynamics: function
            the dynamical system defining the system response at
            each state
        curve: function
            a given trajectory in state-space
    """

    # define some constants
    s_d = 200
    x_d = 30
    y_d = 30

    # initialise vectors
    trajectory_x = np.zeros([s_d])
    trajectory_x_dot = np.zeros([s_d])
    response_x = np.zeros([x_d, y_d])
    response_x_dot = np.zeros([x_d, y_d])

    # discretise domain on trajectory and of the dynamical system
    s = np.linspace(0, 2*np.pi, s_d)
    X, X_dot = np.meshgrid(np.linspace(-1.5, 1.5, x_d), np.linspace(-1.5, 1.5, y_d))

    # generate trajectory
    for i in range(s_d):
        trajectory_x[i] = curve(s[i])[0]
        trajectory_x_dot[i] = curve(s[i])[1]

    # calculate response of the dynamical system at each state
    for i in range(x_d):
        for j in range(y_d):
            response_x[i][j] = dynamics([X[i][j], X_dot[i][j]])[0]
            response_x_dot[i][j] = dynamics([X[i][j], X_dot[i][j]])[1]
    
    # plot dynamical system
    plt.figure()
    ax = plt.gca()
    ax.quiver(X, X_dot, response_x, response_x_dot)
    ax.plot(trajectory_x, trajectory_x_dot)
    plt.xlabel(r"$x$"), plt.ylabel(r"$\dot{x}$")
    ax.set_aspect("equal")
    plt.show()

    return None

if __name__ == "__main__":
    from test_cases import van_der_pol as vpd
    from test_cases import harmonic_oscillator as harm

    plot(vpd.g, harm.solution_curve)
    # plot(harm.dynamical_system, harm.solution_curve)