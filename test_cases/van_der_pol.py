# This file contains the functions defining the dynamical system of the Van der Pol equations,
# the proposed solution curve, and the real solution (limit cycle).

# Thomas Burton - October 2020

import numpy as np

def g(x):
    # nonlinear damping coefficient
    mu = 1
    # initialise response vector
    response = np.zeros(np.shape(x))
    # assign response
    response[0] = x[1]
    response[1] = (mu*(1 - (x[0] ** 2))*x[1]) - x[0]
    return response