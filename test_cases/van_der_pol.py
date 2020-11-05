# This file contains the functions defining the dynamical system of the Van der
# Pol equations, the proposed solution curve, and the real solution (limit
# cycle).

# Thomas Burton - October 2020

import numpy as np

# define optional arguments
defaults = {'mu': 0}

def response(x, defaults = defaults):

    # unpack defaults
    mu = defaults['mu']

    # initialise response vector
    response = np.zeros(np.shape(x))

    # assign response
    response[0] = x[1]
    response[1] = (mu*(1 - (x[0] ** 2))*x[1]) - x[0]

    return response

def jacobian(x, defaults = defaults):

    # unpack defaults
    mu = defaults['mu']

    #initialise jacobian matrix
    jacobian_matrix = np.zeros([np.shape(x), np.shape(x)])

    # compute jacobian elements
    jacobian_matrix[0, 1] = 1
    jacobian_matrix[1, 0] = -(2*mu*x[0]*x[1]) - 1
    jacobian_matrix[1, 1] = mu*(1 - (x[0] ** 2))

    return jacobian_matrix
