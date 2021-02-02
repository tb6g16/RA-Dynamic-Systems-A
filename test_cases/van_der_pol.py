# This file contains the functions defining the dynamical system of the Van der
# Pol equations.

import numpy as np

# define parameters
parameters = {'mu': 0}

def response(x, defaults = parameters):

    # unpack defaults
    mu = defaults['mu']

    # initialise response vector
    response = np.zeros(np.shape(x))

    # assign response
    response[0] = x[1]
    response[1] = (mu*(1 - (x[0] ** 2))*x[1]) - x[0]

    return response

def jacobian(x, defaults = parameters):

    # unpack defaults
    mu = defaults['mu']

    #initialise jacobian matrix
    jacobian = np.zeros([np.shape(x)[0], np.shape(x)[0]])

    # compute jacobian elements
    jacobian[0, 1] = 1
    jacobian[1, 0] = -(2*mu*x[0]*x[1]) - 1
    jacobian[1, 1] = mu*(1 - (x[0] ** 2))

    return jacobian

def nl_factor(x, defaults = parameters):

    # unpack defualts
    mu = defaults['mu']

    # initialise output vector
    nl_vector = np.zeros(np.shape(x))

    # assign values to vector
    nl_vector[1] = -mu*(x[0]**2)*x[1]

    return nl_vector

# these functions are here because the system has non-quadratic nonlinearity
def init_nl_con_grads():

    def nl_con_grad1(x, defaults = parameters):
        return np.zeros(2)

    def nl_con_grad2(x, defaults = parameters):
        mu = defaults['mu']
        vec = np.zeros(2)
        vec[0] = -mu*x[0]*x[1]/np.pi
        vec[1] = -(mu/(2*np.pi))*(x[0]**2)
        return vec

    return [nl_con_grad1, nl_con_grad2]
