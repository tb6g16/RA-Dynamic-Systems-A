# This file holds the function definition for the Lorenz system.

import numpy as np

# define parameters
# parameters = {'rho': 30, 'beta': 8/3, 'sigma': 10} # Davide values
parameters = {'rho': 28, 'beta': 8/3, 'sigma': 10} # Dong (2018)

def response(x, defaults = parameters):
    
    # unpack defaults
    rho = defaults['rho']
    beta = defaults['beta']
    sigma = defaults['sigma']

    # intialise response vector
    response = np.zeros(np.shape(x))

    # assign response
    response[0] = sigma*(x[0] - x[1])
    response[1] = (rho*x[0]) - x[1] - (x[0]*x[2])
    response[2] = (x[0]*x[1]) - (beta*x[2])

    return response

def jacobian(x, defaults = parameters):
    
    # unpack defaults
    rho = defaults['rho']
    beta = defaults['beta']
    sigma = defaults['sigma']

    # initialise jacobian matrix
    jacobian = np.zeros([np.shape(x)[0], np.shape(x)[0]])

    # compute jacobian elements
    jacobian[0, 0] = sigma
    jacobian[0, 1] = -sigma
    jacobian[1, 0] = rho - x[2]
    jacobian[1, 1] = -1
    jacobian[1, 2] = -x[0]
    jacobian[2, 0] = x[1]
    jacobian[2, 1] = x[0]
    jacobian[2, 2] = -beta

    return jacobian

def nl_factor(x, defaults = parameters):

    # initialise output vector
    nl_vector = np.zeros(np.shape(x))

    # assign values
    nl_vector[1] = -x[0]*x[3]
    nl_vector[2] = x[0]*x[1]

    return nl_vector

def init_nl_con_grads():
    
    def nl_con_grad1(x, defaults = parameters):
        pass

    def nl_con_grad2(x, defaults = parameters):
        pass
    
    return [nl_con_grad1, nl_con_grad2]