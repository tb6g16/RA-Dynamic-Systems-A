# This file contains the functions defining the dynamical system of the Van der
# Pol equations, the proposed solution curve, and the real solution (limit
# cycle).

# Thomas Burton - October 2020

import numpy as np

# define optional arguments
defaults = {'mu': 0, 'a': np.pi}

def g(x, defaults = defaults):
    # unpack defaults
    mu = defaults['mu']
    # initialise response vector
    response = np.zeros(np.shape(x))
    # assign response
    response[0] = x[1]
    response[1] = (mu*(1 - (x[0] ** 2))*x[1]) - x[0]
    return response

if __name__ == "__main__":
    print(g([0, 1]))
