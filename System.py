# This file contains the definition of the class that defines the dynamical
# system in state-space.

# Thomas Burton - October 2020

class System:
    """
        This class defines the system of equations, defining a dynamical system
        (a state-space vector field). This class allows the modification of any
        other associated parameters that define the behaviour of the dynamical
        system to modified dynamically (without having to generate a new
        instance of the class).

        Attributes
        ----------
        response: function
            the function defining response of the system for a given state 
            (and optional parameters)
        optionals: dict
            a dictionary defining all the optional parameters used to modify
            the behaviour of the dynamical system
        
        Methods
        -------
        plot() (TO BE IMPLEMENTED)
    """

    __slots__ = ['response', 'optionals']

    def __init__(self, function_file):
        self.response = function_file.g
        self.optionals = function_file.defaults

if __name__ == "__main__":
    from test_cases import van_der_pol as vpd
    
    system = System(vpd)
    
    print(system.response([0, 1]))
    system.optionals['mu'] = 1
    print(system.response([0, 1]))
