import numpy as np


class ComputerExperiment:

    def __init__(self, input_dim, input_box, output_dim, f, fname=None):
        ''' Computer experiment setting

        Parameters
        ----------
        input_dim: integer
            Dimension of the input space
        input_box: list of list
            Input domain
            [[x^min_1, ..., x^min_{input_dim}], [x^max_1, ..., x^max_{input_dim}]]
        self.output_dim
            Number of outputs
        f: function, list of functions
            Functions to be evaluated    
        fname: string
            Function name
        foptargs: list
            Optional parameters
        '''
        self.input_dim = input_dim
        self.input_box = input_box
        self.output_dim = output_dim
        self.f = f
        self.fname = fname

    def eval(self, x):

        n = x.shape[0]

        if isinstance(self.f, list):
            z_ = []
            for i in range(self.output_dim):
                z_.append(self.f[i](x).reshape((n, 1)))
            z = np.hstack(z_)
        else:
            z = self.f(x).reshape((n, self.output_dim))

        return z
