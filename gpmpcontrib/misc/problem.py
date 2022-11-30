import numpy as np

class Problem:
    """ class Problem:
          def __init__(self):
            self.dim_inputs = 1
            self.dim_outputs = 1
            self.box = [[-1], [1]]

          def eval(self, x):
            return gp.misc.testfunctions.twobumps(x)
    """
    
    def __init__(self, dim_inputs, box_inputs, dim_outputs, f, fname=None):

        self.dim_inputs = dim_inputs
        self.box_inputs = box_inputs
        self.dim_outputs = dim_outputs
        self.f = f
        self.fname = fname

    def eval(self, x):
        
        n = x.shape[0]

        if isinstance(self.f, list):
            z_ = []
            for i in range(self.dim_outputs):
                z_.append(self.f[i](x).reshape((n, 1)))
            z = np.hstack(z_)
        else:
            z = self.f(x).reshape((n, self.dim_outputs))
                          
        return z
