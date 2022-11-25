import numpy as np

class Problem:
    """ class Problem:
          def __init__(self):
            self.dim = 1
            self.p = 1
            self.box = [[-1], [1]]

          def eval(self, x):
            return gp.misc.testfunctions.twobumps(x)
    """
    
    def __init__(self, dim, box, p, f, fname=None):

        self.dim = dim
        self.box = box
        self.p = p
        self.f = f
        self.fname = fname

    def eval(self, x):
        
        n = x.shape[0]

        if isinstance(self.f, list):
            z_ = []
            for i in range(self.p):
                z_.append(self.f[i](x).reshape((n, 1)))
            z = np.hstack(z_)
        else:
            z = self.f(x).reshape((n, self.p))
                          
        return z
