import numpy

__all__ = ['tanh', 'sigmoid', 'none', 'Anynlin']

class Nlin(object):
    def __call__(self, x):
        raise NotImplementedError
    
    def _(self, x, y):
        raise NotImplementedError

    def test(self):
        self.test_grad(verbose=False)

    def test_grad(self, eps = 1e-5, verbose=True):
        x = numpy.random.random((2,10))
        y = self(x)
        
        y_c = self._(x, y)
    
        xe = x + eps
        y_e = (self(xe) - y)/eps

        eval = y_c/y_e

        if verbose: print eval
        if eval.max() > 1.01 or eval.min() < 0.99:
            raise ValueError('Gradient is not within norms')

class Tanh(Nlin):
    def __repr__(self):
        return "tanh"
    
    def __call__(self, x):
        return numpy.tanh(x)

    def _(self, x, y):
        return 1.0 - y**2

tanh = Tanh()

class Sigmoid(Nlin):
    def __call__(self, x):
        return 1/(1+numpy.e**(-x))

    def _(self, x, y):
        return y - y**2

sigmoid = Sigmoid()

class Lin(Nlin):
    def __call__(self, x):
        return x

    def _(self, x, y):
        return numpy.ones(x.shape, dtype=x.dtype)

none = Lin()

class Softmax(Nlin):
    def __call__(self, x):
        v = numpy.exp(x - x.mean())
        return v / numpy.sum(v, axis=1).repeat(x.shape[1]).reshape(x.shape)

#    def _(self, x, y):

#softmax = Softmax()

class Anynlin(Nlin):
    def __init__(self, func, eps=1e-5):
        Nlin.__init__(self)
        self.func = func
        self.eps = eps

    def __call__(self, x):
        return self.func(x)

    def _(self, x, y):
        xe = x + self.eps
        ye = self.func(xe)
        return (ye - y)/self.eps
