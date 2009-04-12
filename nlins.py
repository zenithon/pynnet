import numpy

__all__ = ['tanh', 'sigmoid', 'none', 'Anynlin']

class Nlin(object):
    def ___abname(self):
        raise NotImplementedError
    name = property(___abname, doc=r"""This is the name of your nonlinear function (a string).  It only needs to be readable.""")

    def __repr__(self):
        return self.name

    def __call__(self, x):
        r"""
        This is where you apply your function.
        
        Parameters (the names don't have to be the same):
        x -- the inputs

        Your return value must be an array of the same dtype and the
        same shape as the input.
        """
        raise NotImplementedError
    
    def _(self, x, y):
        r"""
        This is where you compute the dericative of your function.
        
        Parameters (the names don't have to be the same):
        x -- the inputs
        y -- the output of your function for these inputs

        Your return value must be an array of the same dtype and the
        same shape as the input.
        """
        raise NotImplementedError

    def test(self):
        x = numpy.random.random((2,10))
        y = self(x)
        self.test_grad(x, y, verbose=False)

    def _estim_grad(self, x, y, eps):
        return (self(x+eps) - y)/eps

    def test_grad(self, x, y, eps=1e-5, verbose=True):
        y_c = self._(x, y)
        y_e = self._estim_grad(x, y, eps)

        eval = y_c/y_e

        if verbose: print eval
        if eval.max() > 1.01 or eval.min() < 0.99:
            raise ValueError('Gradient is not within norms')

class Tanh(Nlin):
    r"""
    TESTS::
    >>> tanh
    tanh
    """
    name = "tanh"
    
    def __call__(self, x):
        return numpy.tanh(x)

    def _(self, x, y):
        r"""
        TESTS::
        >>> tanh.test()
        """
        return 1.0 - y**2

tanh = Tanh()

class Sigmoid(Nlin):
    r"""
    TESTS::
    >>> sigmoid
    sigmoid
    """
    name = "sigmoid"

    def __call__(self, x):
        return 1/(1+numpy.e**(-x))

    def _(self, x, y):
        r"""
        TESTS::
        >>> sigmoid.test()
        """
        return y - y**2

sigmoid = Sigmoid()

class Lin(Nlin):
    r"""
    TESTS::
    >>> none
    none
    """
    name = "none"

    def __call__(self, x):
        return x

    def _(self, x, y):
        r"""
        TESTS::
        >>> none.test()
        """
        return numpy.ones(x.shape, dtype=x.dtype)

none = Lin()

class Softmax(Nlin):
    name = "softmax"

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

    def __repr__(self):
        return repr(self.func)

    def __call__(self, x):
        return self.func(x)

    def _(self, x, y):
        return self._estim_grad(x, y, self.eps)
