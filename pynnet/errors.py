import numpy
from base import theano

__all__ = ['mse', 'nll', 'class_error']

class Error(object):
    def ___abname(self):
        raise NotImplementedError
    name = property(___abname, doc=r"""This is the name of your error function (a string).  It only needs to be readable.""")
    
    def __repr__(self):
        return self.name

    def __call__(self, os, y):
        r"""
        This is where you should apply your error function.
        
        Parameters (the names don't have to be the same):
        os -- the actual output of the network
        y  -- the target outputs

        You must return a single value that is the mean of the error
        over all the examples given.
        """
        raise NotImplementedError

    def th(self, os, y):
        r"""
        This function is to implement the same computation as the
        __call__ above using sympbolics from theano.

        Parameters (the names don't have to be the same):
        os -- the symbolic variable for the output
        y -- the symbolic variable for the targets

        You must return a single symbolic variable that would compute
        the mean of the error over all examples.  You must use only
        symbolic computations for this method.

        Since theano is an optional requirement, test the thruth of
        the theano member of base before actually using any theano
        function.  If you cannot implement the method at all unless
        theano is imported (which is the common case) protect the
        method definition by
        
        if theano:
            def th(self, os, y):
               <bla bla>

        There is no need to implement a corresponding theano version
        of the gradient since it is automatically computed.
        """

        raise NotImplementedError

    def _(self, os, y, C):
        r"""
        This is where you implement the derivative of your function.

        Parameters (the names don't have to be the same):
        os -- the actual outputs of the network
        y  -- the target outputs
        C  -- the computed cost for these inputs

        You must return a gradient of the error over the outputs.  It
        must have the same dimension as one output vector.
        """
        raise NotImplementedError

    def test(self):
        r"""
        Convinience procedure to test the gradient accuracy.
        """
        yc = numpy.random.random((2,5))
        yt = numpy.random.random((2,5))
        self.test_grad(yc, yt, verbose=False)

    def _estim_grad(self, yc, yt, eps):
        r"""
        Estimate the gradient using finite differences.
        """
        C = self(yc, yt)
        ye = yc.copy()
        G = numpy.empty(ye.shape)
        for i, v in numpy.ndenumerate(ye):
            ye[i] += eps
            G[i] = (self(ye, yt) - C) / eps
            ye[i] = v

        return G

    def test_grad(self, yc, yt, eps = 1e-5, verbose=True):
        r"""
        Test that the gradient is good by comparing against finite differences.
        """
        C = self(yc, yt)
        
        y_f = self._(yc, yt, C)
        y_e = self._estim_grad(yc, yt, eps)
            
        eval = y_f/y_e
        
        if verbose: print eval
        if eval.max() > 1.01 or eval.min() < 0.99:
            raise ValueError('Gradient is not within norms')

class Mse(Error):
    r"""
    Error class for Meas Squared Error.

    The error is computed like this:
        ((out-target)^2).mean()

    This is a good measure for regression tasks.

    TESTS::
    
    >>> mse
    mse
    """
    name = "mse"

    def __call__(self, os, y):
        return ((os-y)**2).mean()

    if theano:
        def th(self, os, y):
            return theano.tensor.mean((os-y)**2)

    def _(self, os, y, C):
        r"""
        >>> mse.test()
        """
        return 2*(os-y)/os.size

mse = Mse()

class Nll(Error):
    r"""
    TESTS::

    >>> nll
    nll
    """
    name = "nll"

    def __call__(self, os, y):
        return (-numpy.log(os[y.astype(numpy.bool)])).mean()

    def _(self, os, y, C):
        r"""
        >>> nll.test()
        """
        res = numpy.zeros(os.shape)
        sel = y.astype(numpy.bool)
        res[sel] = -1/(os[sel]*os.size)
        return res

    def test(self):
        r"""
        Convinience procedure to test the gradient accuracy.

        Modified to to use only 0 or 1 as targets.
        """
        yc = numpy.random.random((2,5))
        yt = numpy.random.randint(2, size=(2,5))
        self.test_grad(yc, yt, verbose=False)

nll = Nll()

class Class_error(Error):
	name="class_error"
	
	def __call__(self, os, y):
            return numpy.abs((numpy.round(os)-y)).mean()
        
class_error = Class_error()
