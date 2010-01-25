from base import *

from nlins import *
from errors import *

__all__ = ['SimpleNet']

class SimpleNet(BaseObject):
    r"""
    Simple 3-layers neural network to experiment with new algorithms
    or implementations.

    WARNING: Not all functionality of this class is guaranteed to
    always work.  You should do some sanity checks before trusting
    results.

    If you are not tinkering with the implementation, you should use
    the `pynnet.NNet` class which is documented and stable.
    """
    
    def __init__(self, ninputs, nhidden, noutputs, hnlin=tanh, onlin=none, error=mse, dtype=numpy.float32, use_th=theano):
        r"""
        Parameters:
        ninputs : Dimension of input
        nhidden : Dimension of hidden layer
        noutput : Dimension of output
        hnlin : hidden transfer function
        onlin : output transfer function
        error : Cost function to optimize
        dtype : data type of coefficients
        """
        self.use_th = use_th

        self.hnlin = hnlin
        self.onlin = onlin
        self.err = error

	if use_th:
            m = theano.Module()
            if dtype == numpy.float32:
                sx, sy, m.W1, m.W2 = theano.tensor.fmatrices('x', 'y', 'W1', 'W2')
                m.b1, m.b2 = theano.tensor.fvectors('b1', 'b2')
            elif dtype == numpy.float64:
                sx, sy, m.W1, m.W2 = theano.tensor.dmatrices('x', 'y', 'W1', 'W2')
                m.b1, m.b2 = theano.tensor.dvectors('b1', 'b2')
            sha = theano.dot(sx, m.W1) + m.b1
            shs = self.hnlin.th(sha)
            soa = theano.dot(shs, m.W2) + m.b2
            sos = self.onlin.th(soa)
            m.eval = theano.Method(sx, sos)
            serr = self.err.th(sos, sy)
            m.test = theano.Method([sx, sy], serr)
            sg = theano.tensor.grad(serr, [m.W1, m.W2, m.b1, m.b2])
            m.grad = theano.Method([sx, sy], sg)
            self._m = m.make(W1 = numpy.random.uniform(low=-1/numpy.sqrt(ninputs), high=1/numpy.sqrt(ninputs), size=(ninputs, nhidden)).astype(dtype),
                             W2 = numpy.random.uniform(low=-1/numpy.sqrt(ninputs), high=1/numpy.sqrt(ninputs), size=(nhidden, noutputs)).astype(dtype),
                             b1 = numpy.zeros(nhidden, dtype=dtype),
                             b2 = numpy.zeros(noutputs, dtype=dtype))
        
        else:
            self.W1 = numpy.random.uniform(low=-1/numpy.sqrt(ninputs), high=1/numpy.sqrt(ninputs), size=(ninputs, nhidden)).astype(dtype)
            self.W2 = numpy.random.uniform(low=-1/numpy.sqrt(ninputs), high=1/numpy.sqrt(ninputs), size=(nhidden, noutputs)).astype(dtype)
            self.b1 = numpy.zeros(nhidden, dtype=dtype)
            self.b2 = numpy.zeros(noutputs, dtype=dtype)

    def _save_(self, file):
        file.write('SNx')
        pickle.dump((self.hnlin, self.onlin, self.err, self.use_th), file, pickle.HIGHEST_PROTOCOL)
        if self.use_th:
            pickle.dump(self._m, file, pickle.HIGHEST_PROTOCOL)
        else:
            numpy.save(file, self.W1)
            numpy.save(file, self.b1)
            numpy.save(file, self.W2)
            numpy.save(file, self.b2)
        
    def _load_(self, file):
        s = file.read(3)
        if s != 'SNx':
            raise ValueError('Wrong fromat for SimpleNet in file')
        self.hnlin, self.onlin, self.err, self.use_th = pickle.load(file)
        if self.use_th:
            self._m = pickle.load(file)
        else:
            self.W1 = numpy.load(file)
            self.b1 = numpy.load(file)
            self.W2 = numpy.load(file)
            self.b2 = numpy.load(file)

    def _fprop(self, x):
        x = numpy.atleast_2d(x)
        
        ha = numpy.dot(x, self.W1) + self.b1
        hs = self.hnlin(ha)
        oa = numpy.dot(hs, self.W2) + self.b2
        os = self.onlin(oa)
        
        return ha, hs, oa, os
    
    def test(self, x, y):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._test(x, y)

    def _test(self, x, y):
        if self.use_th:
            return self._m.test(x, y)
        else:
            return self.err(self._eval(x), y)

    def eval(self, x):
        x = numpy.atleast_2d(x)
        return self._eval(x)

    def _eval(self, x):
        if self.use_th:
            return self._m.eval(x)
        else:
            return self._fprop(x)[3]

    def grad(self, x, y, lmbd):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._grad(x, y, lmbd)

    def _grad(self, x, y, lmbd):
        C = dict()
        
        if self.use_th:
            C['W1'], C['W2'], C['b1'], C['b2'] = self._m.grad(x, y)
        else:
            ha, hs, oa, os = self._fprop(x)
            
            C['os'] = self.err._(os, y, self.err(os, y))
            C['oa'] = C['os'] * self.onlin._(oa, os)
            C['hs'] = numpy.dot(self.W2, C['oa'].T).T
            C['ha'] = C['hs'] * self.hnlin._(ha, hs)
            
            C['W2'] = numpy.dot(hs.T, C['oa']) + 2.0 * lmbd * self.W2
            C['b2'] = C['oa'].sum(axis=0)
            C['W1'] = numpy.dot(x.T, C['ha']) + 2.0 * lmbd * self.W1
            C['b1'] = C['ha'].reshape((x.shape[0], -1), order='F').sum(axis=0)

        return [layer(C['W1'], C['b1']), layer(C['W2'], C['b2'])]

    def _estim_grad(self, x, y, eps):
        Ge = dict()
        v = self._test(x, y)
        
        Ge['b1'] = numpy.empty(self.b1.shape, self.b1.dtype)
        for i, w in numpy.ndenumerate(self.b1):
            self.b1[i] += eps
            Ge['b1'][i] = (self._test(x,y) - v)/eps
            self.b1[i] = w

        Ge['b2'] = numpy.empty(self.b2.shape, self.b2.dtype)
        for i, w in numpy.ndenumerate(self.b2):
            self.b2[i] += eps
            Ge['b2'][i] = (self._test(x,y) - v)/eps
            self.b2[i] = w

        Ge['W1'] = numpy.empty(self.W1.shape, self.W1.dtype)
        for i, w in numpy.ndenumerate(self.W1):
            self.W1[i] += eps
            Ge['W1'][i] = (self._test(x,y) - v)/eps
            self.W1[i] = w

        Ge['W2'] = numpy.empty(self.W2.shape, self.W2.dtype)
        for i, w in numpy.ndenumerate(self.W2):
            self.W2[i] += eps
            Ge['W2'][i] = (self._test(x,y) - v)/eps
            self.W2[i] = w

        return [Ge['W1'], Ge['W2']], [Ge['b1'], Ge['b2']]

    def test_grad(self, x, y, verbose=True, eps=1e-4):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._test_grad(x, y, verbose, eps)

    def _test_grad(self, x, y, verbose, eps):
        GcW, Gcb = self._grad(x,y, lmbd=0.0)
        GeW, Geb = self._estim_grad(x, y, eps)

        rb1 = Geb[0]/Gcb[0]
        rb2 = Geb[1]/Gcb[1]
        rW1 = GeW[0]/GcW[0]
        rW2 = GeW[1]/GcW[1]

        if verbose:
            print "b1"
            print rb1
            print "W1"
            print rW1
            print "b2"
            print rb2
            print "W2"
            print rW2
        
        if rb1.max() < 1.01 and rb1.min() > 0.99 \
                and rb2.max() < 1.01 and rb2.min() > 0.99 \
                and rW1.max() < 1.01 and rW1.min() > 0.99 \
                and rW2.max() < 1.01 and rW2.min() > 0.99:
            if verbose: print "TOUS LES GRADIENTS SONT BONS!!!"
        else:
            raise ValueError("Wrong gradients detected")

    def _apply_dir(self, G, alpha=1.0):
        if self.use_th:
            self._m.W1 += alpha*G[0].W
            self._m.b1 += alpha*G[0].b
            self._m.W2 += alpha*G[1].W
            self._m.b2 += alpha*G[1].b
        else:
            self.W1 += alpha*G[0].W
            self.b1 += alpha*G[0].b
            self.W2 += alpha*G[1].W
            self.b2 += alpha*G[1].b

    def train_loop(self, trainer, epochs = 100):
	"""
	train_loop(trainer, epochs = 100)

        trainer: a instance of the Trainer class, describing a training method for the network.
	"""
        return self._train_loop(trainer, epochs)

    def _train_loop(self, trainer, epochs):
        for _ in xrange(epochs):
            trainer.epoch(self)

    layers = property(lambda self: [(self.W1, self.b1), (self.W2, self.b2)])
