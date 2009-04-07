import numpy

from base import *

from nlins import *
from errors import *

__all__ = ['SimpleNet']

class SimpleNet(BaseObject):
    def __init__(self, ninputs, nhidden, noutputs, hnlin=tanh, onlin=none, error=mse, alpha=0.01, lmbd=0.0, dtype=numpy.float32):
        self.W1 = numpy.random.uniform(low=-1/numpy.sqrt(ninputs), high=1/numpy.sqrt(ninputs), size=(ninputs, nhidden)).astype(dtype)
        self.W2 = numpy.random.uniform(low=-1/numpy.sqrt(ninputs), high=1/numpy.sqrt(ninputs), size=(nhidden, noutputs)).astype(dtype)
        self.b1 = numpy.zeros(nhidden, dtype=dtype)
        self.b2 = numpy.zeros(noutputs, dtype=dtype)
        self.hnlin = hnlin
        self.onlin = onlin
        self.err = error
        self.alpha = alpha
        self.lmbd = lmbd

    def _save_(self, file):
        file.write('SN1')
        numpy.save(file, self.W1)
        numpy.save(file, self.b1)
        numpy.save(file, self.W2)
        numpy.save(file, self.b2)
        pickle.dump((self.hnlin, self.onlin, self.err, self.alpha, self.lmbd), file)
        
    def _load_(self, file):
        
        s = file.read(3)
        if s != 'SN1':
            raise ValueError('Wrong fromat for SimpleNet in file')
        print file.tell()
        self.W1 = numpy.load(file)
        self.b1 = numpy.load(file)
        self.W2 = numpy.load(file)
        self.b2 = numpy.load(file)
        self.hnlin, self.onlin, self.err, self.alpha, self.lmbd = pickle.load(file)

    def fprop(self, x):
        x = numpy.atleast_2d(x)
        return self._fprop(x)

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
        return self.err(self._eval(x), y)

    def eval(self, x):
        x = numpy.atleast_2d(x)
        return self._eval(x)

    def _eval(self, x):
        return self._fprop(x)[3]

    def grad(self, x, y):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._grad(x, y)

    def _grad(self, x, y, vals=None):
        if vals is None:
            ha, hs, oa, os = self._fprop(x)
        else:
            ha, hs, oa, os = vals
            
        C = dict()
        
        C['os'] = self.err._(os, y, self.err(os, y))
        C['oa'] = C['os'] * self.onlin._(oa, os)
        C['hs'] = numpy.dot(self.W2, C['oa'].T).T
        C['ha'] = C['hs'] * self.hnlin._(ha, hs)
        
        C['W2'] = numpy.dot(hs.T, C['oa']) + 2.0 * self.lmbd * self.W2
        C['b2'] = C['oa'].sum(axis=0)
        C['W1'] = numpy.dot(x.T, C['ha']) + 2.0 * self.lmbd * self.W1
        C['b1'] = C['ha'].reshape((x.shape[0], -1), order='F').sum(axis=0)

        return C

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

        return Ge

    def test_grad(self, x, y, verbose=True, eps=1e-4):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._test_grad(self, x, y, verbose, eps)

    def _test_grad(self, x, y, verbose, eps):
        lmbd = self.lmbd
        self.lmbd = 0.0
        Gc = self._grad(x,y)
        Ge = self._estim_grad(x, y, eps)
        self.lmbd = lmbd
        
        rb1 = Ge['b1']/Gc['b1']
        rb2 = Ge['b2']/Gc['b2']
        rW1 = Ge['W1']/Gc['W1']
        rW2 = Ge['W2']/Gc['W2']

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

    def epoch_bprop(self, x, y):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._epoch_bprop(x, y)

    def _epoch_bprop(self, x, y):
        G = self._grad(x, y)

        self.W1 -= self.alpha * G['W1']
        self.b1 -= self.alpha * G['b1']
        self.W2 -= self.alpha * G['W2']
        self.b2 -= self.alpha * G['b2']

    def train_loop(self, x, y, epochs = 100):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._train_loop(x, y, epochs)

    def _train_loop(self, x, y, epochs):
        for _ in xrange(epochs):
            self._epoch_bprop(x, y)

