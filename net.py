import numpy

from base import *

from nlins import *
from errors import *

__all__ = ['NNet']

class NNet(BaseObject):
    def __init__(self, layers, nlins=(tanh, none), error=mse, alpha=0.01, lmbd=0.0, dtype=numpy.float32):
        
        lim = 1/numpy.sqrt(layers[0])
        makelayer = lambda i, o: (numpy.random.uniform(low=-lim, high=lim, size=(i, o)).astype(dtype), numpy.zeros(o, dtype=dtype))
        self.layers = [makelayer(i, o) for i, o in zip(layers, layers[1:])]
        if isinstance(nlins, (tuple, list)):
            if len(nlins) == len(layers) - 1:
                self.nlins = tuple(nlins)
            elif len(nlins) == 2:
                
                self.nlins = tuple(nlins[0:1] * (len(layers)-2)) + (nlins[1],)
            else:
                raise ValueError("not enough non-linearities for the number of layers")
        else:
            self.nlins = (nlins,)*(len(layers)-1)
        self.err = error
        self.alpha = alpha
        self.lmbd = lmbd

    def _save_(self, file):
        file.write('NN1')
        pickle.dump(len(self.layers), file)
        for l in self.layers:
            self._dump_layer(l, file)
        pickle.dump((self.nlins, self.err, self.alpha, self.lmbd), file)
        
    def _load_(self, file):
        s = file.read(3)
        if s != 'NN1':
            raise ValueError('File is not a SimpleNet save')
        l = pickle.load(file)
        self.layers = [self._load_layer(file) for _ in xrange(l)]
        self.nlins, self.err, self.alpha, self.lmbd = pickle.load(file)

    def _save_layer(self, l, file):
        numpy.save(file, l[0])
        numpy.save(file, l[1])

    def _load_layer(self, file):
        return (numpy.load(file), numpy.load(file))

    def fprop(self, x):
        x = numpy.atleast_2d(x)
        return self._fprop(x)

    def _fprop(self, x):
        acts = [None] * (len(self.nlins))
        outs = [None] * (len(self.nlins))
        outs[-1] = x
        for i in xrange(len(self.layers)):
            acts[i] = numpy.dot(outs[i-1], self.layers[i][0]) + self.layers[i][1]
            outs[i] = self.nlins[i](acts[i])

        return acts, outs

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
        return self._fprop(x)[1][-1]

    def grad(self, x, y):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._grad(x, y)

    def _grad(self, x, y, vals=None):
        if vals is None:
            acts, outs = self._fprop(x)
        else:
            acts, outs = vals
        
        Gouts = [None] * len(outs)
        Gacts = [None] * len(acts)
        
        Gouts[-1] = self.err._(outs[-1], y, self.err(outs[-1], y))
        Gacts[-1] = Gouts[-1] * self.nlins[-1]._(acts[-1], outs[-1])
        
        for i in xrange(-2, -len(outs) - 1, -1):
            Gouts[i] = numpy.dot(self.layers[i+1][0], Gacts[i+1].T).T
            Gacts[i] = Gouts[i] * self.nlins[i]._(acts[i], outs[i])
        
        GWs = [None] * len(self.layers)
        Gbs = [None] * len(self.layers)
        
        for i in xrange(-1,-len(self.layers), -1):
            GWs[i] = numpy.dot(outs[i-1].T, Gacts[i]) + 2.0 * self.lmbd * self.layers[i][0]
            Gbs[i] = Gacts[i].sum(axis=0)

        GWs[0] = numpy.dot(x.T, Gacts[0]) + 2.0 * self.lmbd * self.layers[0][0]
        Gbs[0] = Gacts[0].sum(axis=0)

        return Gacts, Gouts, GWs, Gbs
    
#        G['W2'] = numpy.dot(hs.T, G['oa']) + 2.0 * self.lmbd * self.W2
#        G['b2'] = G['oa'].sum(axis=0)
#        G['W1'] = numpy.dot(x.T, G['ha']) + 2.0 * self.lmbd * self.W1
#        G['b1'] = G['ha'].reshape((x.shape[0], -1), order='F').sum(axis=0)
    
    def _estim_grad(self, x, y, eps):
        v = self._test(x, y)

        GWs = [numpy.empty(l[0].shape, l[0].dtype) for l in self.layers]
        Gbs = [numpy.empty(l[1].shape, l[1].dtype) for l in self.layers]
        
        for l, GW, Gb in zip(self.layers, GWs, Gbs):
            W = l[0]
            b = l[1]

            for i, w in numpy.ndenumerate(W):
                W[i] += eps
                GW[i] = (self._test(x,y) - v)/eps
                W[i] = w

            for i, w in numpy.ndenumerate(b):
                b[i] += eps
                Gb[i] = (self._test(x,y) - v)/eps
                b[i] = w

        return GWs, Gbs

    def test_grad(self, x, y, verbose=True, eps=1e-4):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._test_grad(x, y, verbose, eps)

    def _test_grad(self, x, y, verbose, eps):
        lmbd = self.lmbd  
        self.lmbd = 0.0  # compute gradient without lambda
        Gacts, Gouts, GcWs, Gcbs = self._grad(x,y)
        GeWs, Gebs = self._estim_grad(x, y, eps)
        self.lmbd = lmbd

        wrong = False

        for GWc, GWe in zip(GcWs, GeWs):
            rW = GWc/GWe
            if verbose:
                print rW
            if rW.max() > 1.01 or rW.min() < 0.99:
                wrong = True

        for Gbc, Gbe in zip(Gcbs, Gebs):
            rb = Gbc/Gbe
            if verbose:
                print rb
            if rb.max() > 1.01 or rb.min() < 0.99:
                wrong = True

        if wrong:
            raise ValueError("Wrong gradient(s) detected")
    
    def epoch_bprop(self, x, y):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._epoch_bprop(x, y)

    def _epoch_bprop(self, x, y):
        Gacts, Gouts, GWs, Gbs = self._grad(x, y)

        for l, GW, Gb in zip(self.layers, GWs, Gbs):
            W, b = l
            W -= self.alpha * GW
            b -= self.alpha * Gb

    def _epoch_steep(self, x, y):
        Gacts, Gouts, GWs, Gbs = self._grad(x, y)

        self.dWs = [-GW for GW in GWs]
        self.dbs = [-Gb for Gb in Gbs]

        self._linesearch(self, y, x)

    def _linesearch(self, x, y, maxloops=None):
        best_err = self._test(x, y)
        cur_err = 0
        alpha = 0
        if maxloops is None:
            maxloops = int(1/self.alpha)

        layers = [(l[0].copy(), l[1].copy()) for l in self.layers]

        for i in xrange(maxloops):
            alpha = i * self.alpha
            for l, ol, dW, db in zip(self.layers, layers, self.dWs, self.dbs):
                W, b = l
                oW, ob = ol
                W = oW + alpha * dW
                b = ob + alpha * db
            cur_err = self._test(x, y)
            if cur_err < best_err:
                best_err = cur_err
            else:
                alpha = (i-1)*self.alpha
                break        

        for l, ol, dW, db in zip(self.layers, layers, self.dWs, self.dbs):
            W, b = l
            oW, ob = ol
            W = oW + alpha * dW
            b = ob + alpha * db

    def train_loop(self, x, y, epochs = 100):
        x = numpy.atleast_2d(x)
        y = numpy.atleast_2d(y)
        return self._train_loop(x, y, epochs)

    def _train_loop(self, x, y, epochs):
        for _ in xrange(epochs):
            self._epoch_bprop(x, y)

