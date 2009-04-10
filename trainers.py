from base import *

__all__ = ['bprop']

class Trainer(BaseObject):
    def __init__(self, x, y):
        self.x = numpy.atleast_2d(x)
        self.y = numpy.atleast_2d(y)

    def _save_(self, file):
        raise NotImplementedError

    def reset(self):
        r"""
        Resets the internal state of the trainer to zero.

        Can be used to change the net being trained or to simply restart training anew.

        If you have nothing to do just omit it, the default implementation does just that: nothing.
        """
        pass

    def epoch(self, nnet):
        r"""
        Computes (v, d) such that adding v*d[i] for each i in the coefficients of the network should improve the error.
        
        In a simple back-propagation algorithm, this would return (-alpha, grad).
        """
        raise NotImplementedError

class bprop(Trainer):
    def __init__(self, x, y, alpha=0.01, lmbd=0.0):
        Trainer.__init__(self, x, y)
        self.alpha = alpha
        self.lmbd = lmbd
    
    def epoch(self, nnet):
        GWs, Gbs = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        nnet._apply_dir(GWs, Gbs, -self.alpha)

class steepest(Trainer):
    def __init__(self, x, y, alpha=0.01, lmbd=0.0):
        Trainer.__init__(self, x,y)
        self.alpha = alpha
        self.lmbd = lmbd
    
    def epoch(self, nnet):
        GWs, Gbs = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        
        self.dWs = [-GW for GW in GWs]
        self.dbs = [-Gb for Gb in Gbs]
        
        self._linesearch(nnet)
    
    def _linesearch(self, nnet):
        r"""
        Cheap line search.  Could be improved.
        """
        best_err = nnet._test(self.x, self.y)
        cur_err = 0
        alpha = 0
        maxloops = int(1/self.alpha)

        layers = [(l[0].copy(), l[1].copy()) for l in nnet.layers]

        for i in xrange(1, maxloops):
            alpha = i * self.alpha
            print "iter:", i
            print alpha
            print nnet.W1
            for (W, b), (oW, ob), dW, db in zip(nnet.layers, layers, self.dWs, self.dbs):
                print dW
                W = oW + alpha * dW
                b = ob + alpha * db
            print nnet.W1
            cur_err = nnet._test(self.x, self.y)
            print best_err, cur_err
            if cur_err < best_err:
                best_err = cur_err
            else:
                alpha = (i-1)*self.alpha
                break

        print alpha

        for l, ol, dW, db in zip(nnet.layers, layers, self.dWs, self.dbs):
            W, b = l
            oW, ob = ol
            W = oW + alpha * dW
            b = ob + alpha * db

class conj(Trainer):
    pass
