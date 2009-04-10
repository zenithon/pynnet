from base import *

__all__ = ['bprop', 'steepest', 'conj']

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

def _linesearch(nnet, x, y, GWs, Gbs, step):
        r"""
        Cheap line search.  Could be improved.
        """
        best_err = nnet._test(x, y)
        cur_err = 0
        alpha = 0
        maxtries = int(1/step)

        for i in xrange(1, maxtries):
            nnet._apply_dir(GWs, Gbs, -step)
            cur_err = nnet._test(x, y)
            if cur_err < best_err:
                best_err = cur_err
            else:
                nnet._apply_dir(GWs, Gbs, step)
                break

class steepest(Trainer):
    def __init__(self, x, y, step=0.01, lmbd=0.0):
        Trainer.__init__(self, x,y)
        self.step = step
        self.lmbd = lmbd
    
    def epoch(self, nnet):
        GWs, Gbs = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        _linesearch(nnet, self.x, self.y, GWs, Gbs, self.step)

def beta_FR(GWs, Gbs, oGWs, oGbs, dWs, dbs):
    return sum((GW**2).sum()+(Gb**2).sum() for GW, Gb in zip(GWs, Gbs)) / \
        sum((oGW**2).sum()+(oGb**2).sum() for oGW, oGb in zip(oGWs, oGbs))

def beta_PR(GWs, Gbs, oGWs, oGbs, dWs, dbs):
    return sum((GW*(GW-oGW)).sum()+(Gb*(Gb-oGb)).sum() for GW, Gb, oGW, oGb in zip(GWs, Gbs, oGWs, oGbs)) / \
        sum((oGW**2).sum()+(oGb**2).sum() for oGW, oGb in zip(oGWs, oGbs))

def beta_HS(GWs, Gbs, oGWs, oGbs, dWs, dbs):
    return sum((GW*(GW-oGW)).sum()+(Gb*(Gb-oGb)).sum() for GW, Gb, oGW, oGb in zip(GWs, Gbs, oGWs, oGbs)) / \
        sum((dW*(GW-oGW)).sum()+(db*(Gb-oGb)).sum() for GW, Gb, oGW, oGb, dW, db in zip(GWs, Gbs, oGWs, oGbs, dWs, dbs))

class conj(Trainer):
    def __init__(self, x, y, step=0.01, lmbd=0.0, beta_method='PR'):
        Trainer.__init__(self, x, y)
        self.step = step
        self.lmbd = lmbd
        if beta_method == 'FR':
            self.beta = beta_FR
        elif beta_method == 'PR':
            self.beta = beta_PR
        elif beta_method == 'HS':
            self.beta = beta_HS
        else:
            raise ValueError('Unknown beta method "%s"'%(beta_method,))

    def reset(self):
        del self.dWs
        del self.dbs
        del self.GWs
        del self.Gbs

    def epoch(self, nnet):
        GWs, Gbs = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        try:
            beta = max(self.beta(GWs, Gbs, self.GWs, self.Gbs, self.dWs, self.dbs), 0)
            dWs = [GW + beta*dW for GW, dW in zip(GWs, self.dWs)]
            dbs = [Gb + beta*db for Gb, db in zip(Gbs, self.dbs)]
            self.GWs = GWs
            self.Gbs = Gbs
            self.dWs = dWs
            self.dbs = dbs
        except AttributeError:
            self.dWs = self.GWs = GWs
            self.dbs = self.Gbs = Gbs
        
        _linesearch(nnet, self.x, self.y, self.dWs, self.dbs, self.step)
