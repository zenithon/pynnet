from base import *

__all__ = ['bprop', 'steepest', 'conj']

class Trainer(object):
    def __init__(self, x, y):
        self.x = numpy.atleast_2d(x)
        self.y = numpy.atleast_2d(y)

    def reset(self):
        r"""
        Resets the internal state of the trainer.

        Can be used to change the net being trained or to simply
        restart training anew.

        If you have nothing to do just omit it, the default
        implementation does just that: nothing.
        """
        pass

    def epoch(self, nnet):
        r"""
        Computes what is needed to do an epoch.

        Parameters:
        nnet -- the neural net to train

        See the documentation of `AbstractNet` to see what methods you
        can call to interact with the network.

        This method is also responsible to apply the changes to the
        network (usually using the `AbstractNet._apply_dir()` API) to
        make it evolve toward the computed direction.

        The computations involved should not do more than one pass
        over all the training data.
        """
        raise NotImplementedError

class bprop(Trainer):
    def __init__(self, x, y, alpha=0.01, lmbd=0.0):
        Trainer.__init__(self, x, y)
        self.alpha = alpha
        self.lmbd = lmbd
    
    def epoch(self, nnet):
        G = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        nnet._apply_dir(G, -self.alpha)

def _linesearch(nnet, x, y, G, step):
        r"""
        Cheap line search.  Could be improved.
        """
        best_err = nnet._test(x, y)
        cur_err = 0
        alpha = 0
        maxtries = int(1/step)

        for i in xrange(1, maxtries):
            nnet._apply_dir(G, -step)
            cur_err = nnet._test(x, y)
            if cur_err < best_err:
                best_err = cur_err
            else:
                nnet._apply_dir(G, step)
                break

class steepest(Trainer):
    def __init__(self, x, y, step=0.01, lmbd=0.0):
        Trainer.__init__(self, x,y)
        self.step = step
        self.lmbd = lmbd
    
    def epoch(self, nnet):
        G = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        _linesearch(nnet, self.x, self.y, G, self.step)

def beta_FR(G, oG, d):
    return sum((Gl.W**2).sum()+(Gl.b**2).sum() for Gl in G) / \
        sum((oGl.W**2).sum()+(oGl.b**2).sum() for oGl in oG)

def beta_PR(G, oG, d):
    return sum((Gl.W*(Gl.W-oGl.W)).sum()+(Gl.b*(Gl.b-oGl.b)).sum() for Gl, oGl in zip(G, oG)) / \
        sum((oGl.W**2).sum()+(oGl.b**2).sum() for oGl in oG)

def beta_HS(G, oG, d):
    return sum((Gl.W*(Gl.W-oGl.W)).sum()+(Gl.b*(Gl.b-oGl.b)).sum() for Gl, oGl in zip(G, oG)) / \
        sum((dl.W*(Gl.W-oGl.W)).sum()+(dl.b*(Gl.b-oGl.b)).sum() for Gl, oGl, dl in zip(G, oG, d))

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
        del self.d
        del self.G

    def epoch(self, nnet):
        G = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        try:
            beta = max(self.beta(G, self.G, self.d), 0)
            d = [layer(Gl.W + beta*dl.W, Gl.b + beta*dl.b) for Gl, dl in zip(G, self.d)]
            self.G = G
            self.d = d
        except AttributeError:
            self.d = self.G = G
        
        _linesearch(nnet, self.x, self.y, self.d, self.step)
