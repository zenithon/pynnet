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
    def __init__(self, x, y, lmbd=0.0, alpha=0.01):
        r"""
        Standard back-propagation trainer.

        Parameters:
        x -- inputs
        y -- targets
        lmbd -- (default: 0.0) the weight decay term
        alpha -- (default: 0.01) the gradient step to take at each iteration
        """
        Trainer.__init__(self, x, y)
        self.alpha = alpha
        self.lmbd = lmbd
    
    def epoch(self, nnet):
        G = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        nnet._apply_dir(G, -self.alpha)

def _linesearch(nnet, x, y, G, maxalpha, eps):
        r"""
        Cheap line search.  Could be improved.
        """
        near = 0.0
        far = maxalpha
        where = maxalpha
        near_err = nnet._test(x, y)
        nnet._apply_dir(G, -where)
        far_err = nnet._test(x, y)
        
        while far-near > eps:
            here = (near+far)/2
            nnet._apply_dir(G, where-here)
            where = here
            here_err = nnet._test(x, y)
            if here_err < near_err:
                near = here
                near_err = here_err
            else:
                far = here
                far_err = here_err

class steepest(Trainer):
    def __init__(self, x, y, lmbd=0.0, eps=0.01, maxalpha=20.0):
        r"""
        Steepest descent trainer.

        Parameters:
        x -- inputs
        y -- targets
        lmbd -- (default: 0.0) the weight decay term
        eps -- (default: 0.01) the tolerance level for the minimum
        maxalpha -- (default: 20.0) the farthest along the gradient to look
        """
        Trainer.__init__(self, x,y)
        self.lmbd = lmbd
        self.eps = eps
        self.maxalpha = maxalpha
    
    def epoch(self, nnet):
        G = nnet._grad(self.x, self.y, lmbd=self.lmbd)
        _linesearch(nnet, self.x, self.y, self.d, maxalpha=self.maxalpha, eps=self.eps)

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
    def __init__(self, x, y, lmbd=0.0, eps=0.01, maxalpha=10, beta_method='PR'):
        r"""
        Conjugate gradient trainer.

        Parameters:
        x -- inputs
        y -- targets
        lmbd -- (default: 0.0) the weight decay term
        eps -- (default: 0.01) the tolerance level for the minimum
        maxalpha -- (default: 20.0) the farthest along the gradient to look
        beta_method -- (default: 'PR') one of 'FR', 'PR', and 'HS'.  The formula to use to compute the beta term.
        """
        Trainer.__init__(self, x, y)
        self.lmbd = lmbd
        self.eps = eps
        self.maxalpha = maxalpha
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
        
        _linesearch(nnet, self.x, self.y, self.d, maxalpha=self.maxalpha, eps=self.eps)
