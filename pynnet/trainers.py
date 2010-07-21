from base import *
from itertools import izip
import time

__all__ = ['early_stopping', 'get_updates']

def get_updates(params, err, alpha):
    r"""
    Returns a dictionary of updates suitable for theano.function.

    The updates are what would be done in one step of backprop over
    parameters `params` with error function `err` and step `alpha`.

    A typical call of this function over a built network looks like
    this:
    
       updts = get_updates(net.params, net.cost, 0.01)
    
    Tests:
    >>> W = theano.shared(numpy.random.random((12, 8)), name='W')
    >>> b = theano.shared(numpy.random.random((8,)), name='b')
    >>> import theano.tensor as T
    >>> xs = T.dmatrix('x')
    >>> ys = T.dmatrix('y')
    >>> x = numpy.random.random((50, 12))
    >>> y = numpy.random.random((50, 8))
    >>> err = T.mean((T.tanh(T.dot(xs, W)+b) - ys)**2)
    >>> up = get_updates([W, b], err, 0.125)
    >>> f = theano.function([xs, ys], err, updates=up)
    >>> f(x, y) > f(x, y)
    True
    """
    a = theano.tensor.TensorConstant(theano.tensor.fscalar, alpha)
    gparams = theano.tensor.grad(err, params)
    return dict((p, p - gp*a) for p, gp in izip(params, gparams))

def early_stopping(train, valid, test, patience=10, patience_increase=2,
                   improvement_treshold=0.995, validation_frequency=5,
                   n_epochs=1000, verbose=True, print_time=True):
    r"""
    An implementation of the early stopping algorithm.

    If you don't have any specific needs you can use this as-is, but
    otherwise you can copy this code in your own files and modify it
    according to whatever you are trying to do.

    :notests:
    """
    best_valid_score = float('inf')
    best_iter = 0
    test_score = 0
    
    start = time.clock()
    for epoch in xrange(n_epochs):
        cost = train()
        if verbose == 2:
            print "epoch:", epoch, "train cost:", cost
        if epoch % validation_frequency == 0:
            valid_score = valid()
            if verbose:
                print 'epoch %i, valid error %f'%(epoch, valid_score)
            if valid_score < best_valid_score:
                if valid_score < best_valid_score * improvement_treshold:
                    patience = max(patience, epoch*patience_increase)
                    best_valid_score = valid_score
                    
                    if verbose:
                        test_score = test()
                        print 'epoch %i, test error %f'%(epoch, test_score)

        if patience < epoch:
            break
    end = time.clock()
    if verbose:
        print "Best score obtained at epoch %i, score = %f, valid = %f"%(epoch, test_score, best_valid)
    if print_time:
        print "Time taken: %f min"%((end-start)/60.,)
    return best_iter, best_valid, test_score
