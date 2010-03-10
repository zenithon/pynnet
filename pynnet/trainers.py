from base import *
from itertools import izip

__all__ = ['early_stopping', 'bprop', 'eval_net', 'minibatch_eval', 
           'minibatch_epoch']

def get_updates(nnet, err, alpha):
    gparams = theano.tensor.grad(nnet.error, nnet.params)
    return dict((p, p - gp*alpha) for p, gp in izip(nnet.params, gparams))

def early_stopping(train, valid, test, patience=10, patience_increase=2,
                   improvement_threshold=0.995, validation_frequency=5,
                   n_epochs=1000, verbose=True, time=True):
    best_params = None
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
        print "Best score obtained at epoch %i, score = %f", epoch, test_score
    if time:
        print "Time taken: %f min"%((end-start)/60.,)

def bprop(x, y, nnet, alpha=0.01):
    sx = theano.shared(value=x)
    sy = theano.shared(value=y)
    nnet.build(sx, sy)
    return theano.function([], nnet.cost, 
                           updates=get_updates(nnet, nnet.cost, alpha))

def eval_net(x, y, nnet):
    sx = theano.shared(value=x)
    sy = theano.shared(value=y)
    nnet.build(sx, sy)
    return theano.function([], nnet.cost)

class minibatch_eval(object):
    def __init__(self, dataiterf, batchsize, nnet, x, y):
        self.iterf = dataiterf
        self.batchsize = batchsize
        nnet.build(x, y)
        self.cost = theano.function([x, y], nnet.cost)
        
    def eval(self):
        return numpy.mean(self.cost(x, y) for x, y in 
                          self.iterf(self.batchsize))

class RepeatIter(object):
    def __init__(self, iterf, *args, **kwargs):
        self.iterf = iterf
        self.args = args
        self.kwargs = kwargs
        self.state = self.iterf(*self.args, **self.kwargs)

    def __iter__(self):
        return self
    
    def next(self):
        try:
            return self.state.next()
        except StopIteration:
            self.state = self.iterf(*self.args, **self.kwargs)
            return self.state.next()

class minibatch_epoch(object):
    def __init__(self, dataiterf, batchsize, nnet, x, y, alpha=0.01):
        nnet.build(x, y)
        self.cost = theano.function([x, y], nnet.cost,
                                    updates=get_updates(nnet, nnet.cost, alpha))
        self.it = RepeatIterf(dataiterf, batchsize)
        
    def eval(self):
        x, y = self.it.next()
        return self.cost(x, y)
    
