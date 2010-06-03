from base import *
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pynnet.nlins import sigmoid
from pynnet.errors import cross_entropy

from itertools import izip

__all__ = ['RMBLayer']

class RBMLayer(BaseLayer):
    r"""
    Restricted Boltzmann Machine layer.

    Examples:
    >>> r = RBMLayer(10, 8)
    >>> r = RBMLayer(12, 12, sampled=True)
    >>> r = RBMLayer(4, 3, persistent=True)
    
    Attributes:
    `W` -- (shared matrix, read-only) Connection weights matrix.
    `b` -- (shared vector, read-only) Hidden bias vector.
    `c` -- (shared vector, read-only) Visible bias vector.
    `k` -- (integer, read-write) Number of steps of CD or PCD.
    `sampled` -- (bool, read-write) Whether the output is the
                 probabilites vector a sample of the distribution.
    `persistent` -- (bool, read-write) If True, use PCD, otherwise use
                    CD for pretraining. (NOT WORKING, yet)
    """
    def __init__(self, n_in, n_out, rng=numpy.random, k=1, sampled=False,
                 persistent=False, dtype=theano.config.floatX, name=None):
        r"""
        Tests:
        >>> r = RBMLayer(10, 12)
        >>> r.W.value.shape
        (10, 12)
        >>> r.b.value.shape
        (12,)
        >>> r.c.value.shape
        (10,)
        """
        BaseLayer.__init__(self, name)
        w_range = 4.*numpy.sqrt(6./(n_in+n_out))
        W_values = rng.uniform(low=-w_range, high=w_range,
                               size=(n_in, n_out)).astype(dtype)
        self.W = theano.shared(value=W_values, name='W')
        b_values = numpy.zeros((n_out,), dtype=dtype)
        self.b = theano.shared(value=b_values, name='b')
        c_values = numpy.zeros((n_in,), dtype=dtype)
        self.c = theano.shared(value=c_values, name='c')
        self.k = k
        self.sampled = sampled
        self.persistent = False # FIXME (cannot use PCD)
        self.dtype = dtype
        
    def _prop(self, v, W, b):
        r"""
        :nodoc:

        Tests:
        >>> r = RBMLayer(3, 2)
        >>> x = T.matrix('x')
        >>> pre_s, s = r._prop(x, r.W, r.b)
        >>> theano.pp(pre_s)
        '((x \\dot W) + b)'
        >>> theano.pp(s)
        'sigmoid(((x \\dot W) + b))'
        >>> f = theano.function([x], [s, pre_s])
        """
        pre_sigm = T.dot(v, W) + b
        return pre_sigm, sigmoid(pre_sigm)
        
    def _sample(self, x, W, b):
        r"""
        :nodoc:
        
        Tests:
        >>> r = RBMLayer(5, 4)
        >>> x = T.matrix('x')
        >>> pre, post, y = r._sample(x, r.W, r.b)
        >>> theano.pp(y)
        'RandomFunction{binomial}(<RandomStateType>, sigmoid(((x \\dot W) + b)).shape, 1, sigmoid(((x \\dot W) + b)))'
        >>> f = theano.function([x], y)
        """
        pre_sigm, sigm = self._prop(x, W, b)
        y = RandomStreams().binomial(size=sigm.shape, n=1, p=sigm,
                                   dtype=self.dtype)
        return pre_sigm, sigm, y

    def _gibbs_hvh(self, h):
        r"""
        :nodoc:
        
        Tests:
        >>> r = RBMLayer(5, 4)
        >>> x = T.matrix('x')
        >>> _, _, h = r._sample(x, r.W, r.b)
        >>> pv1, v1, v1s, ph1, h1, h1s = r._gibbs_hvh(h)
        >>> f = theano.function([x], pv1)
        >>> f = theano.function([x], v1)
        >>> f = theano.function([x], v1s)
        >>> f = theano.function([x], ph1)
        >>> f = theano.function([x], h1)
        >>> f = theano.function([x], h1s)
        """
        pre_sigm_v1, v1, v1_sample = self._sample(h, self.W.T, self.c)
        pre_sigm_h1, h1, h1_sample = self._sample(v1_sample, self.W, self.b)
        return pre_sigm_v1, v1, v1_sample, pre_sigm_h1, h1, h1_sample
    
    def free_energy(self, v_sample):
        r"""
        Returns the free energy expression for sample `v_sample`.
        
        Tests:
        >>> r = RBMLayer(4, 3, dtype='float32')
        >>> x = T.matrix('sample')
        >>> e = r.free_energy(x)
        >>> theano.pp(e)
        '(sum(((-Sum{1}(log((1 + exp(((sample \\dot W) + b)))))) - (sample \\dot c))) / float32(((-Sum{1}(log((1 + exp(((sample \\dot W) + b)))))) - (sample \\dot c)).shape)[0])'
        >>> f = theano.function([x], e)
        """
        wx_b = T.dot(v_sample, self.W) + self.b
        vbias_term = T.dot(v_sample, self.c)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)), axis = 1)
        return T.mean(-hidden_term - vbias_term)
    
    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> r = RBMLayer(3, 2, sampled=False, dtype='float32')
        >>> x = T.fmatrix('x')
        >>> r.build(x, input_shape=(4, 3))
        >>> r.params
        [W, b]
        >>> r.input
        x
        >>> r.output_shape
        (4, 2)
        >>> theano.pp(r.output)
        'sigmoid(((x \\dot W) + b))'
        >>> f = theano.function([x], r.output)
        >>> y = f(numpy.random.random((4, 3)))
        >>> y.shape
        (4, 2)
        >>> y.dtype
        dtype('float32')
        >>> y.max() <= 1.
        True
        >>> y.min() >= 0.
        True
        
        >>> r = RBMLayer(3, 2, sampled=True, dtype='float32')
        >>> x = T.fmatrix('x')
        >>> r.build(x, input_shape=(4, 3))
        >>> r.output_shape
        (4, 2)
        >>> theano.pp(r.output)
        'RandomFunction{binomial}(<RandomStateType>, sigmoid(((x \\dot W) + b)).shape, 1, sigmoid(((x \\dot W) + b)))'
        >>> f = theano.function([x], r.output)
        >>> y = f(numpy.random.random((4, 3)))
        >>> y.shape
        (4, 2)
        >>> y.dtype
        dtype('float32')
        >>> ((y == 0.) | (y == 1.)).all()
        True
        """
        self.input = input

        pre_sigm_ph, ph, ph_sample = self._sample(self.input, self.W, self.b)

        if self.sampled:
            self.output = ph_sample
        else:
            self.output = ph

        if input_shape is not None:
            self.output_shape = (input_shape[0], self.W.value.shape[1])

        if self.persistent: # FIXME (cannot use PCD)
            pass
            #chain_start = self.persistent
        else:
            chain_start = ph_sample
            
        [pre_sigm_nvs, nv, nv_samples, pre_sigm_nhs, nh, nh_samples], self.rbm_updates = \
            theano.scan(self._gibbs_hvh, 
                        outputs_info=[None,None,None,None,None,chain_start],
                        n_steps=self.k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        target = nv_samples[-1]
        
        self.rbm_target = target
        self.rbm_cost = self.free_energy(self.input) - self.free_energy(target)
        self.rbm_params = [self.W, self.b, self.c]
        self.params = [self.W, self.b]

        if self.persistent:
            # FIXME (cannot use PCD)
            # Note that this works only if persistent is a shared variable
            updates[self.persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            self.cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            self.cost = cross_entropy(sigmoid(pre_sigm_nvs[-1]), self.input)
    
    def pretrain_helper(self, lr=0.1):
        r"""
        Returns the cost function and the updates that should be used for
        pretraining.

        You *must* use this function of the equivalent to get the
        proper updates to use for pretraining.  For finetuning, you
        can use normal procedures.

        Tests:
        >>> r = RBMLayer(3, 2)
        >>> x = T.matrix('x')
        >>> r.build(x)
        >>> cost, updts = r.pretrain_helper()
        >>> sorted(map(repr, updts.keys()))
        ['<RandomStateType>', '<RandomStateType>', '<RandomStateType>', 'W', 'b', 'c']
        >>> f = theano.function([x], r.cost, updates=updts)
        >>> xr = numpy.random.random((5, 3))
        >>> c1 = f(xr)

        # cost is weird: sometimes goes up, sometimes goes down 
        # (maybe due to sampling) so don't run tests below
        #>>> c2 = f(xr)
        #>>> c1 > c2
        #True
        """
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(self.rbm_cost, self.rbm_params,
                         consider_constant=[self.rbm_target])
        a = T.cast(lr, dtype=self.dtype)
        updts = dict((p, p-gp*a) for p, gp in izip(self.rbm_params, gparams))
        updts.update(self.rbm_updates)
        return self.rbm_cost, updts
