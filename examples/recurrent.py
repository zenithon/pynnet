# An example recurrent network.

# RUNTIME: about 15 mins

# The task is, given a sequence of symbols, predict the next one.
# The symbols are represented using three bits.  They follow a pattern
# where one symbol with a 0 first is followed by 0-3 symbols with a 1 first.
# the last two bits encode the number of symbols remaining until the next
# one with a 0 first

# Note that the error does not go down that much since the network
# cannot actually predict exactly which [0, x, x] symbol is next.  It
# should predict the 0 correctly though.

from pynnet import *
from pynnet.nodes import errors
from pynnet.training import get_updates
import numpy, theano

# we define the datasets
def genseq(i):
    res = [[0, (i&2)/2, i&1]]
    while i > 0:
        i -= 1
        res.append([1, (i&2)/2, i&1])
    return res
trainseq = theano.shared(numpy.asarray(sum(map(genseq, numpy.random.randint(low=0, high=4, size=(600,))), []), dtype=theano.config.floatX))
trainx = trainseq
trainy = theano.shared(numpy.concatenate([trainseq.get_value()[1:], trainseq.get_value()[:1]], axis=0))

testseq = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0]]
testx = testseq
testy = testseq[1:] + testseq[:1]

x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')

map_in = SimpleNode(x, 3, 6)
rn = RecurrentWrapper(map_in, lambda x_n: SimpleNode(x_n, 12, 6),
                      outshp=(6,), name='rl')
out = SimpleNode(rn, 6, 3)

cost = errors.mse(out, y)

eval_sub = theano.function([x], out.output)
def eval(x):
    res = eval_sub(x)
    # we clear the memory of the recurrent layer between input
    # sequences because otherwise the network starts in an unknown
    # state.
    rn.clear()
    return res

test_sub = theano.function([x, y], cost.output)
def test(x, y):
    res = test_sub(x, y)
    # clear here too
    rn.clear()
    return res

train_sub = theano.function([], cost.output,
                            updates=get_updates(cost.params,cost.output,0.05),
                            givens={x: trainx, y: trainy})

def train():
    res = train_sub()
    rn.clear()
    return res

# Since we didn't do any training (yet) the net has poor performance
print "Test at start:", test(testx, testy)

# Now to do some training
for _ in range(100):
    train()

print "Test after 100:", test(testx, testy)

# Do some more training
for _ in range(900):
    train()

print "Test after 1000:", test(testx, testy)

# Let's see what we get
print "Target:", testy
print "Output:", eval(testx)
