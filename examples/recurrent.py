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
trainy = theano.shared(numpy.concatenate([trainseq.value[1:], trainseq.value[:1]], axis=0))

testseq = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0]]
testx = testseq
testy = testseq[1:] + testseq[:1]

x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')

map_in = SimpleNode(x, 3, 6)
rn = RecurrentWrapper(map_in, lambda x_n: SimpleNode(x_n, 12, 6),
                      outshp=(6,), name='rl')
out = SimpleNode(rn, 6, 3)

rnet = NNet(out, y, error=errors.mse)

eval = theano.function([x], rnet.output)
test = theano.function([x, y], rnet.cost)

train = theano.function([], rnet.cost, updates=trainers.get_updates(rnet.params, rnet.cost, 0.05), givens={x: trainx, y: trainy})

# Since we didn't do any training (yet) the net has poor performance
print "Test at start:", test(testx, testy)

# clear the memory before training
rn.clear()

# Now to do some training
for _ in range(100):
    train()
    # we clear the memory between each training pass
    rn.clear()

print "Test after 100:", test(testx, testy)

rn.clear()

# Do some more training
for _ in range(900):
    train()
    rn.clear()

print "Test after 1000:", test(testx, testy)
rn.clear()

# Let's see what we get
print "Target:", testy
print "Output:", eval(testx)
