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

rnet = NNet([SimpleLayer(3, 6),
             RecurrentWrapper(SimpleLayer(12,6), (6,), name='rl'),
             SimpleLayer(6, 3)], error=errors.mse)

rl = rnet.get_layer('rl')

x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')

rnet.build(x, y)

eval = theano.function([x], rnet.output)
test = theano.function([x, y], rnet.cost)

# we can build the same network multiple times and it will use 
# the same parameters for each.  Here we use shared variables to go faster.

rnet.build(trainx, trainy)
train = theano.function([], rnet.cost, updates=trainers.get_updates(rnet.params, rnet.cost, 0.05))

# Since we didn't do any training (yet) the net has poor performance
print "Test at start:", test(testx, testy)

# clear the memory before training
rl.clear()

# Now to do some training
for _ in range(100):
    train()
    # we clear the memory between each training pass
    rl.clear()

print "Test after 100:", test(testx, testy)

rl.clear()

# Do some more training
for _ in range(900):
    train()
    rl.clear()

print "Test after 1000:", test(testx, testy)
rl.clear()

# Let's see what we get
print "Target:", testy
print "Output:", eval(testx)
