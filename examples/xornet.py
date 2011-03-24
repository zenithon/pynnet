# This is a test file for simplenet doing a xor regression test
# It is an experiment in which we try to make a network learn the
# patterns of input and output for the xor function.

# RUNTIME: about 30 seconds

import theano
from pynnet import *
from pynnet.nodes import errors
from pynnet.training import get_updates

sx = theano.tensor.matrix('x')
sy = theano.tensor.matrix('y')

# We initialize an MLP with one hidden layer of two units.
h = SimpleNode(sx, 2, 2)
out = SimpleNode(h, 2, 1)
cost = errors.mse(out, sy)

# We can build functions from expressions to use our network
eval = theano.function([sx], out.output)
test = theano.function([sx, sy], cost.output)
train = theano.function([sx, sy], cost.output, 
                        updates=get_updates(cost.params, cost.output, 0.01))


x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# At the start the error is terrible
print "Error at start:", test(x, y)

# So we train a little
for i in range(10):
    train(x, y)

# Now the error should be a bit better
print "Error after 10:", test(x, y)

# And we train more
for i in range(10000):
    train(x, y)

# Now the error should be really low
print "Error after 10000:", test(x, y)

# We can look at the actual output of the network
print "Output for [0, 1]:", eval([[0, 1]])
